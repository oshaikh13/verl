from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Iterable, Sequence

try:
    import cloudpickle as _pickle
except ImportError:  # pragma: no cover - fallback if cloudpickle unavailable
    import pickle as _pickle  # type: ignore

try:
    import ray
except ImportError:  # pragma: no cover - ray is optional
    ray = None  # type: ignore

from .temporal_bm25 import InMemoryBM25Temporal

logger = logging.getLogger(__name__)


def _ray_available() -> bool:
    return ray is not None and ray.is_initialized()


def _add_rows_to_retriever(
    retriever: InMemoryBM25Temporal,
    rows: Sequence[dict],
    default_namespace: str,
    namespace: str | None,
    visible_delay: int,
) -> None:
    base_ns = namespace or default_namespace
    for row in rows:
        text = row.get("text")
        if not text:
            continue
        now_ts = int(row.get("now_ts", 0))
        utility = float(row.get("utility", 0.0))
        target_ns = row.get("namespace", base_ns)
        retriever.add(
            text=text,
            timestamp=now_ts,
            utility=utility,
            namespace=target_ns,
            visible_delay=visible_delay,
        )


def _query_batch_from_retriever(
    retriever: InMemoryBM25Temporal,
    queries: Iterable[str],
    cutoff_ts_list: Iterable[int],
    top_k: int,
    time_decay_lambda: float | None,
    namespaces: Sequence[str] | None,
    default_namespace: str,
) -> list[list[dict]]:
    queries = list(queries)
    cutoff_ts_list = list(cutoff_ts_list)
    if len(queries) != len(cutoff_ts_list):
        raise ValueError("queries and cutoff_ts_list must have the same length")

    if namespaces is None or len(namespaces) == 0:
        namespaces = [default_namespace] * len(queries)
    else:
        namespaces = list(namespaces)
        if len(namespaces) == 1 and len(queries) > 1:
            namespaces = namespaces * len(queries)

    results = []
    for query, cutoff, ns in zip(queries, cutoff_ts_list, namespaces, strict=True):
        results.append(
            retriever.query(
                query=query,
                namespace=ns,
                cutoff_ts=int(cutoff),
                top_k=top_k,
                time_decay_lambda=time_decay_lambda,
            )
        )
    return results


if ray is not None:

    @ray.remote
    class _RetrieverServer:
        def __init__(self, factory_bytes: bytes, default_namespace: str):
            factory: Callable[[], InMemoryBM25Temporal] = _pickle.loads(factory_bytes)
            self._retriever = factory()
            self._default_namespace = default_namespace

        def reset(self):
            self._retriever.reset()

        def add_candidates(self, rows: Sequence[dict], namespace: str | None, visible_delay: int):
            _add_rows_to_retriever(
                self._retriever,
                rows,
                default_namespace=self._default_namespace,
                namespace=namespace,
                visible_delay=visible_delay,
            )

        def query_batch(
            self,
            queries: Sequence[str],
            cutoff_ts_list: Sequence[int],
            top_k: int,
            time_decay_lambda: float | None,
            namespaces: Sequence[str] | None,
        ) -> list[list[dict]]:
            return _query_batch_from_retriever(
                self._retriever,
                queries,
                cutoff_ts_list,
                top_k,
                time_decay_lambda,
                namespaces,
                default_namespace=self._default_namespace,
            )


class DistributedRetriever:
    """
    Wrapper that optionally shares the retriever across workers via a Ray actor.
    """

    def __init__(
        self,
        retriever_factory: Callable[[], InMemoryBM25Temporal],
        default_namespace: str = "global",
        shared: bool = False,
        actor_name: str | None = None,
    ):
        self._retriever_factory = retriever_factory
        self._default_namespace = default_namespace
        self._shared = shared and _ray_available()
        if shared and not self._shared:
            logger.warning(
                "Think/revise retriever requested shared mode, but Ray is not available. "
                "Falling back to per-worker storage."
            )

        if self._shared:
            factory_bytes = _pickle.dumps(self._retriever_factory)
            name = actor_name or f"verl_think_retriever_{default_namespace}"
            try:
                self._actor = ray.get_actor(name)
                logger.info(f"Retrieved existing Ray actor: {name}")
            except ValueError:
                # Actor doesn't exist, try to create it
                try:
                    self._actor = _RetrieverServer.options(
                        name=name,
                        max_concurrency=8,
                        lifetime="detached",
                    ).remote(factory_bytes, default_namespace)
                    logger.info(f"Created new Ray actor: {name}")
                except ValueError as e:
                    # Another worker created it between our check and creation attempt
                    # This is expected in multi-worker initialization
                    if "already exists" in str(e):
                        logger.info(f"Actor {name} was created by another worker, retrieving it")
                        # Retry with exponential backoff in case actor is still registering
                        for attempt in range(5):
                            try:
                                self._actor = ray.get_actor(name)
                                break
                            except ValueError:
                                if attempt < 4:
                                    time.sleep(0.1 * (2 ** attempt))  # 0.1s, 0.2s, 0.4s, 0.8s, 1.6s
                                else:
                                    raise
                    else:
                        raise
        else:
            self._actor = None
            self._retriever = retriever_factory()
            self._lock = threading.Lock()

    def reset(self) -> None:
        if self._shared:
            ray.get(self._actor.reset.remote())
        else:
            with self._lock:
                self._retriever.reset()

    def add_candidates(
        self,
        rows: Sequence[dict],
        namespace: str | None = None,
        visible_delay: int = 0,
    ) -> None:
        if not rows:
            return

        if self._shared:
            ray.get(self._actor.add_candidates.remote(rows, namespace, visible_delay))
        else:
            with self._lock:
                _add_rows_to_retriever(
                    self._retriever,
                    rows,
                    default_namespace=self._default_namespace,
                    namespace=namespace,
                    visible_delay=visible_delay,
                )

    def query_batch(
        self,
        queries: Iterable[str],
        cutoff_ts_list: Iterable[int],
        top_k: int,
        time_decay_lambda: float | None = None,
        namespaces: Sequence[str] | None = None,
    ) -> list[list[dict]]:
        if self._shared:
            return ray.get(
                self._actor.query_batch.remote(
                    list(queries),
                    list(cutoff_ts_list),
                    top_k,
                    time_decay_lambda,
                    list(namespaces) if namespaces is not None else None,
                )
            )

        with self._lock:
            return _query_batch_from_retriever(
                self._retriever,
                queries,
                cutoff_ts_list,
                top_k,
                time_decay_lambda,
                namespaces,
                default_namespace=self._default_namespace,
            )
