"""Utilities for synchronising an in-memory retriever across processes."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

from accelerate import Accelerator

from .temporal_bm25 import InMemoryBM25Temporal


class DistributedRetriever:
    """Thin wrapper that keeps retriever state in sync across processes.

    The implementation deliberately keeps the interface minimal so the trainer
    can operate both in single-process and distributed ``accelerate`` setups
    without relying on external infrastructure.
    """

    def __init__(
        self,
        retriever: InMemoryBM25Temporal,
        accelerator: Accelerator,
        default_namespace: Optional[str] = None,
    ) -> None:
        self.retriever = retriever
        self.accelerator = accelerator
        self.default_namespace = default_namespace or "default"

    # ------------------------------------------------------------------ helpers
    def reset(self, namespace: Optional[str] = None) -> None:
        """Reset shared state."""
        ns = namespace or self.default_namespace
        self.retriever.reset(ns)
        self.accelerator.wait_for_everyone()

    # ------------------------------------------------------------------- queries
    def query_batch(
        self,
        queries: Iterable[str],
        *,
        cutoff_ts_list: Optional[Iterable[int]] = None,
        top_k: int = 5,
        time_decay_lambda: Optional[float] = None,
        namespaces: Optional[Iterable[str]] = None,
    ) -> list[list[dict[str, Any]]]:
        query_list = list(queries)
        cutoff_iter = list(cutoff_ts_list) if cutoff_ts_list is not None else [None] * len(query_list)
        ns_iter = list(namespaces) if namespaces is not None else [self.default_namespace] * len(query_list)

        results = []
        for prompt, cutoff, namespace in zip(query_list, cutoff_iter, ns_iter):
            hits = self.retriever.query(
                namespace=namespace,
                query=prompt,
                top_k=top_k,
                cutoff_ts=cutoff,
                time_decay_lambda=time_decay_lambda,
            )
            results.append(hits)
        return results

    # ------------------------------------------------------------------- updates
    def add_candidates_parsimonious(
        self,
        local_rows: list[dict[str, Any]],
        *,
        dedup_sim_fn: Callable[[str, str], float],
        mmr_select_fn: Callable[[list[dict[str, Any]], Callable[[str, str], float], int, float], list[dict[str, Any]]],
        top_m: int,
        alpha: float,
        visible_delay: int = 0,
        namespace: Optional[str] = None,
    ) -> None:
        gathered = self.accelerator.gather_object(local_rows)
        if self.accelerator.num_processes > 1:
            # Flatten list of lists; gather_object guarantees identical order everywhere.
            candidates = [item for sublist in gathered for item in sublist]
        else:
            candidates = local_rows

        if not candidates:
            return

        candidates.sort(key=lambda row: row.get("utility", 0.0), reverse=True)
        selected = mmr_select_fn(candidates, dedup_sim_fn, top_m, alpha)
        ns = namespace or self.default_namespace
        for row in selected:
            self.retriever.add(
                namespace=ns,
                text=row["text"],
                now_ts=int(row.get("now_ts", 0)),
                utility=float(row.get("utility", 0.0)),
                metadata=row.get("metadata"),
                visible_delay=visible_delay,
                bucket_key=row.get("bucket_key"),
            )
        self.accelerator.wait_for_everyone()
