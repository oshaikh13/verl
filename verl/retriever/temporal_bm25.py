from __future__ import annotations

import math
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

TokenList = list[str]


def _tokenize(text: str) -> TokenList:
    # simple word tokenizer that lowercases and keeps alphanumerics
    if not text:
        return []
    return re.findall(r"[^\W_]+", text.lower())


@dataclass
class _Document:
    text: str
    tokens: TokenList
    tf: Counter
    length: int
    timestamp: int
    visible_ts: int
    utility: float


class InMemoryBM25Temporal:
    """
    Simple in-memory BM25 with optional temporal decay and deduplication.
    """

    def __init__(
        self,
        dedup_threshold: float = 0.95,
        dedup_sim_fn: Callable[[str, str], float] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self._k1 = k1
        self._b = b
        self._dedup_threshold = dedup_threshold
        self._dedup_sim_fn = dedup_sim_fn
        self._docs: dict[str, list[_Document]] = defaultdict(list)
        self._doc_freqs: dict[str, Counter] = defaultdict(Counter)
        self._doc_len_sum: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._docs.clear()
            self._doc_freqs.clear()
            self._doc_len_sum.clear()

    def add(
        self,
        text: str,
        timestamp: int,
        utility: float,
        namespace: str = "global",
        visible_delay: int = 0,
    ) -> None:
        if not text:
            return
        tokens = _tokenize(text)
        if not tokens:
            return
        doc = _Document(
            text=text,
            tokens=tokens,
            tf=Counter(tokens),
            length=len(tokens),
            timestamp=timestamp,
            visible_ts=timestamp + max(0, int(visible_delay)),
            utility=float(utility),
        )
        with self._lock:
            docs = self._docs[namespace]
            if self._dedup_sim_fn and self._dedup_threshold is not None:
                for existing in reversed(docs):
                    sim = self._dedup_sim_fn(existing.text, text)
                    if sim >= self._dedup_threshold:
                        return
            docs.append(doc)
            unique_tokens = set(tokens)
            self._doc_freqs[namespace].update(unique_tokens)
            self._doc_len_sum[namespace] += doc.length

    def query(
        self,
        query: str,
        namespace: str,
        cutoff_ts: int,
        top_k: int,
        time_decay_lambda: float | None = None,
    ) -> list[dict]:
        tokens = _tokenize(query)
        if not tokens:
            return []
        with self._lock:
            docs = [doc for doc in self._docs.get(namespace, []) if doc.visible_ts <= cutoff_ts]
            if not docs:
                return []
            doc_freqs = self._doc_freqs.get(namespace, Counter())
            total_docs = len(self._docs.get(namespace, [])) or 1
            avg_dl = self._doc_len_sum.get(namespace, 0) / max(1, len(self._docs.get(namespace, [])))

        scores = []
        idf_cache: dict[str, float] = {}
        for token in tokens:
            df = doc_freqs.get(token, 0)
            # add-one style smoothing to avoid div by zero
            idf_cache[token] = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))

        for doc in docs:
            score = 0.0
            for token in tokens:
                if token not in doc.tf:
                    continue
                tf = doc.tf[token]
                numerator = tf * (self._k1 + 1)
                denominator = tf + self._k1 * (1 - self._b + self._b * doc.length / max(avg_dl, 1e-8))
                score += idf_cache[token] * numerator / denominator

            if time_decay_lambda is not None:
                delta = max(0, cutoff_ts - doc.timestamp)
                score *= math.exp(-time_decay_lambda * delta)

            score += doc.utility
            scores.append(
                {
                    "text": doc.text,
                    "score": score,
                    "timestamp": doc.timestamp,
                    "utility": doc.utility,
                }
            )

        scores.sort(key=lambda item: item["score"], reverse=True)
        if top_k is not None and top_k > 0:
            scores = scores[:top_k]
        return scores

    def query_batch(
        self,
        queries: Iterable[str],
        namespace: str,
        cutoff_ts_list: Iterable[int],
        top_k: int,
        time_decay_lambda: float | None,
    ) -> list[list[dict]]:
        return [
            self.query(query=q, namespace=namespace, cutoff_ts=ts, top_k=top_k, time_decay_lambda=time_decay_lambda)
            for q, ts in zip(queries, cutoff_ts_list, strict=True)
        ]
