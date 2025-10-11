"""Simple in-memory BM25 retriever with temporal decay support.

The implementation is intentionally lightweight so it can be used during
training without introducing heavyweight dependencies. Documents can be
added under arbitrary namespaces and queried later with optional time
decay and visibility controls.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
from typing import Callable, DefaultDict, Iterable, Optional


TokenizeFn = Callable[[str], list[str]]


_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _default_tokenize(text: str) -> list[str]:
    return _WORD_PATTERN.findall(text.lower())


@dataclass
class _Document:
    text: str
    tokens: list[str]
    term_freq: Counter
    length: int
    utility: float
    ts: int
    visible_ts: int
    metadata: Optional[dict] = None
    bucket_key: Optional[tuple] = None


class InMemoryBM25Temporal:
    """Very small BM25 implementation with temporal decay.

    Parameters mirror the typical BM25 formulation with optional time decay
    that down-weights older memories when a ``time_decay_lambda`` is supplied.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenize_fn: TokenizeFn = _default_tokenize,
    ) -> None:
        self.k1 = k1
        self.b = b
        self._tokenize = tokenize_fn
        self._namespaces: DefaultDict[str, list[_Document]] = defaultdict(list)
        self._doc_freq: DefaultDict[str, Counter] = defaultdict(Counter)
        self._avgdl: DefaultDict[str, float] = defaultdict(float)

    # ------------------------------------------------------------------ helpers
    def reset(self, namespace: Optional[str] = None) -> None:
        """Clear either a specific namespace or the whole index."""
        if namespace is None:
            self._namespaces.clear()
            self._doc_freq.clear()
            self._avgdl.clear()
        else:
            self._namespaces.pop(namespace, None)
            self._doc_freq.pop(namespace, None)
            self._avgdl.pop(namespace, None)

    def _update_stats_on_add(self, namespace: str, doc: _Document) -> None:
        docs = self._namespaces[namespace]
        docs.append(doc)
        df = self._doc_freq[namespace]
        for term in doc.term_freq:
            df[term] += 1

        total_len = sum(d.length for d in docs)
        self._avgdl[namespace] = total_len / len(docs) if docs else 0.0

    def add(
        self,
        *,
        namespace: str,
        text: str,
        now_ts: int,
        utility: float = 0.0,
        metadata: Optional[dict] = None,
        visible_delay: int = 0,
        bucket_key: Optional[tuple] = None,
    ) -> None:
        """Add a document to the index.

        Args:
            namespace: Memory namespace.
            text: Document text.
            now_ts: Timestamp when the document is observed.
            utility: Optional utility score used during selection.
            metadata: Arbitrary metadata payload.
            visible_delay: Delay (in the same units as ``ts``) before the memory
                becomes queryable.
            bucket_key: Optional key used for deduplication/overwriting.
        """
        tokens = self._tokenize(text)
        term_freq = Counter(tokens)
        length = len(tokens)
        doc = _Document(
            text=text,
            tokens=tokens,
            term_freq=term_freq,
            length=length,
            utility=utility,
            ts=now_ts,
            visible_ts=now_ts + max(0, visible_delay),
            metadata=metadata,
            bucket_key=bucket_key,
        )

        if bucket_key is not None:
            docs = self._namespaces[namespace]
            for idx, existing in enumerate(docs):
                if existing.bucket_key == bucket_key:
                    # Update metadata/effective timestamp but keep statistics stable.
                    docs[idx].utility = (existing.utility + utility) / 2 if docs[idx].utility else utility
                    docs[idx].ts = max(existing.ts, now_ts)
                    docs[idx].visible_ts = max(existing.visible_ts, doc.visible_ts)
                    docs[idx].metadata = metadata or existing.metadata
                    docs[idx].text = text
                    docs[idx].tokens = tokens
                    docs[idx].term_freq = term_freq
                    docs[idx].length = length
                    # Stats remain accurate because token distribution is replaced.
                    self._recompute_namespace_stats(namespace)
                    return

        self._update_stats_on_add(namespace, doc)

    # ------------------------------------------------------------------ scoring
    def _idf(self, namespace: str, term: str) -> float:
        df = self._doc_freq[namespace][term]
        n_docs = len(self._namespaces[namespace])
        return math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def _score_doc(self, namespace: str, doc: _Document, query_terms: list[str]) -> float:
        score = 0.0
        avgdl = self._avgdl[namespace] or 1.0
        for term in query_terms:
            if term not in doc.term_freq:
                continue
            idf = self._idf(namespace, term)
            freq = doc.term_freq[term]
            denom = freq + self.k1 * (1 - self.b + self.b * doc.length / avgdl)
            score += idf * ((freq * (self.k1 + 1)) / denom)
        return score

    # ---------------------------------------------------------------- queries
    def query(
        self,
        *,
        namespace: str,
        query: str,
        top_k: int = 5,
        cutoff_ts: Optional[int] = None,
        time_decay_lambda: Optional[float] = None,
    ) -> list[dict]:
        """Query documents using BM25 with optional exponential time decay."""
        if namespace not in self._namespaces or not self._namespaces[namespace]:
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        cutoff = cutoff_ts if cutoff_ts is not None else math.inf
        docs = self._namespaces[namespace]
        results: list[tuple[float, _Document]] = []
        for doc in docs:
            if doc.visible_ts > cutoff:
                continue
            score = self._score_doc(namespace, doc, query_terms)
            if time_decay_lambda is not None and cutoff_ts is not None:
                delta = max(0, cutoff_ts - doc.ts)
                score *= math.exp(-time_decay_lambda * delta)
            if score > 0:
                results.append((score, doc))

        results.sort(key=lambda x: x[0], reverse=True)
        top_results = results[:top_k]
        output = [
            {
                "score": score,
                "text": doc.text,
                "ts": doc.ts,
                "metadata": doc.metadata,
                "utility": doc.utility,
            }
            for score, doc in top_results
        ]
        return output

    def _recompute_namespace_stats(self, namespace: str) -> None:
        docs = self._namespaces[namespace]
        df = Counter()
        for doc in docs:
            df.update(doc.term_freq.keys())
        self._doc_freq[namespace] = df
        total_len = sum(doc.length for doc in docs)
        self._avgdl[namespace] = total_len / len(docs) if docs else 0.0

    def iter_namespace(self, namespace: str) -> Iterable[_Document]:
        return iter(self._namespaces.get(namespace, []))
