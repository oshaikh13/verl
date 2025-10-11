"""Utility helpers for managing retrieved traces/memories."""

from __future__ import annotations

from collections import defaultdict
import math
import re
from typing import Callable, Iterable, Sequence


def _char_ngrams(text: str, n: int) -> set[str]:
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    if not cleaned:
        return set()
    padded = f" {cleaned} "
    return {padded[i : i + n] for i in range(max(1, len(padded) - n + 1))}


def jaccard_ngrams(a: str, b: str, n: int = 3) -> float:
    """Jaccard similarity over character n-grams."""
    set_a = _char_ngrams(a, n)
    set_b = _char_ngrams(b, n)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def mmr_select(
    items: Sequence[dict],
    sim_fn: Callable[[str, str], float],
    top_m: int,
    alpha: float,
) -> list[dict]:
    """Maximal Marginal Relevance selection."""
    if top_m <= 0:
        return []

    remaining = list(items)
    selected: list[dict] = []
    seen_bucket = defaultdict(int)

    while remaining and len(selected) < top_m:
        if not selected:
            best_idx = 0
            best_score = remaining[0].get("utility", 0.0)
        else:
            best_idx = None
            best_score = -math.inf
            for idx, cand in enumerate(remaining):
                relevance = float(cand.get("utility", 0.0))
                diversity = 0.0
                for chosen in selected:
                    diversity = max(diversity, sim_fn(cand["text"], chosen["text"]))
                score = alpha * relevance - (1 - alpha) * diversity
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is None:
                best_idx = 0
        chosen = remaining.pop(best_idx)
        bucket = chosen.get("bucket_key")
        if bucket is not None and seen_bucket[bucket]:
            continue
        if bucket is not None:
            seen_bucket[bucket] += 1
        selected.append(chosen)

    return selected
