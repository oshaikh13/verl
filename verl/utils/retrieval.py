from __future__ import annotations

import re
from typing import Callable, Iterable, Sequence, Tuple


def jaccard_ngrams(text_a: str, text_b: str, n: int = 3) -> float:
    """
    Compute Jaccard similarity over word n-grams.
    """

    def _ngrams(text: str) -> set[Tuple[str, ...]]:
        tokens = re.findall(r"[^\W_]+", text.lower())
        if not tokens:
            return set()
        if len(tokens) < n:
            return {tuple(tokens)}
        return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}

    set_a = _ngrams(text_a)
    set_b = _ngrams(text_b)
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def mmr_select(
    items: Sequence[tuple[str, float, object]],
    sim_fn: Callable[[str, str], float],
    top_m: int,
    alpha: float = 0.7,
) -> list[tuple[str, float, object]]:
    """
    Maximal Marginal Relevance selection balancing relevance and diversity.
    """
    if top_m <= 0 or not items:
        return []

    selected: list[tuple[str, float, object]] = []
    candidates = list(items)

    while candidates and len(selected) < top_m:
        best_item = None
        best_score = float("-inf")
        for item in candidates:
            text, utility, payload = item
            relevance = utility
            if not selected:
                diversity_penalty = 0.0
            else:
                diversity_penalty = max(sim_fn(text, sel[0]) for sel in selected)
            score = alpha * relevance - (1 - alpha) * diversity_penalty
            if score > best_score:
                best_score = score
                best_item = item
        if best_item is None:
            break
        selected.append(best_item)
        candidates.remove(best_item)

    return selected
