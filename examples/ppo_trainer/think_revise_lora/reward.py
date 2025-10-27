"""
Simple token-overlap reward used by the think/revise LoRA demo.

This function supports both scalar and batched invocation. When the batch reward
manager calls it, the inputs are lists; when the naive manager calls it, they
may be scalars. We measure recall-style Jaccard overlap between the model
response and the provided ground-truth tokens.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[^\W_]+", text.lower()))


def _score_single(
    solution_str: str,
    ground_truth: str | None,
    extra_info: dict[str, Any] | None,
) -> float:
    if not solution_str:
        return 0.0

    if not ground_truth and extra_info:
        ground_truth = extra_info.get("target") or extra_info.get("answer")

    if not ground_truth:
        return 0.0

    pred_tokens = _token_set(solution_str)
    ref_tokens = _token_set(str(ground_truth))

    if not ref_tokens:
        return 0.0

    intersection = pred_tokens & ref_tokens
    return len(intersection) / max(1, len(ref_tokens))


def compute_score(
    data_sources: Sequence[str] | str,
    solution_strs: Sequence[str] | str,
    ground_truths: Sequence[str | None] | str | None,
    extra_infos: Sequence[dict[str, Any] | None] | dict[str, Any] | None = None,
    **_: Any,
) -> list[float] | float:
    """
    Batched overlap score between completions and ground truth.

    Args mirror the signature expected by `BatchRewardManager`. If scalar inputs
    are provided we return a single float; otherwise a list of floats aligned
    with the batch order.
    """

    # Scalar mode (naive reward manager)
    if isinstance(solution_strs, str):
        gt = ground_truths if isinstance(ground_truths, str) or ground_truths is None else None
        extra = extra_infos if isinstance(extra_infos, dict) else None
        return _score_single(solution_strs, gt, extra)

    # Batched mode
    batched_ground_truths: Iterable[str | None]
    if ground_truths is None:
        batched_ground_truths = [None] * len(solution_strs)
    else:
        batched_ground_truths = ground_truths

    batched_extra_infos: Iterable[dict[str, Any] | None]
    if extra_infos is None:
        batched_extra_infos = [None] * len(solution_strs)
    else:
        batched_extra_infos = extra_infos

    return [
        _score_single(sol, gt, extra)
        for sol, gt, extra in zip(solution_strs, batched_ground_truths, batched_extra_infos, strict=True)
    ]
