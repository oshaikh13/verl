"""
GPT verifier-based reward function for the think/revise LoRA demo.

This function supports batched invocation. Uses an LLM verifier to evaluate 
candidate responses with structured validation and penalty scoring.
"""

from __future__ import annotations

import json
import os
import re

from dotenv import load_dotenv
# Load .env from the same directory as this file
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_CURRENT_DIR, ".env"))
from typing import Any, Sequence, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from pydantic import BaseModel
from openai import OpenAI


# ---------- Config / Client ----------
client = OpenAI(
    api_key=os.environ.get("MODEL_API_KEY"),
    base_url=os.environ.get("MODEL_BASE_URL"),
)
VERIFIER_PROMPT_PATH = os.environ.get(
    "VERIFIER_PROMPT_PATH", 
    os.path.join(_CURRENT_DIR, "verifier_prompt.txt")
)
VERIFIER_MODEL = os.environ.get("MODEL_NAME", "gpt-4o-mini")
print(f"Verifier MODEL_BASE_URL={client.base_url}, MODEL={VERIFIER_MODEL}")

if os.path.exists(VERIFIER_PROMPT_PATH):
    with open(VERIFIER_PROMPT_PATH, "r", encoding="utf-8") as f:
        verifier_prompt = f.read()
else:
    print(f"Warning: Verifier prompt not found at {VERIFIER_PROMPT_PATH}")
    verifier_prompt = ""

# ---------- Schemas for structured parse ----------

@dataclass
class NormalizedCandidate:
    text_for_judge: str
    is_valid: bool
    errors: List[str]
    penalty_score: float  # Range: 0 (perfect) to -0.5 (worst)


class CandidateScore(BaseModel):
    id: int
    score: float


class VerifierResponse(BaseModel):
    reasoning: str
    candidates: List[CandidateScore]


# ---------- Helper Functions ----------

def _normalize_and_validate_candidate(raw: Optional[str]) -> NormalizedCandidate:
    """
    Normalize by removing all 'user' blocks and keeping only 'assistant' blocks.
    Validate (without enforcing exact block order or exact item counts):
      - <think> contains ONLY <claim>...</claim>
      - <revise> contains ONLY <claim>...</claim>
      - <actions> contains ONLY <action>...</action>
    Penalty spans from 0 to -0.5 (more negative = worse). Severe errors deduct more.
    """
    errors: List[str] = []
    penalty = 0.0  # start at 0, subtract for errors; clamp to [-0.5, 0]

    if not raw or not raw.strip():
        return NormalizedCandidate("", False, ["Empty input."], -0.5)


    text = raw

    # Keep ONLY assistant blocks; remove user blocks.
    assistant_blocks = re.findall(r"(?mis)^\s*assistant\s*(.*?)(?=^\s*(?:assistant|user)\b|\Z)", text)
    text_for_judge = "\n\n".join(block.strip() for block in assistant_blocks).strip() if assistant_blocks else text.strip()

    # Extract blocks; ensure exactly one of each (no order requirement)
    def extract_once(name: str, s: str):
        return list(re.finditer(fr"<{name}>(.*?)</{name}>", s, re.DOTALL))

    think_matches = extract_once("think", text_for_judge)
    revise_matches = extract_once("revise", text_for_judge)
    actions_matches = extract_once("actions", text_for_judge)

    def check_exactly_one(tag: str, matches: list):
        nonlocal penalty
        if not matches:
            errors.append(f"Missing <{tag}> block.")
            penalty -= 0.5
            return None
        if len(matches) > 1:
            errors.append(f"Multiple <{tag}> blocks found ({len(matches)}).")
            penalty -= 0.25
        return matches[0] if matches else None

    think_m = check_exactly_one("think", think_matches)
    revise_m = check_exactly_one("revise", revise_matches)
    actions_m = check_exactly_one("actions", actions_matches)

    think_inner = think_m.group(1) if think_m else ""
    revise_inner = revise_m.group(1) if revise_m else ""
    actions_inner = actions_m.group(1) if actions_m else ""

    # Helpers
    def only_allowed_items(inner: str, item_tag: str) -> bool:
        leftover = re.sub(fr"\s*<{item_tag}>.*?</{item_tag}>\s*", "", inner, 0, re.DOTALL)
        return leftover.strip() == ""

    def balanced_tag_counts(s: str, tag: str) -> bool:
        return s.count(f"<{tag}>") == s.count(f"</{tag}>")

    # Validate <think> (only <claim>)
    if think_m:
        if not only_allowed_items(think_inner, "claim"):
            errors.append("Non-<claim> content inside <think>.")
            penalty -= 0.25
        if not balanced_tag_counts(think_inner, "claim"):
            errors.append("Imbalanced <claim> tags in <think>.")
            penalty -= 0.25
        foreign_tags = re.findall(r"</?([a-zA-Z0-9:_-]+)[^>]*>", think_inner)
        foreign_tags = [t for t in foreign_tags if t not in {"claim"}]
        if foreign_tags:
            errors.append(f"Unexpected tags inside <think>: {sorted(set(foreign_tags))}")
            penalty -= 0.25

    # Validate <revise> (only <claim>, no exact count)
    if revise_m:
        if not only_allowed_items(revise_inner, "claim"):
            errors.append("Non-<claim> content inside <revise>.")
            penalty -= 0.25
        if not balanced_tag_counts(revise_inner, "claim"):
            errors.append("Imbalanced <claim> tags in <revise>.")
            penalty -= 0.25
        foreign_tags = re.findall(r"</?([a-zA-Z0-9:_-]+)[^>]*>", revise_inner)
        foreign_tags = [t for t in foreign_tags if t not in {"claim"}]
        if foreign_tags:
            errors.append(f"Unexpected tags inside <revise>: {sorted(set(foreign_tags))}")
            penalty -= 0.25

    # Validate <actions> (only <action>, no exact count)
    if actions_m:
        if not only_allowed_items(actions_inner, "action"):
            errors.append("Non-<action> content inside <actions>.")
            penalty -= 0.25
        if not balanced_tag_counts(actions_inner, "action"):
            errors.append("Imbalanced <action> tags in <actions>.")
            penalty -= 0.25
        foreign_tags = re.findall(r"</?([a-zA-Z0-9:_-]+)[^>]*>", actions_inner)
        foreign_tags = [t for t in foreign_tags if t not in {"action"}]
        if foreign_tags:
            errors.append(f"Unexpected tags inside <actions>: {sorted(set(foreign_tags))}")
            penalty -= 0.25

    # Unexpected tags anywhere in the assistant text (outside known tags)
    known = {"think", "revise", "actions", "claim", "action"}
    top_level_tags = re.findall(r"</?([a-zA-Z0-9:_-]+)[^>]*>", text_for_judge)
    extraneous_top = [t for t in set(top_level_tags) if t not in known]
    if extraneous_top:
        errors.append(f"Unexpected tags present: {sorted(extraneous_top)}")
        penalty -= 0.25

    # Clamp penalty to [-0.5, 0.0]
    penalty = max(-0.5, min(0.0, penalty))
    is_valid = len(errors) == 0

    text_for_judge = actions_m.group(0) if actions_m else ""
    
    return NormalizedCandidate(text_for_judge=text_for_judge, is_valid=is_valid, errors=errors, penalty_score=penalty)


def _build_candidates_block(c_texts: List[str]) -> str:
    """Build the markdown block passed to the verifier with all candidates."""
    out = []
    for j, c in enumerate(c_texts):
        out.append(f"- **Candidate {j + 1}**:\n{c}\n")
    return "".join(out)


def _safe_parse_verdict(prompt_text: str) -> VerifierResponse:
    """
    Calls the verifier model and returns the full structured response
    (reasoning + candidates). Raises on API/parse failure; caller should handle.
    """
    response = client.chat.completions.parse(
        model=VERIFIER_MODEL,
        messages=[{"role": "user", "content": prompt_text}],
        response_format=VerifierResponse,
    )
    return response.choices[0].message.parsed  # VerifierResponse


def _print_rollouts_dump(rollouts: List[dict], flat_rewards: List[float]) -> None:
    print("\n=== Rollouts & Rewards ===")
    for gi, group in enumerate(rollouts):
        print(f"\n--- Group {gi + 1} ---")
        if "ground_truth" in group:
            print("Ground truth:\n", group.get("ground_truth", ""))

        if group.get("reasoning"):
            print("\nVerifier reasoning:\n", group["reasoning"])

        validity_mask = group.get("valid", [])
        err_reasons = group.get("errors", [])
        full = group.get("full_completions", [])
        rewards = group.get("rewards", [])

        for ci, (cand, r) in enumerate(zip(full, rewards)):
            is_valid = validity_mask[ci] if ci < len(validity_mask) else True
            errors = err_reasons[ci] if ci < len(err_reasons) else []

            print(f"\nCandidate {ci + 1}:")
            print(f"Valid: {is_valid}")
            if errors:
                print(f"Errors/Penalties: {errors}")
            print(f"Completion:\n{cand}")
            print(f"Final Reward: {r}")
    print("\n=== Flat rewards ===")
    print(flat_rewards)


def _maximum_human_likelihood(
    *,
    completions: List[str],
    solution: List[str], 
    penalty_score: float = -0.5,
    max_workers: Optional[int] = None,
    log_path: Optional[str] = None,
) -> List[float]:
    """
    Compute rewards by grouping candidates that share the same ground-truth string.
    Prompts are not used.
    """
    n = len(completions)
    if n == 0:
        return []

    # 1) Build groups by ground truth text, preserving first-seen order
    sol_to_indices: dict[str, List[int]] = {}
    order: List[str] = []
    for idx, s in enumerate(solution):
        key = s or ""
        if key not in sol_to_indices:
            sol_to_indices[key] = []
            order.append(key)
        sol_to_indices[key].append(idx)

    groups: List[List[int]] = [sol_to_indices[k] for k in order]

    def _process_group(indices: List[int]) -> Tuple[List[int], List[float], dict]:
        i0 = indices[0]
        gt = solution[i0] or ""

        c_group = [completions[i] for i in indices]

        # Normalize & validate each candidate
        normalized: List[str] = []
        validity_mask: List[bool] = []
        err_reasons: List[List[str]] = []
        format_penalties: List[float] = []
        for c in c_group:
            norm = _normalize_and_validate_candidate(c)
            normalized.append(norm.text_for_judge)
            validity_mask.append(norm.is_valid)
            err_reasons.append(norm.errors)
            format_penalties.append(norm.penalty_score)

        # If all invalid → skip verifier
        if not any(validity_mask):
            group_rewards = [float(p) for p in format_penalties]
            group_record = {
                "indices": indices,
                "ground_truth": gt,
                "candidates": normalized,
                "valid": validity_mask,
                "errors": err_reasons,
                "rewards": group_rewards,
                "full_completions": c_group,
                "reasoning": "",
            }
            return indices, group_rewards, group_record

        # Build verifier prompt for this ground truth
        c_group_str = _build_candidates_block(normalized)
        prompt_text = verifier_prompt.replace("{ground_truth}", gt).replace("{candidates}", c_group_str)

        group_rewards: List[float] = []
        reasoning: str = ""
        try:
            verdict = _safe_parse_verdict(prompt_text)
            reasoning = verdict.reasoning or ""
            scored = verdict.candidates

            for j in range(len(indices)):
                if j >= len(scored):
                    group_rewards.append(float(format_penalties[j]))
                elif not validity_mask[j]:
                    group_rewards.append(float(format_penalties[j]))
                else:
                    group_rewards.append(float(scored[j].score))
        except Exception:
            group_rewards = [
                float(format_penalties[j]) if not validity_mask[j] else float(penalty_score)
                for j in range(len(indices))
            ]

        group_record = {
            "indices": indices,
            "ground_truth": gt,
            "candidates": normalized,
            "valid": validity_mask,
            "errors": err_reasons,
            "rewards": group_rewards,
            "full_completions": c_group,
            "reasoning": reasoning,
        }
        return indices, group_rewards, group_record

    rewards_by_index: dict[int, float] = {}
    rollouts: List[dict] = []

    # 2) Parallelize per ground-truth cluster
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_process_group, idxs): tuple(idxs) for idxs in groups}
        for fut in as_completed(futures):
            idxs, group_rewards, rec = fut.result()
            for local_pos, original_idx in enumerate(idxs):
                rewards_by_index[original_idx] = group_rewards[local_pos]
            rollouts.append(rec)

    # 3) Reassemble in original order
    rewards: List[float] = [rewards_by_index[i] for i in range(n)]

    # Optional logging
    if log_path:
        log_dir = os.path.dirname(os.path.abspath(log_path))
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            for rec in rollouts:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    _print_rollouts_dump(rollouts, rewards)
    return rewards


def compute_score(
    data_sources: Sequence[str],                 # kept for interface compatibility, unused
    solution_strs: Sequence[str],               # candidate completions
    ground_truths: Sequence[str] | None,        # literal answers
    extra_infos: Sequence[dict[str, Any] | None] | None = None,  # ignored
    **kwargs: Any,
) -> list[float]:
    """
    Score candidate completions against provided ground truths using an LLM verifier.
    Grouping is by ground_truths, which may be out of order [[[but clustered.
    """

    # split everything after the first assistant\n mention
    

    format_think_str = ("<think>\n"
    "<claim> ...claim content... </claim>\n"
    "<claim> ...claim content... </claim>\n"
    "...\n"
    "</think>")

    format_revise_str = ("<revise>\n"
    "<claim> ...revised claim content... </claim>\n"
    "<claim> ...revised claim content... </claim>\n"
    "...\n"
    "</revise>")

    format_actions_str = ("<actions>\n"
    "<action> ...action content... </action>\n"
    "<action> ...action content... </action>\n"
    "...\n"
    "</actions>")

    format_strs = {
        "think": format_think_str,
        "revise": format_revise_str,
        "actions": format_actions_str,
    }

    # replace these strings with an empty string in the solution_strs
    for key, value in format_strs.items():
        solution_strs = [s.replace(value, "") for s in solution_strs]

    res = _maximum_human_likelihood(
        completions=solution_strs,
        solution=ground_truths,                      # directly the answers
        penalty_score=kwargs.get("penalty_score", -0.5),
        max_workers=kwargs.get("max_workers", None),
        # log_path=kwargs.get("log_path", "./rollouts.jsonl"),
    )

    return res