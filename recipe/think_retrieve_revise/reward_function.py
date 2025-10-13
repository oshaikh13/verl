# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Text overlap reward function for Think-Retrieve-Revise agent loop.

This reward function evaluates the quality of the generated actions by measuring
text overlap with ground truth, considering both exact matches and semantic similarity.
"""

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional


def extract_actions(solution_str: str) -> List[str]:
    """Extract actions from the solution string.
    
    Args:
        solution_str: The full response from the model
        
    Returns:
        List of extracted action strings
    """
    # Extract content within <action>...</action> tags
    action_pattern = r"<action>(.*?)</action>"
    actions = re.findall(action_pattern, solution_str, re.IGNORECASE | re.DOTALL)
    
    # Clean up the actions
    cleaned_actions = []
    for action in actions:
        cleaned = action.strip()
        if cleaned:
            cleaned_actions.append(cleaned)
    
    return cleaned_actions


def normalize_text(text: str) -> str:
    """Normalize text for comparison.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
    """
    # Convert to lowercase and remove extra whitespace
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    
    # Remove common punctuation for better matching
    normalized = re.sub(r'[^\w\s]', '', normalized)
    
    return normalized


def compute_text_overlap(text1: str, text2: str) -> float:
    """Compute text overlap between two strings.
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Overlap score between 0.0 and 1.0
    """
    if not text1 or not text2:
        return 0.0
    
    # Normalize both texts
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    
    # Use SequenceMatcher for similarity
    matcher = SequenceMatcher(None, norm1, norm2)
    return matcher.ratio()


def compute_action_rewards(actions: List[str], ground_truth: str) -> Dict[str, float]:
    """Compute rewards for individual actions.
    
    Args:
        actions: List of generated actions
        ground_truth: Ground truth text
        
    Returns:
        Dictionary with reward metrics
    """
    if not actions:
        return {
            "exact_match": 0.0,
            "best_overlap": 0.0,
            "avg_overlap": 0.0,
            "coverage": 0.0
        }
    
    # Normalize ground truth
    gt_normalized = normalize_text(ground_truth)
    
    # Compute overlaps for each action
    overlaps = []
    exact_matches = 0
    
    for action in actions:
        action_normalized = normalize_text(action)
        
        # Check for exact match
        if action_normalized == gt_normalized:
            exact_matches += 1
        
        # Compute overlap
        overlap = compute_text_overlap(action, ground_truth)
        overlaps.append(overlap)
    
    # Calculate metrics
    best_overlap = max(overlaps) if overlaps else 0.0
    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
    
    # Coverage: how much of ground truth is covered by best action
    coverage = best_overlap
    
    # Exact match reward (binary)
    exact_match = 1.0 if exact_matches > 0 else 0.0
    
    return {
        "exact_match": exact_match,
        "best_overlap": best_overlap,
        "avg_overlap": avg_overlap,
        "coverage": coverage
    }


def compute_score(
    data_source: str, 
    solution_str: str, 
    ground_truth: str, 
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs
) -> float:
    """Main reward function for Think-Retrieve-Revise agent loop.
    
    This function evaluates the quality of generated actions by measuring
    text overlap with ground truth. It considers multiple metrics:
    - Exact match: Binary reward for perfect matches
    - Best overlap: Highest overlap score among all actions
    - Average overlap: Mean overlap across all actions
    - Coverage: How well the actions cover the ground truth
    
    Args:
        data_source: Source of the data (e.g., dataset name)
        solution_str: Full response from the model including think/revise/actions
        ground_truth: Expected answer or target text
        extra_info: Additional information (optional)
        **kwargs: Additional keyword arguments
        
    Returns:
        Reward score between 0.0 and 1.0
    """
    # Extract actions from the solution
    actions = extract_actions(solution_str)
    
    if not actions:
        # No actions found, return minimal reward
        return 0.0
    
    # Compute action rewards
    rewards = compute_action_rewards(actions, ground_truth)
    
    # Access log-likelihoods from extra_info if available
    log_likelihoods = {}
    if extra_info:
        log_likelihoods = {
            "think_log_likelihood": extra_info.get("think_log_likelihood"),
            "revise_log_likelihood": extra_info.get("revise_log_likelihood"), 
            "actions_log_likelihood": extra_info.get("actions_log_likelihood"),
            "ground_truth_log_likelihood": extra_info.get("ground_truth_log_likelihood")
        }
    
    # Weighted combination of metrics
    # You can adjust these weights based on your preferences
    weights = {
        "exact_match": 0.4,      # High weight for exact matches
        "best_overlap": 0.3,     # Good weight for best overlap
        "avg_overlap": 0.2,      # Moderate weight for average overlap
        "coverage": 0.1          # Lower weight for coverage
    }
    
    # Compute weighted score
    total_score = sum(rewards[metric] * weights[metric] for metric in weights)
    
    # Optional: Incorporate log-likelihood information
    if log_likelihoods.get("actions_log_likelihood") is not None:
        # You can use log-likelihood as a confidence measure
        # Higher log-likelihood = more confident generation
        confidence_bonus = min(0.1, log_likelihoods["actions_log_likelihood"] / 100.0)
        total_score += confidence_bonus
    
    # Ensure score is between 0 and 1
    final_score = max(0.0, min(1.0, total_score))
    
    # Optional: Add bonus for having multiple good actions
    if len(actions) > 1 and rewards["avg_overlap"] > 0.5:
        # Small bonus for multiple decent actions
        final_score = min(1.0, final_score + 0.05)
    
    return final_score


def compute_score_with_details(
    data_source: str, 
    solution_str: str, 
    ground_truth: str, 
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Extended reward function that returns detailed metrics.
    
    Args:
        data_source: Source of the data
        solution_str: Full response from the model
        ground_truth: Expected answer
        extra_info: Additional information
        **kwargs: Additional keyword arguments
        
    Returns:
        Dictionary with detailed reward information
    """
    actions = extract_actions(solution_str)
    rewards = compute_action_rewards(actions, ground_truth)
    
    # Compute final score
    final_score = compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
    
    return {
        "score": final_score,
        "num_actions": len(actions),
        "actions": actions,
        "rewards": rewards,
        "ground_truth": ground_truth
    }


# Alternative reward function names for different use cases
def text_overlap_reward(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Alias for compute_score for backward compatibility."""
    return compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)


def action_quality_reward(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Alternative reward function focusing on action quality."""
    actions = extract_actions(solution_str)
    
    if not actions:
        return 0.0
    
    # Focus more on exact matches and high-quality overlaps
    rewards = compute_action_rewards(actions, ground_truth)
    
    # Higher weight for exact matches and best overlap
    score = (rewards["exact_match"] * 0.6 + 
             rewards["best_overlap"] * 0.4)
    
    return score
