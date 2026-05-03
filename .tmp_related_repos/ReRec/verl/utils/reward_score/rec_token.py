import torch
from typing import List
from transformers import PreTrainedTokenizer
import re
import warnings
import math
from verl.utils.reward_score.rec import extract_recommendation

def _create_flexible_pattern(item: str) -> str:
    '''Strip trailing parenthesized year from item title (e.g. "Toy Story (1995)" -> "Toy Story")'''
    return re.sub(r'\s*\([0-9]{4}\)\s*$', '', item)

def compute_token_score(
    valid_response_ids: torch.Tensor,
    ground_truth: str,
    response_reward: float,
    tokenizer: PreTrainedTokenizer,
    min_token_reward_ratio: float = 0.5,
    max_token_reward_ratio: float = 1.0,
    mode: str = "rule",
) -> tuple[torch.Tensor, list]:
    """
    Compute per-token reward scores.

    Args:
        valid_response_ids: Token ids of the response.
        ground_truth: Semicolon-separated ground truth items.
        response_reward: Overall reward score for the response.
        tokenizer: Tokenizer used for decoding.
        min_token_reward_ratio: Minimum per-token reward ratio.
        max_token_reward_ratio: Maximum per-token reward ratio.
        mode: Scoring mode, one of "decrease" or "rule".

    Returns:
        tuple[torch.Tensor, list]: Per-token reward scores and token positions.
    """
    if mode == "decrease":
        return compute_token_score_decrease(
            valid_response_ids, ground_truth, response_reward,
            tokenizer, min_token_reward_ratio, max_token_reward_ratio
        )
    elif mode == "rule":
        return compute_token_score_rule(
            valid_response_ids, ground_truth, response_reward,
            tokenizer, min_token_reward_ratio, max_token_reward_ratio
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

def compute_token_score_decrease(
    valid_response_ids: torch.Tensor,
    ground_truth: str,
    response_reward: float,
    tokenizer: PreTrainedTokenizer,
    min_token_reward_ratio: float = 0.5,
    max_token_reward_ratio: float = 1.0,
    mode = "linear"
) -> tuple[torch.Tensor, list]:
    """
    Compute per-token reward scores using the decrease mode.
    Reward smoothly decreases with distance from ground truth item tokens.
    """
    # Parse ground truth into a list of items
    gt_items = [item.strip() for item in ground_truth.split(";") if item.strip()]

    # Initialize token scores to full response reward
    token_scores = torch.ones_like(valid_response_ids, dtype=torch.float32) * response_reward

    # If no ground truth items, return uniform reward
    if not gt_items:
        return token_scores, []

    # Decode response token ids to string
    response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)

    # Find character-level positions of each ground truth item in the response
    gt_positions = []
    for item in gt_items:
        matches = list(re.finditer(item, response_str, re.IGNORECASE))
        for match in matches:
            start_idx = match.start()
            end_idx = match.end()
            gt_positions.append((start_idx, end_idx))

    # If no matches found, return uniform reward
    if not gt_positions:
        return token_scores, []

    # Build character-to-token mapping
    char_to_token_map = []
    current_pos = 0

    # Decode each token individually
    tokens = [tokenizer.decode([id_]) for id_ in valid_response_ids]

    # Map each character position to its token index
    for token_idx, token in enumerate(tokens):
        token_len = len(token)
        for _ in range(token_len):
            char_to_token_map.append(token_idx)
            current_pos += 1

    # Convert character positions to token positions
    token_positions = []
    for start_char, end_char in gt_positions:
        if start_char < len(char_to_token_map) and end_char <= len(char_to_token_map):
            start_token = char_to_token_map[start_char]
            end_token = char_to_token_map[end_char - 1] + 1
            token_positions.append((start_token, end_token))
        else:
            if start_char >= len(char_to_token_map):
                start_char = len(char_to_token_map) - 1
            if end_char > len(char_to_token_map):
                end_char = len(char_to_token_map)
            if start_char < end_char:  # ensure positions are valid
                start_token = char_to_token_map[start_char]
                end_token = char_to_token_map[end_char - 1] + 1
                token_positions.append((start_token, end_token))

    # Compute each token's distance to the nearest ground truth token span
    all_min_distances = []
    for token_idx in range(len(valid_response_ids)):
        min_distance = float('inf')
        for start_token, end_token in token_positions:
            if start_token <= token_idx < end_token:
                min_distance = 0
                break
            if token_idx < start_token:
                distance = abs(token_idx - start_token)
            else:
                distance = abs(token_idx - (end_token - 1))
            min_distance = min(min_distance, distance)
        all_min_distances.append(min_distance)

    # Compute per-token reward based on distance
    for token_idx in range(len(valid_response_ids)):
        min_distance = all_min_distances[token_idx]

        # Compute reward ratio based on distance
        if min_distance == 0:
            ratio = max_token_reward_ratio
        else:
            # Exponential decay: ratio = max at distance 0, approaches min at infinity
            if mode == "exp":
                decay_rate = 0.03  # controls decay speed
                ratio = min_token_reward_ratio + (max_token_reward_ratio - min_token_reward_ratio) * math.exp(-decay_rate * min_distance)
            elif mode == "linear":
                max_distance = max(all_min_distances)
                normalized_distance = min_distance / max_distance
                # Linear interpolation between max and min ratio
                ratio = max_token_reward_ratio - normalized_distance * (max_token_reward_ratio - min_token_reward_ratio)
            else:
                raise ValueError(f"Unknown mode: {mode}")

        token_scores[token_idx] = ratio * response_reward

    return token_scores, token_positions

def compute_token_score_rule(
    valid_response_ids: torch.Tensor,
    ground_truth: str,
    response_reward: float,
    tokenizer: PreTrainedTokenizer,
    min_token_reward_ratio: float = 0.5,
    max_token_reward_ratio: float = 1.0,
) -> tuple[torch.Tensor, list]:
    """
    Compute per-token reward scores using the rule mode.
    Splits the response into reasoning and recommendation segments:
    - Reasoning segment: paragraphs mentioning wrong recommendation items get min ratio; others get max.
    - Recommendation segment (\\boxed{...}): skipped entirely.
    """

    # Initialize token scores to full response reward
    token_scores = torch.ones_like(valid_response_ids, dtype=torch.float32) * response_reward

    # Decode response
    response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
    recommendation = extract_recommendation(solution_str=response_str)

    # If no recommendation or ground truth, return uniform reward
    if not recommendation or not ground_truth:
        return token_scores, []

    # Decode each token individually
    tokens = [tokenizer.decode([id_]) for id_ in valid_response_ids]

    # Build character-to-token mapping
    char_to_token_map = []
    for token_idx, token in enumerate(tokens):
        token_len = len(token)
        for _ in range(token_len):
            char_to_token_map.append(token_idx)

    # Initialize all tokens to max reward ratio
    token_scores = torch.ones_like(valid_response_ids, dtype=torch.float32) * max_token_reward_ratio * response_reward
    fuzzy_recommendation = _create_flexible_pattern(recommendation)

    # Check whether the recommendation matches ground truth
    rec_match_gt = (recommendation.strip().lower() == ground_truth.strip().lower() or fuzzy_recommendation in ground_truth.strip().lower()) if recommendation and ground_truth else False

    # Process reasoning segment: split by newline into paragraphs
    paragraphs = response_str.split('\n')

    current_char_pos = 0

    if not rec_match_gt:
        for paragraph in paragraphs:
            paragraph_len = len(paragraph)
            if paragraph_len == 0:
                current_char_pos += 1  # skip newline character
                continue
            if '\\boxed' in paragraph:
                current_char_pos += paragraph_len + 1
                continue
            if fuzzy_recommendation in paragraph or recommendation in paragraph:
                # Convert paragraph character positions to token positions and apply min ratio
                if current_char_pos < len(char_to_token_map):
                    start_token = char_to_token_map[current_char_pos]
                    end_char_pos = current_char_pos + paragraph_len
                    if end_char_pos <= len(char_to_token_map):
                        end_token = char_to_token_map[end_char_pos - 1] + 1
                    for token_idx in range(start_token, end_token):
                        token_scores[token_idx] = min_token_reward_ratio * response_reward
            current_char_pos += paragraph_len + 1
    return token_scores, []
