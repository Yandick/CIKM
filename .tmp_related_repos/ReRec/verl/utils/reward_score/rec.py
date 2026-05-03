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

import re
import numpy as np
from typing import Dict, List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def extract_recommendation(solution_str):
    """Extract the recommendation from the solution string."""
    answer_pattern = r'\\boxed{([^}]+)}'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if not matches:
        return None
    
    # If there are 2 or more matches, return the first one
    return matches[0].group(1).strip()


def format_reward(solution_str):
    """Format the reward for the solution string.
       
       Check if the solution follows the required format:
       1. Must contain at least one <think>...</think> section
       2. Must contain exactly one <recommendation>...</recommendation> section
       3. All tags must be properly closed and in valid_tags list
       
       Returns:
       - 1.0: if format is correct
       - -1.0: if format is incorrect
    """

    recommendation_pattern = r"\\boxed{([^}]+)}"
    if not re.search(recommendation_pattern, solution_str, re.DOTALL):  # Missing recommendation
        return -0.5
    else:
        return 0.5
    

    

def calculate_recommendation_metrics(recommendation, ground_truth, k=1):
    """Calculate precision@k, recall@k, f1@k, ndcg@k and hit@k for recommendations.
    
    Args:
        recommendation: list of recommended items
        ground_truth: list of ground truth items
        k: number of top items to consider
        
    Returns:
        dict: Dictionary containing precision@k, recall@k, f1@k, ndcg@k and hit@k scores
    """
    # Only take top k recommendations
    recommendation = list(set(recommendation))[:k]
    ground_truth = list(set(ground_truth))
    
    # Calculate precision@k
    hits = sum(1 for item in recommendation if item in ground_truth)
    precision_at_k = hits / min(len(recommendation), k) if k > 0 else 0.0
    
    # Calculate recall@k
    recall_at_k = hits / len(ground_truth) if len(ground_truth) > 0 else 0.0
    
    # Calculate f1@k
    f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k) if (precision_at_k + recall_at_k) > 0 else 0.0
    
    # Calculate hit@k
    hit_at_k = 1.0 if hits > 0 else 0.0
    
    # Calculate ndcg@k
    dcg = 0.0
    idcg = 0.0
    
    # Calculate DCG
    for i, item in enumerate(recommendation):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
            
    # Calculate IDCG (ideal DCG)
    for i in range(min(len(ground_truth), k)):
        idcg += 1.0 / np.log2(i + 2)
        
    ndcg_at_k = dcg / idcg if idcg > 0 else 0.0
    
    if k==1:
        return {
            'ndcg@k': ndcg_at_k
        }
    else:
        return {
            'precision@k': precision_at_k,
            'recall@k': recall_at_k,
            'f1@k': f1_at_k,
            'ndcg@k': ndcg_at_k,
            'hit@k': hit_at_k
    }


def recommendation_reward(solution_str, ground_truth, metrics='ndcg@k', k=1):
    """Recommendation reward for the solution string.
       ground_truth: ground truth recommended items
       metrics: precision@k, recall@k, f1@k, ndcg@k
       k: number of top items to consider
    """
    
    recommendation = extract_recommendation(solution_str=solution_str)
    if recommendation is None:
        if k==1:
            return -0.5, {'ndcg@k': 0.0}
        else:
            return -0.5, {'precision@k': 0.0, 'recall@k': 0.0, 'f1@k': 0.0, 'ndcg@k': 0.0, 'hit@k': 0.0}
    if type(ground_truth) == str:
        ground_truth = ground_truth.split(';')
    if type(recommendation) == str:
        recommendation = recommendation.split(';')

    metric_scores = calculate_recommendation_metrics(recommendation, ground_truth, k=k)

    if metrics not in metric_scores:
        raise NotImplementedError(f"Metric '{metrics}' not implemented")

    metric_scores['recommendation'] = metric_scores[metrics]
    return metric_scores['recommendation'], metric_scores


def compute_qas_score(solution_str: str, ground_truth: str) -> float:
    # Configure retry strategy
    retry_strategy = Retry(
        total=5,  # max retries
        backoff_factor=0.5,  # retry interval
        status_forcelist=[500, 502, 503, 504]  # HTTP status codes to retry
    )

    # Create session with retry strategy
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    try:
        response = session.post(
            "http://localhost:8000/compute_qas",
            json={
                "solution_str": solution_str,
                "ground_truth": ground_truth
            },
            timeout=(5, 10)  # (connect timeout, read timeout)
        )

        if response.status_code == 200:
            return response.json()["qas_score"]
        else:
            print(f"Failed to compute QAS: {response.text}")
            return 0
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {str(e)}")
        return 0
    finally:
        session.close()


def compute_ias_score(solution_str: str, ground_truth: str) -> float:
    # Configure retry strategy
    retry_strategy = Retry(
        total=5,  # max retries
        backoff_factor=0.5,  # retry interval
        status_forcelist=[500, 502, 503, 504]  # HTTP status codes to retry
    )

    # Create session with retry strategy
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    try:
        response = session.post(
            "http://localhost:9000/compute_ias",
            json={
                "solution_str": solution_str,
                "ground_truth": ground_truth
            },
            timeout=(5, 10)  # (connect timeout, read timeout)
        )

        if response.status_code == 200:
            return response.json()["ias_score"]
        else:
            print(f"Failed to compute IAS: {response.text}")
            return 0
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {str(e)}")
        return 0
    finally:
        session.close()

def compute_score(solution_str: str, ground_truth: str, score_type=['recommendation'], k=1) -> Dict[str, float]:
    reward_weight = {
        'format': 0.1,
        'recommendation': 1.0,
        'qas': 0.01,
        'ias': 0.01
    }

    solution_str = re.sub(r"\s*(<|>|/)\s*", r"\1", solution_str)  # handle qwen2.5vl-32b format

    reward_dict = {}
    reward_dict['format'] = format_reward(solution_str)

    reward_dict['recommendation'], all_recommendation_metrics = recommendation_reward(solution_str, ground_truth, k=k)
    if 'qas' in score_type:
        reward_dict['qas'] = compute_qas_score(solution_str, ground_truth)
    if 'ias' in score_type:
        reward_dict['ias'] = compute_ias_score(solution_str, ground_truth)

    reward_dict.update(all_recommendation_metrics)

    overall = 0.0
    for reward_key, weight in reward_weight.items():
        if reward_key in score_type:
            overall += weight * reward_dict[reward_key]
    
    reward_dict['score'] = overall
    return reward_dict


def reward_func(data_source, solution_str, ground_truth, extra_info=None, k=1):
    if 'dual_graph_enhanced' in extra_info and extra_info['dual_graph_enhanced']:
        return compute_score(solution_str, ground_truth, score_type=['format','recommendation','ias','qas'], k=k)
    else:
        return compute_score(solution_str, ground_truth, score_type=['format','recommendation'], k=k)

