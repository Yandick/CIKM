#!/usr/bin/env python3
"""
This script is used to evaluate the predictions.

Usage:
python scripts/eval.py --pred_path predictions/predictions.json

"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import ast
import re

from verl.utils.reward_score.rec import recommendation_reward


def eval(prediction, ground_truth, k=1):
    """Evaluate a single prediction and return NDCG@k."""
    reward_dict = {}
    reward_dict['recommendation'], all_recommendation_metrics = recommendation_reward(prediction, ground_truth, k=k)
    return all_recommendation_metrics["ndcg@k"]


def load_predictions(pred_path: str) -> pd.DataFrame:
    """Load prediction result file."""
    print(f"Loading predictions: {pred_path}")
    try:
        if pred_path.endswith('.parquet'):
            df = pd.read_parquet(pred_path)
        elif pred_path.endswith('.json'):
            df = pd.read_json(pred_path)
        elif pred_path.endswith('.csv'):
            df = pd.read_csv(pred_path)
        else:
            raise ValueError(f"Unsupported file format: {pred_path}")
        
        print(f"Predictions loaded successfully, {len(df)} records")
        print(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Failed to load predictions: {e}")
        sys.exit(1)


def evaluate_predictions(df: pd.DataFrame, k=1) -> Dict[str, float]:
    """Evaluate predictions in the DataFrame."""
    print("Starting evaluation...")
    
    total_samples = len(df)
    all_metrics = []
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"Progress: {idx}/{total_samples}")
        
        # Extract prediction text
        prediction = row.get('prediction', '')
        metrics = eval(prediction, row['reward_model']['ground_truth'], k=k)
        all_metrics.append(metrics)
    
    avg_metrics = sum(all_metrics) / len(all_metrics)
    print(f"Average metric: {avg_metrics}")
    return all_metrics



def main():
    parser = argparse.ArgumentParser(description="Evaluate prediction results")
    parser.add_argument("--pred_path", type=str, default="predictions/ReQRec-book_movie_predictions.json", help="Path to the predictions file")
    
    args = parser.parse_args()
    
    # Validate file path
    if not os.path.exists(args.pred_path):
        print(f"Error: Predictions file does not exist: {args.pred_path}")
        sys.exit(1)

    
    print("=" * 50)
    print("Starting evaluation pipeline")
    print("=" * 50)
    
    # 1. Load predictions
    df = load_predictions(args.pred_path)
    
    # 2. Evaluate predictions
    metrics = evaluate_predictions(df, k=1)
    
    print("=" * 50)
    print("Evaluation complete!")
    print("=" * 50)


if __name__ == "__main__":
    main() 