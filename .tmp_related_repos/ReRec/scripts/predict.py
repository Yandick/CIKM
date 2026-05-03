#!/usr/bin/env python3
"""
This script is used to predict the model on the specified dataset.

Usage:
python scripts/predict.py --model_path models/model_name --data_path data/dataset.parquet --output_path predictions/predictions.json --batch_dir predictions/batches --max_tokens 2048 --batch_size 1024 --include_prompts --clean_batch_files

"""

import argparse
import json
import os
import sys
import glob
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from vllm import LLM, SamplingParams


def load_dataset(data_path: str) -> pd.DataFrame:
    """Load dataset"""
    print(f"Loading dataset: {data_path}")
    try:
        df = pd.read_parquet(data_path)
        print(f"Dataset loaded successfully, {len(df)} records")
        print(f"Dataset columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        sys.exit(1)


def create_prompts(df: pd.DataFrame) -> List[str]:
    prompts = []
    for _, row in df.iterrows():
        try:
            prompt = row['prompt'][0]['content']
            prompts.append(prompt)
        except KeyError as e:
            print(f"Warning: Variable {e} not found in data, skipping this row")
            continue
    
    return prompts


def predict_and_save_batch(
    llm: LLM,
    sampling_params: SamplingParams,
    batch_prompts: List[str],
    batch_df: pd.DataFrame,
    output_dir: str,
    batch_idx: int,
    total_batches: int,
    include_prompts: bool = False
):
    """Predict one batch and save as a separate JSON file"""
    print(f"Processing batch {batch_idx + 1}/{total_batches}")
    
    # Run inference
    batch_outputs = llm.generate(batch_prompts, sampling_params)
    
    # Extract predictions
    predictions = []
    for output in batch_outputs:
        generated_text = output.outputs[0].text.strip()
        predictions.append(generated_text)
    
    # Create result DataFrame
    result_df = batch_df.copy()
    result_df['prediction'] = predictions[:len(batch_df)]
    
    if include_prompts:
        result_df['prompt'] = batch_prompts[:len(batch_df)]
    
    # Save this batch result as a separate JSON file
    try:
        
        batch_filename = f"batch_{batch_idx:04d}.json"
        batch_path = os.path.join(output_dir, batch_filename)
        
        result_df.to_json(batch_path, orient='records', indent=2, force_ascii=False)
        print(f"Batch {batch_idx + 1} saved to: {batch_path}, contains {len(result_df)} records")
        
    except Exception as e:
        print(f"Failed to save batch {batch_idx + 1}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def merge_batch_files(output_dir: str, final_output_path: str):
    """Merge all batch files into the final result"""
    print("Merging all batch files...")
    
    # Get all batch files
    batch_files = glob.glob(os.path.join(output_dir, "batch_*.json"))
    batch_files.sort()  # Ensure ordered merge
    
    if not batch_files:
        print("Error: No batch files found")
        sys.exit(1)
    
    print(f"Found {len(batch_files)} batch files")
    
    all_records = []
    
    for batch_file in batch_files:
        print(f"Merging: {os.path.basename(batch_file)}")
        try:
            with open(batch_file, 'r', encoding='utf-8') as f:
                batch_records = json.load(f)
                all_records.extend(batch_records)
        except Exception as e:
            print(f"Failed to read batch file {batch_file}: {e}")
            continue
    
    # Save merged results
    try:
        if final_output_path.endswith('.json'):
            with open(final_output_path, 'w', encoding='utf-8') as f:
                json.dump(all_records, f, indent=2, ensure_ascii=False)
        elif final_output_path.endswith('.parquet'):
            final_df = pd.DataFrame(all_records)
            final_df.to_parquet(final_output_path, index=False)
        elif final_output_path.endswith('.csv'):
            final_df = pd.DataFrame(all_records)
            final_df.to_csv(final_output_path, index=False, encoding='utf-8')
        else:
            # Default to JSON
            with open(final_output_path, 'w', encoding='utf-8') as f:
                json.dump(all_records, f, indent=2, ensure_ascii=False)
        
        print(f"Merging complete, final file contains {len(all_records)} records")
        print(f"Final result saved to: {final_output_path}")
        
        # Clean up batch files (optional)
        if args.clean_batch_files:
            for batch_file in batch_files:
                os.remove(batch_file)
            print("Batch files cleaned up")
        
    except Exception as e:
        print(f"Failed to save final result: {e}")
        sys.exit(1)


def predict_with_vllm(
    model_path: str,
    df: pd.DataFrame,
    prompts: List[str],
    output_dir: str,
    final_output_path: str,
    max_tokens: int = 2048,
    batch_size: int = 128,
    include_prompts: bool = False
):
    """Run inference with vLLM and save per batch"""
    print(f"Loading model: {model_path}")
    
    try:
        # Initialize LLM
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=1,  # Adjust based on number of GPUs if needed
            gpu_memory_utilization=0.8
        )
        
        # Set sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_tokens
        )
        
        print(f"Starting inference: {len(prompts)} samples, batch size: {batch_size}")
        
        # Create batch output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process and save per batch
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_df = df.iloc[i:i + batch_size].reset_index(drop=True)
            
            # Predict and save this batch
            predict_and_save_batch(
                llm=llm,
                sampling_params=sampling_params,
                batch_prompts=batch_prompts,
                batch_df=batch_df,
                output_dir=output_dir,
                batch_idx=i // batch_size,
                total_batches=total_batches,
                include_prompts=include_prompts
            )
        
        print(f"All batches completed!")
        
        # Merge all batch files
        merge_batch_files(output_dir, final_output_path)
        
    except Exception as e:
        print(f"Model loading or inference failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run inference with vLLM")
    parser.add_argument("--model_path", type=str, default="models/ReQRec-book", help="Path to the model")
    parser.add_argument("--data_path", type=str, default="data/movie_test_w_and_wo_history.parquet", help="Path to the dataset")
    parser.add_argument("--output_path", type=str, default="predictions/ReQRec-book_movie_predictions.json", help="Path to the final output file")
    parser.add_argument("--batch_dir", type=str, default="predictions/batches", help="Directory to save batch files")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum generated tokens")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--include_prompts", action="store_true", help="Include prompts in the output")
    parser.add_argument("--clean_batch_files", action="store_true", help="Delete batch files after merging")
    
    global args
    args = parser.parse_args()
    
    # Validate file paths
    if not os.path.exists(args.data_path):
        print(f"Error: Dataset file does not exist: {args.data_path}")
        sys.exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model path does not exist: {args.model_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 50)
    print("Starting prediction pipeline")
    print("=" * 50)
    
    # 1. Load dataset
    df = load_dataset(args.data_path)
    
    # 2. Create prompts
    prompts = create_prompts(df)
    
    # 3. Run inference and save per batch
    predict_with_vllm(
        model_path=args.model_path,
        df=df,
        prompts=prompts,
        output_dir=args.batch_dir,
        final_output_path=args.output_path,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        include_prompts=args.include_prompts
    )
    
    print("=" * 50)
    print("Prediction complete!")
    print("=" * 50)


if __name__ == "__main__":
    main() 