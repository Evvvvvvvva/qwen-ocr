import pandas as pd
from datasets import Dataset
import os
from convert2sharegpt import data_preprocess

def update_parquet_with_tokenized(parquet_path, tokenized_jsonl, output_parquet):
    # Step 1: Load original and tokenized data
    df = pd.read_parquet(parquet_path)
    tokenized_df = pd.read_json(tokenized_jsonl, lines=True)

    # Step 2: Rename tokenized column and merge
    tokenized_df = tokenized_df.rename(columns={"latex": "latex_tokenized"})
    merged_df = df.merge(tokenized_df, on="id", how="inner")

    # Step 3: Replace 'latex_formula' with the tokenized version
    merged_df["latex_formula"] = merged_df["latex_tokenized"]
    merged_df = merged_df.drop(columns=["latex_tokenized"])

    # Step 4: Convert to HuggingFace Dataset
    hf_dataset = Dataset.from_pandas(merged_df)

    # Step 5: Add messages column using map
    hf_dataset = hf_dataset.map(data_preprocess)

    # Step 6: Save back to Parquet
    hf_dataset.to_pandas().to_parquet(output_parquet, index=False)
    print(f"Updated and saved to: {output_parquet}")

def merge_results(tokenized_jsonl, results_json):
    tokenized_df = pd.read_json(tokenized_jsonl, lines=True)
    results_df = pd.read_json(results_json)
    # Step 2: Rename tokenized column and merge
    tokenized_df = tokenized_df.rename(columns={"latex": "gt_tokenized"})
    merged_df = results_df.merge(tokenized_df, on="id", how="inner")

    # Step 3: Replace 'latex_formula' with the tokenized version
    merged_df["gt"] = merged_df["gt_tokenized"]
    merged_df = merged_df.drop(columns=["gt_tokenized"])
    merged_df.to_json(f"tokenzied_{results_json}", orient="records", indent=2, force_ascii=False)


def merge_gt():
    # Apply to each split
    root_dir = "latex-formulas-processed"
    save_dir = "latex-formulas-tokenized"
    os.makedirs(save_dir, exist_ok=True)
    update_parquet_with_tokenized(f"{root_dir}/train.parquet", "tokenized_train.jsonl", f"{save_dir}/train.parquet")
    update_parquet_with_tokenized(f"{root_dir}/validation.parquet", "tokenized_val.jsonl", f"{save_dir}/validation.parquet")
    update_parquet_with_tokenized(f"{root_dir}/test.parquet", "tokenized_test.jsonl", f"{save_dir}/test.parquet")
    # merge_results('tokenized_test_1k.jsonl', 'baseline_inference_results.json')