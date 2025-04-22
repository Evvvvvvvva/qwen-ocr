from datasets import load_dataset
import os
from datasets import DatasetDict, Dataset
from PIL import Image
import random
import pandas as pd

def data_preprocess(example):
    sys_prompt = 'You are a helpful assistant.'
    instruction = 'Extract the formulas in the image, and output in Latex format.'
    # Get the ground-truth latex from the example.
    ground_truth = example['latex_formula']
    # Build the messages list. Note that we wrap the instruction with <image> as a marker.
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"<image>{instruction}"},
        {"role": "assistant", "content": ground_truth}
    ]
    # Returned dictionary will be appended to the hf dataset
    return {
        "messages": messages
    }

def data_preprocess_unimer(example):
    # Define fixed prompt and instruction.
    sys_prompt = 'You are a helpful assistant.'
    instruction = 'Extract the formulas in the image, and output in Latex format.'
    
    # Get the ground-truth latex from the example.
    ground_truth = example['latex_formula']
    
    # Build the messages list. Note that we wrap the instruction with <image> as a marker.
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"<image>{instruction}"},
        {"role": "assistant", "content": ground_truth}
    ]
    
    # Return a dictionary with two columns: "images" and "messages".
    return {
        "image": Image.open(example['image']),
        "messages": messages
    }

def convert_hmer():
    from load_datasets import load_hmer_dataset
    hf_ds = load_hmer_dataset()

    hf_ds = hf_ds.map(data_preprocess)
    split_80_20 = hf_ds.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_80_20["train"]    # 80% of the data
    temp_dataset = split_80_20["test"]        # 20% of the data

    split_val_test = temp_dataset.train_test_split(test_size=0.5, seed=42)
    val_dataset = split_val_test["train"]    # 10% of the total data
    test_dataset = split_val_test["test"]      # 10% of the total data
    splits = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    output_dir = 'hmer-stg2-processed'
    drive_dir = './'
    fp = os.path.join(drive_dir, output_dir)
    os.makedirs(fp, exist_ok=True)
    for split_name, ds in splits.items():
        splits["train"].to_parquet(fp+'/train.parquet')
        splits["validation"].to_parquet(fp+'/validation.parquet')
        splits["test"].to_parquet(fp+'/test.parquet')
    print(f"Stage 2 Dataset saved to: {fp}/[train|validation|test].parquet")
    # return splits

def convert_latex_formulas():
    hf_ds = load_dataset("OleehyO/latex-formulas", "cleaned_formulas")["train"]
    def assign_id(example, idx):
        example["id"] = idx
        return example
    ds_with_id = hf_ds.map(assign_id, with_indices=True)
    ds_with_id = ds_with_id.shuffle(seed=42)

    processed_ds = ds_with_id.map(data_preprocess)
    split_80_20 = processed_ds.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_80_20["train"]    # 80% of the data
    temp_dataset = split_80_20["test"]        # 20% of the data

    split_val_test = temp_dataset.train_test_split(test_size=0.5, seed=42)
    val_dataset = split_val_test["train"]    # 10% of the total data
    test_dataset = split_val_test["test"]      # 10% of the total data

    print("Example sample:", train_dataset[0])
    splits = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        })
    # Save the processed dataset as a Parquet file.
    output_dir = "latex-formulas-processed"
    os.makedirs(output_dir, exist_ok=True)
    splits["train"].to_parquet(output_dir+'/train.parquet')
    splits["validation"].to_parquet(output_dir+'/validation.parquet')
    splits["test"].to_parquet(output_dir+'/test.parquet')
    print(f"Stage 1 Dataset saved to: {output_dir}/[train|validation|test].parquet")

def convert_unimer():
    # deprecated
    from load_datasets import load_subset_from_json
    all_samples = load_subset_from_json('unimer_train_10k.json')["Old Data Mixture"]
    
    for idx, sample in enumerate(all_samples):
        sample["id"] = os.path.splitext(os.path.basename(sample["image"]))[0]

    hf_ds = Dataset.from_list(all_samples)

    hf_ds = hf_ds.map(data_preprocess_unimer)

    splits = DatasetDict({
        "train": hf_ds
    })
    output_dir = 'unimer-10k-processed'
    os.makedirs(output_dir, exist_ok=True)
    for split_name, ds in splits.items():
        ds.to_parquet(os.path.join(output_dir, f"{split_name}.parquet"))

    print(f"Stage 1 Dataset saved to: {output_dir}/[train|validation|test].parquet")
    return splits

def sample_from_parquet_pandas(parquet_path, output_path, sample_size=500, seed=42):
    df = pd.read_parquet(parquet_path)
    sampled_df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    dataset = Dataset.from_pandas(sampled_df)
    dataset.to_parquet(output_path)

if __name__ == '__main__':
    convert_latex_formulas()
    convert_hmer()