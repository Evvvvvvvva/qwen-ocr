from datasets import load_dataset, Dataset, concatenate_datasets
import os
import pickle
import random
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd


def save_subset_to_json(subset_dict, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(subset_dict, f, ensure_ascii=False, indent=2)

def load_subset_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_data(image_path, math_file):
    image_names = set(os.listdir(image_path)) 
    image_paths = []
    math_gts = []

    with open(math_file, 'r') as f:
        for i, line in enumerate(f, start=1):
            image_name = f'{i-1:07d}.png'
            if line.strip() and image_name in image_names:
                image_paths.append(os.path.join(image_path, image_name))
                math_gts.append(line.strip())

    if len(image_paths) != len(math_gts):
        raise ValueError("Mismatch between image paths and formulas")

    return image_paths, math_gts

def load_unimer_dataset(is_train=False, sample_size=100):
    random.seed(42)
    if is_train:
        set_names = [
            "Old Data Mixture"
        ]
        image_paths = [
            "./data/UniMER-1M/images"
        ]
        math_files = [
            "./data/UniMER-1M/train.txt"
        ]
        save_file_path = 'unimer_train_10k.json'
    else:
        assert os.path.exists("./data/UniMER-Test"), "Please download UniMER-Test from https://opendatalab.com/OpenDataLab/UniMER-Dataset/tree/main/raw"
        set_names = [
            "Simple Print Expression(SPE)",
            "Screen Capture Expression(SCE)",
        ]
        image_paths = [
            "./data/UniMER-Test/spe",
            "./data/UniMER-Test/sce",
        ]
        math_files = [
            "./data/UniMER-Test/spe.txt",
            "./data/UniMER-Test/sce.txt",
        ]
        save_file_path = 'unimer_test.json'

    dataset = {}

    for set_name, image_path, math_file in zip(set_names, image_paths, math_files):
        image_list, math_gts = load_data(image_path, math_file)

        if len(image_list) < sample_size:
            raise ValueError(f"Not enough samples in {set_name}")

        indices = random.sample(range(len(image_list)), sample_size)

        subset_samples = []
        for idx in indices:
            if is_train and len(math_gts[idx]) < 30:
                # skip too simple cases
                continue
            subset_samples.append({
                'image': image_list[idx],
                'latex_formula': math_gts[idx]
            })
        print(f"Total sample size: {len(subset_samples)}")
        dataset[set_name] = subset_samples

    save_subset_to_json(dataset, save_file_path)
    return dataset

def load_hme100k_dataset(sample_size=2000):
    root_dir = './data/HME100k/train'
    assert os.path.exists(root_dir), "Please download HME100k from https://ai.100tal.com/dataset"
    with open(os.path.join(root_dir, 'images.pkl'), 'rb') as f:
        images = pickle.load(f)
    f.close()

    captions = {}
    with open(os.path.join(root_dir, 'caption.txt'), 'r') as f:
        for line in f:
            tmp = line.strip().split()
            image_name = tmp[0]
            caption = ' '.join(tmp[1:])
            captions[image_name] = caption
            # image = images[image_name]
    f.close()
    
    sampled_images = random.sample(list(captions.keys()), sample_size)
    all_samples = []
    for image_name in sampled_images:
        image = images[image_name]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        latex = captions[image_name]
        sample = {
            'id': f'hme_{image_name}',
            'image': image,
            'latex_formula': latex
        }
        all_samples.append(sample)

    return all_samples

def load_hmer_dataset():
    crohme = load_dataset('linxy/LaTeX_OCR', 'human_handwrite', shuffle_files=True)['train']
    def assign_id(example, idx):
        example["id"] = f'crohme_{idx}'
        return example
    crohme_with_id = crohme.map(assign_id, with_indices=True)
    crohme_with_id = crohme_with_id.shuffle(seed=42).select(range(1100))
    crohme_with_id = crohme_with_id.map(lambda x: {'image': Image.fromarray(x['image'])})
    crohme_with_id = crohme_with_id.map(lambda x: {'latex_formula': x['text']})
    hme100k = Dataset.from_list(load_hme100k_dataset(sample_size=1100))
    hmer = concatenate_datasets([crohme_with_id, hme100k])
    return hmer
