import torch
import time
import random
from datasets import load_dataset, Dataset
import evaluate # huggingface evaluate
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
import numpy as np
import json
import argparse
from tqdm import tqdm
from qwen_inference import run_model
from quick_latex import render_latex_quicklatex
from anls import anls_score  # https://pypi.org/project/anls/
from PIL import Image
from rapidfuzz.distance import Levenshtein
import os
import re
import io
from cdm.evaluation import run_cdm

def cdm_score(result_path, output_path):
    cdm, exp_cdm = run_cdm(result_path, output_path)
    return cdm, exp_cdm

def bleu_score(predictions, references):
    bleu = evaluate.load("bleu", keep_in_memory=True, experiment_id=random.randint(1, int(1e8)))
    return  bleu.compute(predictions=predictions, references=references)["bleu"]

def clean_latex_environments(latex_code: str) -> str:
    """
    Remove whitespaces inside the braces of LaTeX \\begin{...} and \\end{...}
    """
    def replacer(match):
        cmd = match.group(1)  # either 'begin' or 'end'
        content = match.group(2)
        cleaned_content = content.replace(' ', '')  # remove spaces
        return f'\\{cmd}{{{cleaned_content}}}'

    pattern = r'\\(begin|end)\s*{\s*([^}]*)\s*}'
    cleaned_code = re.sub(pattern, replacer, latex_code)
    cleaned_code = re.sub(r'\$\s+\$', '$$', cleaned_code)
    return cleaned_code


def smart_latex_tokenizer(expr):
    # expr = re.sub(r"\\begin\{.*?\}", "", expr)
    # expr = re.sub(r"\\end\{.*?\}", "", expr)

    pattern = r'''
        (\\[a-zA-Z]+)        |  # LaTeX command like \frac, \alpha
        ([0-9]*\.?[0-9]+)    |  # numbers: 12, 3.14
        ([a-zA-Z])           |  # single variables
        (\^|_|{|}|\(|\)|\[|\]) |  # special math syntax
        (\+|\-|\*|/|=|<|>|,)   |  # operators & punctuation
        (\\.)                 |  # escaped special characters: \_, \%, etc.
        (\s+)                 |  # whitespace (optional to keep)
        (.)                   # catch all (e.g., unknown symbols)
    '''

    tokens = [t for t in re.findall(pattern, expr, re.VERBOSE)]
    # Flatten and remove empty strings
    tokens = [item for group in tokens for item in group if item.strip()]
    return ' '.join(tokens)


def normalize_text(text):
    """Remove unnecessary whitespace from LaTeX code."""
    text_reg = r'(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})'
    letter = '[a-zA-Z]'
    noletter = '[\W_^\d]'
    names = [x[0].replace(' ', '') for x in re.findall(text_reg, text)]
    text = re.sub(text_reg, lambda match: str(names.pop(0)), text)
    news = text
    while True:
        text = news
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, noletter), r'\1\2', text)
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, letter), r'\1\2', news)
        news = re.sub(r'(%s)\s+?(%s)' % (letter, noletter), r'\1\2', news)
        if news == text:
            break
    return text

def normalize_latex(latex, convert_format=False):
    """
    Normalize LaTeX expressions to ensure consistency in evaluation.
    If convert_format is True, replace '$$...$$' with '\\begin{align*}...\\end{align*}'.
    """
    latex = latex.strip()
    if convert_format:
        # latex = latex.replace("$$", "\\begin{align*}", 1)
        # latex = latex.replace("$$", "\\end{align*}", 1)
        latex = latex.replace("$$", "", 1)
        latex = latex.replace("$$", "", 1)
    # latex = smart_latex_tokenizer(latex)
    latex = normalize_text(latex)
    return latex

def images_are_equal(img1, img2):
    """
    Compare two PIL images for exact pixel match.
    Returns:
        bool: True if identical in size and all pixel values, else False.
    """
    if img1 is None or img2 is None:
        return False
    if img1.size != img2.size:
        return False
    return np.array_equal(np.array(img1), np.array(img2))

def is_latex_exact_match(gt, pred):
    """Compares two LaTeX strings for exact match after normalizing white spaces."""
    gt = gt.replace("\n", " ").replace(" ", "")
    pred = gt.replace("\n", " ").replace(" ", "")
    return gt == pred

def load_evalset(load_unimer=False, load_hme=False):
    if load_unimer:
        if not os.path.exists('unimer_test.json'):
            from load_datasets import load_unimer_dataset
            return load_unimer_dataset()
        from load_datasets import load_subset_from_json
        return load_subset_from_json('unimer_test.json'), 'unimer'
    elif load_hme:
        return load_dataset("./hmer-stg2-processed", split="test", trust_remote_code=True), 'HMER'
    else:
        return load_dataset("./latex-formulas-tokenized", split="test", trust_remote_code=True), 'latex-formulas-tokenized'

def inference_(processor, model, dataset_iter, save_path):
    """Runs inference on the dataset and saves predictions to a JSON file."""
    results = []
    for sample in tqdm(dataset_iter, desc="Running Inference"):
        # print(sample)
        image_item = sample["image"]
        if type(image_item) == dict:
            image_item = sample["image"]["bytes"]
            image_item = Image.open(io.BytesIO(image_item))
        gt = sample["latex_formula"]
        pred = run_model(processor, model, image_item)

        results.append({
            "id": sample["id"],
            "gt": gt,
            "pred": pred
        })

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Inference results saved to {save_path}")
    
    return save_path


def unimer_preprocess(example):
    image_path = example['image']
    subset_folder = os.path.basename(os.path.dirname(image_path))
    filename = os.path.basename(image_path)
    image_id = os.path.join(subset_folder, filename)
    return {
        "id": image_id
    }


def inference(processor, model, dataset, save_path, dataset_name):
    """
    Runs inference on either a HuggingFace Dataset or a dict subset.
    Saves predictions to a JSON file.

    Args:
        processor: the preprocessor/tokenizer
        model: the model to run
        dataset: HuggingFace Dataset or dict of subsets
        save_path: output json path
    """
    results = {}

    if isinstance(dataset, dict):
        for subset_name, samples in dataset.items():
            match = re.search(r"\((.*?)\)", subset_name)
            print(subset_name)
            if match:
                set_id = match.group(1)
            hf_ds = Dataset.from_list(samples)
            hf_ds = hf_ds.map(unimer_preprocess)
            results[f'{dataset_name}-{set_id}'] = inference_(processor, model, hf_ds, set_id + '_' + save_path)
        return results
    # If input is a HuggingFace Dataset
    elif isinstance(dataset, Dataset):
        results[dataset_name] = inference_(processor, model, dataset, save_path)
        return results
    else:
        raise TypeError("Unsupported dataset type. Must be HuggingFace Dataset or dict.")

def compute_standard_metrics(predictions, references):
    lev_dist = []
    anls_s = []
    exp_rate = []
    for pred, gt in zip(predictions, references):
        lev_dist.append(Levenshtein.normalized_distance(pred, gt))
        anls_s.append(anls_score(prediction=pred, gold_labels=[gt], threshold=0.5))
        exp_rate.append(1 if pred == gt else 0)

    return {
        'bleu': bleu_score(predictions, references),
        'edit': sum(lev_dist) / len(lev_dist),
        'anls': sum(anls_s) / len(anls_s),
        'exp': sum(exp_rate) / len(exp_rate)
    }

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    f.close()
    return data

def load_results(results_path):
    if 'jsonl' in results_path:
        results = load_jsonl(results_path)
    else:
        with open(results_path, 'r') as f:
            results = json.load(f)
        f.close()
    return results

def eval_unimer(results_path, evaluation_save_path, model_name, dataset_info, convert_format=False):
    # deprecated
    results = load_results(results_path) 
    subset_dict = {'spe':{'gt': [],'pred': []}, 'cpe': {'gt': [],'pred': []}, 'sce': {'gt': [],'pred': []}, 'hwe': {'gt': [],'pred': []}}
    pred_ltx = []
    gt_ltx = []
    curr_set = 'spe'
    eval_results = {}
    for result in results:
        next_set = result['id'].split('/')[0]
        if next_set == 'hwe':
            gt_ltx.append(normalize_latex(result['gt'], convert_format=False))
            pred_ltx.append(normalize_latex(result['pred'], convert_format=convert_format))
        # if curr_set == next_set:
        #     gt_ltx.append(normalize_latex(result['gt'], convert_format=False))
        #     pred_ltx.append(normalize_latex(result['pred'], convert_format=convert_format))
        # else:
        #     eval_result = compute_standard_metrics(pred_ltx, gt_ltx)
        #     eval_result['model_name'] = model_name
        #     eval_result['dataset_info'] = 'unimer'
        #     eval_results[curr_set] = eval_result
        #     curr_set = next_set
        #     pred_ltx = []
        #     gt_ltx = []
        #     gt_ltx.append(normalize_latex(result['gt'], convert_format=False))
        #     pred_ltx.append(normalize_latex(result['pred'], convert_format=convert_format))
    eval_result = compute_standard_metrics(pred_ltx, gt_ltx)
    eval_result['model_name'] = model_name
    eval_result['dataset_info'] = f'unimer-{dataset_info}'
    eval_results['hwe'] = eval_result
    with open(evaluation_save_path, 'w') as f:
        json.dump(eval_results, f, indent=4)
    
    print(json.dumps(eval_results, indent=4))


def eval_by_batch(results_path, evaluation_save_path, model_name, dataset_info, convert_format=False):
    results = load_results(results_path)
    gt_ltx = [normalize_latex(result['gt'], convert_format=False) for result in results]
    pred_ltx = [normalize_latex(result['pred'], convert_format=convert_format) for result in results]
    # String-Based Metric
    eval_result = compute_standard_metrics(pred_ltx, gt_ltx)
    # Image-Based Metric
    eval_result['cdm'], eval_result['exp_cdm'] = cdm_score(os.path.abspath(results_path), 'cdm-results')
    eval_result['model_name'] = model_name
    eval_result['dataset_info'] = dataset_info
    with open(evaluation_save_path, 'w') as f:
        json.dump(eval_result, f, indent=4)
    
    print(json.dumps(eval_result, indent=4))

def eval_image(index, gt, pred, gt_successes,pred_successes,img_exact_matches):
    gt_img = render_latex_quicklatex(clean_latex_environments(gt))
    pred_img = render_latex_quicklatex(clean_latex_environments(pred))
    
    # Track rendering success
    if type(gt_img) is str: 
        print(f"Error GT Latex: {gt_img}")
        gt_img = None
    else:
        gt_successes[index] = True
    if type(pred_img) is str: 
        print(f"Error Pred Latex: {pred_img}")
        pred_img = None
    else:
        pred_successes[index] = True
    
    # Compare rendered images
    img_exact_matches[index] = images_are_equal(gt_img, pred_img)
        

def evaluate_results(results_path, evaluation_save_path, model_name, dataset_info, convert_format=False, eval_img=True):
    """Loads inference results, computes evaluation metrics, and saves them to a file."""
    start_time = time.time()
    results = load_results(results_path)
    total = len(results)
    gt_successes = np.zeros(total, dtype=bool)
    pred_successes = np.zeros(total, dtype=bool)
    img_exact_matches = np.zeros(total, dtype=bool)
    latex_exact_matches = np.zeros(total, dtype=bool)
    anls_scores = np.zeros(total)
    incorrect_samples = []
    
    for index, result in enumerate(tqdm(results, desc="Evaluating Metrics")):
        gt = result['gt']
        pred = result['pred']
        pred = normalize_latex(pred, convert_format=convert_format)
        gt = normalize_latex(gt, convert_format=False)
        # print(gt)
        # print(pred)
        # Normalize LaTeX format before comparison if enabled
        latex_exact_matches[index] = is_latex_exact_match(gt, pred)
        
        if eval_img:
            # Render LaTeX formulas into images
            eval_image(index, gt, pred, gt_successes,pred_successes,img_exact_matches)
            # Track incorrect samples
            if not img_exact_matches[index]:
                incorrect_samples.append(result['id'])
        
        # Compute ANLS score
        # anls_scores[index] = anls_score(prediction=pred, gold_labels=[gt], threshold=0.5)
    
    valid_mask = gt_successes
    num_valid = np.sum(gt_successes)
    
    # Compute evaluation metrics only for valid ground truth renders
    if num_valid > 0:
        render_accuracy = np.sum(img_exact_matches[valid_mask]) / num_valid
        latex_accuracy = np.sum(latex_exact_matches[valid_mask]) / num_valid
        avg_anls = np.sum(anls_scores[valid_mask]) / num_valid
        pred_success_rate = np.sum(pred_successes[valid_mask]) / num_valid
    else:
        render_accuracy = 0.0
        latex_accuracy = 0.0
        avg_anls = 0.0
        pred_success_rate = 0.0
    
    runtime = time.time() - start_time
    incorrect_sample_ids = random.sample(incorrect_samples, min(10, len(incorrect_samples)))
    
    evaluation_results = {
        "model_name": model_name,
        "dataset_info": dataset_info,
        "total_samples": total,
        "prediction_normalized": convert_format,
        "render_success_rate": round(pred_success_rate, 2),
        "render_exact_match_accuracy": round(render_accuracy, 2),
        "latex_exact_match_accuracy": round(latex_accuracy, 2),
        "anls_score": round(avg_anls, 2),
        "evaluation_runtime_seconds": round(runtime, 2),
        "incorrect_sample_ids": incorrect_sample_ids
    }
    
    with open(evaluation_save_path, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    
    print(json.dumps(evaluation_results, indent=4))

def main():
    """Main execution function handling argument parsing and pipeline execution."""
    parser = argparse.ArgumentParser(description="Optimized Model Evaluation Pipeline")
    parser.add_argument("--inference-only", action="store_true", help="Run model inference without evaluation")
    parser.add_argument("--evaluate-only", action="store_true", help="Calculate evaluation metrics without inference")
    parser.add_argument("--inference-results", type=str, default="baseline_inference_results.json", help="Filename to save inference results")
    parser.add_argument("--evaluation-results", type=str, default="baseline_evaluation_results.json", help="Filename to save evaluation results")
    parser.add_argument("--convert-format", action="store_true", help="Convert $$...$$ to align environment in evaluation")
    parser.add_argument("--eval-lora", action="store_true", help="Evaluated model trained with lora")
    parser.add_argument("--load-unimer", action="store_true", help="Evaluated on UniMER Test subset")
    parser.add_argument("--load-hme", action="store_true", help="Evaluated on HME Test subset")
    parser.add_argument("--lora-ckpt", type=str, default="/home/shiqilin/llm/LLaMA-Factory/saves/qwen2.5_vl-3b-v8-cont-3/lora/sft/checkpoint-6000", help="LoRA checkpoint path")
    args = parser.parse_args()

    model_name = "./Qwen2.5-VL-3B-Instruct-Pretrained"
    lora_ckpt = args.lora_ckpt
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {args.inference_results: args.inference_results}
    if not args.evaluate_only:
        print("Loading model...")
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True, 
            trust_remote_code=True, attn_implementation="flash_attention_2"
        )
        if args.eval_lora:
            print("Applying LoRA checkpoint...")
            model = PeftModel.from_pretrained(model, lora_ckpt)
        model.to(device)
        model.eval()
        print("Loading dataset...")
        dataset, dataset_name = load_evalset(args.load_unimer, args.load_hme)
        results = inference(processor, model, dataset, args.inference_results, dataset_name)
     
    if not args.inference_only:
        model_info = lora_ckpt if args.eval_lora else model_name
        convert_format = True if args.eval_lora else False
        for set_id, result in results.items():
            eval_save_path = args.evaluation_results if args.evaluation_results else result.replace('inference', 'evaluation')
            eval_by_batch(result, eval_save_path, model_info, set_id, convert_format)

        

if __name__ == "__main__":
    main()
