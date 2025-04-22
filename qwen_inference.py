"""
Inference Pipeline for Qwen2.5-VL

Requirements (install via pip):
    pip install transformers datasets openai pillow matplotlib requests
"""

import argparse
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
import torch

def run_model(processor, model, image):
    """
    Run inference on a single QA pair using the Qwen2.5-VL inference pipeline.
    
    Parameters:
      - image: a PIL Image object. If the qa_pair contains an image part,
        we will replace it with the provided `image` object.
      - qa_pair: list of conversation turns (each a dict with "role" and "content").
      - include_history: if True, include the entire conversation history as context;
        otherwise, use only the last user turn.
      
    Returns:
      - Generated answer as a string.
    """
    messages = []
    sys_prompt = 'You are a helpful assistant.'
    instruction = 'Extract the formulas in the image, and output in Latex format.'

    content_list = [{"type": "image", "image": image}, {"type": "text", "text": instruction}]
    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": content_list}]

    # Prepare the text prompt using the processor's chat template.
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # print(text_prompt)
    """
    print(text_prompt)
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    <|vision_start|><|image_pad|><|vision_end|>Extract the formulas in the image, and output in Latex format.<|im_end|>
    <|im_start|>assistant
    """
    
    # Process visual inputs (images/videos) from the messages.
    image_inputs, video_inputs = process_vision_info(messages)

    # Create the full inputs for the model.
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    """
    inputs =
    {'input_ids': tensor, 
    'attention_mask': tensor, 
    'pixel_values': tensor, 
    'image_grid_thw': tensor}
    """
    inputs = inputs.to(model.device)
    with torch.no_grad():
      # Inference: Generate output tokens.
      generated_ids = model.generate(**inputs, max_new_tokens=4096)
      
    # Trim the generated ids by removing the prompt portion.
    # Assuming a single example; adjust accordingly for batching.
    prompt_length = inputs.input_ids.shape[1]
    generated_ids_trimmed = generated_ids[:, prompt_length:]
    
    # Decode the generated tokens.
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Pipeline for Qwen2.5-VL")
    parser.add_argument("--lora-ckpt", type=str, default="", help="Lora Checkpoint Path")
    parser.add_argument("--dataset", type=str, default="", help="Dataset Path")
    parser.add_argument("--data-split", type=str, default="test", help="Dataset Path")
    parser.add_argument("--save-dir", type=str, default="inference-results", help="Path to save rendered results")
    parser.add_argument("--image-path", type=str, help="Path to test image")
    args = parser.parse_args()

    # import patch_qwen
    import os
    import io
    import torch
    from PIL import Image
    from datasets import load_dataset
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel
    from quick_latex import *

    os.makedirs(args.save_dir, exist_ok=True)
    model_name = "./Qwen2.5-VL-3B-Instruct-Pretrained"
    lora_ckpt = args.lora_ckpt
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_info = args.dataset
    eval_lora = True if args.lora_ckpt else False
    print("Loading model...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True, 
        trust_remote_code=True, attn_implementation="flash_attention_2"
    )
    if eval_lora:
        print("Applying LoRA checkpoint...")
        model = PeftModel.from_pretrained(model, lora_ckpt)
    model.to(device)
    model.eval()  
    
    if dataset_info:
      if eval_lora:
       run_dir = lora_ckpt.split('/')[-1] + '_' + dataset_info
      else:
         run_dir = 'pretrained_' + dataset_info
      save_path = os.path.join(args.save_dir, run_dir)
      os.makedirs(save_path, exist_ok=True)

      print(f"Loading the {dataset_info} dataset - split {args.data_split}...")
      dataset = load_dataset(dataset_info, split=f"{args.data_split}", trust_remote_code=True)
      print("Dataset loaded with", len(dataset), "samples.")
      for sample in tqdm(dataset, desc="Model Inference:"):
        # Each sample should have an "images" field and "latex_formula" field.
        image_item = sample["image"]
        if isinstance(image_item, dict):
           image_item = image_item['bytes']
           image_item = Image.open(io.BytesIO(image_item))
        gt = sample["latex_formula"]
        pred = run_model(processor, model, image_item)
        print('Ground Truth: \n',gt)
        print('Prediction: \n',pred)
        render_latex_quicklatex(gt, os.path.join(save_path, f'{sample['id']}_gt.png'))
        render_latex_quicklatex(pred, os.path.join(save_path, f'{sample['id']}_pred.png'))
    elif os.path.exists(args.image_path):
      if eval_lora:
        run_dir = lora_ckpt.split('/')[-1] + '_results'
      else:
        run_dir = 'pretrained_results'
      save_path = os.path.join(args.save_dir, run_dir)
      os.makedirs(save_path, exist_ok=True)

      image_path = args.image_path
      if '.HEIC' in args.image_path:
        import pyheif
        heif_file = pyheif.read(image_path)
        image_item = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )
      else: image_item = Image.open(image_path)
      print(f'Running inference for {image_path}')
      pred = run_model(processor, model, image_item)
      print('Prediction: \n',pred)
      render_latex_quicklatex(pred, os.path.join(save_path, f'{image_path.split('/')[-1].split('.')[0]}_pred.png'))

