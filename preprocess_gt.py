import json
from tqdm import tqdm
import os
import sys
from quick_latex import *
from PIL import Image

from cdm.modules.tokenize_latex import tokenize_latex

import re

def smart_unwrap_align(latex: str) -> str:
    """
    Remove \begin{align*}...\end{align*} if no multi-line or alignment found.
    """
    # Extract inner content
    # content = re.sub(r'\\begin\{align\*\}', '', latex)
    # content = re.sub(r'\\end\{align\*\}', '', content).strip()

    has_multiline = "\\\\" in latex
    has_alignment = "&" in latex

    if has_multiline or has_alignment:
        # Real multi-line align → keep align* wrapper
        # print('find multiline: ', latex)
        # return "\\begin{align*}"+f"{latex}"+"\\end{align*}"
        return f"\\begin{{align*}} {latex} \\end{{align*}}"

        # return f"\\begin{{array}}{{rl}}{latex}\\end{{array}}"
    else:
        # Just a single line formula, unwrap align
        return latex



def tokenize_jsonl(input_path, output_path, broken_id_path="broken_ids.jsonl"):
    processed_ids = set()
    broken_ids = set()

    # Load already processed entries
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("latex", "").strip():
                        processed_ids.add(entry["id"])
                except:
                    continue

    # Load previously broken IDs from JSONL
    if os.path.exists(broken_id_path):
        with open(broken_id_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    broken_ids.add(json.loads(line)["id"])
                except:
                    continue

    print(f"Resuming: {len(processed_ids)} processed, {len(broken_ids)} broken IDs skipped.")

    # Process new entries
    with open(input_path, "r", encoding="utf-8") as in_file, \
         open(output_path, "a", encoding="utf-8") as out_file:

        for line in tqdm(in_file, desc=f"Tokenizing {os.path.basename(input_path)}"):
            try:
                entry = json.loads(line)
                _id = entry["id"]

                if _id in processed_ids or _id in broken_ids:
                    continue

                latex = entry.get("latex", "").strip()
                if not latex:
                    # Empty LaTeX → log as broken
                    broken_ids.add(_id)
                    with open(broken_id_path, "a", encoding="utf-8") as bf:
                        bf.write(json.dumps({"id": _id}) + "\n")
                    continue

                latex = smart_unwrap_align(latex)
                # print(f"processing id {_id}: {latex}")
                # Ground Truth Tokenization
                tokenized = tokenize_latex(latex)
                # Latex Renderer Check
                renderer_passed = isinstance(render_latex_quicklatex(latex), Image)
    
                if tokenized[0] and renderer_passed:
                    # print("tokenization succeeded")
                    result = {"id": _id, "latex": tokenized[1]}
                    out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    processed_ids.add(_id)
                else:
                    broken_ids.add(_id)
                    with open(broken_id_path, "a", encoding="utf-8") as bf:
                        bf.write(json.dumps({"id": _id}) + "\n")

            except Exception as e:
                print(f"[Error] id={entry.get('id', '???')}: {e}")
                if entry.get("id") not in broken_ids:
                    broken_ids.add(entry["id"])
                    with open(broken_id_path, "a", encoding="utf-8") as bf:
                        bf.write(json.dumps({"id": entry["id"]}) + "\n")
                continue

    print(f"Finished tokenizing. Output written to: {output_path}")
    print(f"Broken IDs continuously saved to: {broken_id_path}")

def remove_empty_latex(jsonl_path, keep_num=100):
    lines = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(line) == keep_num:
                break
            try:
                obj = json.loads(line)
                if obj.get("latex", "").strip():
                    lines.append(json.dumps(obj, ensure_ascii=False))
            except:
                continue
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def preprocess_gt():
    tokenize_jsonl("raw_train.jsonl", "tokenized_train.jsonl")
    remove_empty_latex("tokenized_train.jsonl", keep_num=53000)
    tokenize_jsonl("raw_val.jsonl", "tokenized_val.jsonl")
    remove_empty_latex("tokenized_val.jsonl", keep_num=800)
    tokenize_jsonl("raw_test.jsonl", "tokenized_test.jsonl")
    remove_empty_latex("tokenized_test.jsonl", keep_num=1000)
