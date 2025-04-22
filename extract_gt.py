import os
import pyarrow.parquet as pq
import json

def strip_latex_wrappers(latex):
    import re
    return re.sub(r'\\begin\{.*?\}|\s*\\end\{.*?\}', '', latex).strip()

def extract_parquet_to_jsonl(parquet_path, output_jsonl):
    table = pq.read_table(parquet_path)
    ids = table.column("id").to_pylist()
    latexes = table.column("latex_formula").to_pylist()

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for _id, latex in zip(ids, latexes):
            if latex:
                clean_latex = strip_latex_wrappers(latex)
                f.write(json.dumps({"id": str(_id), "latex": clean_latex}, ensure_ascii=False) + "\n")

    print(f"Extracted {output_jsonl}")

def extract_gt():
    root_dir = "latex-formulas-processed"
    extract_parquet_to_jsonl(f"{root_dir}/train.parquet", "raw_train.jsonl")
    extract_parquet_to_jsonl(f"{root_dir}/validation.parquet", "raw_val.jsonl")
    extract_parquet_to_jsonl(f"{root_dir}/test.parquet", "raw_test.jsonl")
