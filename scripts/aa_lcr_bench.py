"""aa_lcr_bench.py — AA-LCR 冒烟计时测试（远程，验证 100k 上下文可行性）。

作用：加载 Qwen3.5-4B（bf16），用官方模板构建第 1 题的 ~94k prompt，
计时 prefill+生成 64 tokens，打印峰值显存。决定 eager/sdpa 是否可行，
并给出单题耗时估算 → 验证 5-6h 目标是否成立。
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

import torch

MODEL_PATH = "/data/yucheng/madm-llm/models/Qwen3.5-4B"
CSV_PATH = "/data/yucheng/AA-LCR_Dataset.csv"
DOCS_ROOT = "/data/yucheng/aa_lcr_data/lcr"
MAX_NEW = 64


def load_row(idx: int):
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[idx]


def load_docs(row) -> list[str]:
    category = row["document_category"]
    set_id = row["document_set_id"]
    filenames = [s.strip() for s in row["data_source_filenames"].split(";") if s.strip()]
    docs = []
    for fn in filenames:
        p = os.path.join(DOCS_ROOT, category, set_id, fn)
        if not os.path.exists(p):
            raise FileNotFoundError(f"缺少文档: {p}")
        with open(p, encoding="utf-8") as f:
            docs.append(f.read())
    return docs


def build_prompt(docs: list[str], question: str) -> str:
    documents_text = "\n\n".join(
        f"BEGIN DOCUMENT {i + 1}:\n{doc}\nEND DOCUMENT {i + 1}"
        for i, doc in enumerate(docs)
    )
    return (
        "BEGIN INPUT DOCUMENTS\n\n"
        f"{documents_text}\n\n"
        "END INPUT DOCUMENTS\n\n"
        "Answer the following question using the input documents provided above.\n\n"
        "START QUESTION\n\n"
        f"{question}\n\n"
        "END QUESTION"
    )


def main():
    row = load_row(0)
    docs = load_docs(row)
    prompt = build_prompt(docs, row["question"])
    print(f"question_id={row['question_id']} category={row['document_category']} "
          f"docs={len(docs)} csv_input_tokens={row['input_tokens']}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16).to("cuda")
    print(f"model loaded, attn_impl 默认: {getattr(model.config, 'attn_implementation', 'n/a')}")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    t0 = time.time()
    batch = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", enable_thinking=False, truncation=False)
    print(f"template+tokens: {batch['input_ids'].shape[1]} tokens, {time.time()-t0:.1f}s")
    batch = {k: v.to("cuda") for k, v in batch.items()}

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **batch, max_new_tokens=MAX_NEW, do_sample=False,
            eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id,
            stop_strings=["<|im_end|>"], tokenizer=tok)
    dt = time.time() - t0
    new_tokens = out[0, batch["input_ids"].shape[1]:]
    print(f"generate: {dt:.1f}s ({MAX_NEW} new tok, {(dt/MAX_NEW*1e3):.0f} ms/tok decode)")
    print(f"prefill≈{(dt - MAX_NEW*(dt/MAX_NEW)):.1f}s | peak VRAM: "
          f"{torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    dec = tok.decode(new_tokens, skip_special_tokens=True)
    print("--- 生成(前200) ---")
    print(repr(dec[:200]))
    print(f"全部完成，总计 {(time.time()-t0):.1f}s（含生成）")


if __name__ == "__main__":
    main()
