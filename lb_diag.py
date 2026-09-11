"""LongBench 低分诊断：用与 run_longbench 相同的 _generate 逻辑生成 3 条，
打印 截断后 input（前 400 字符）+ 生成输出，定位是生成垃圾还是模型真答不对。"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

from src.config_loader import load_config
from src.longbench import _load_task_data, _build_prompt, _truncate_middle, _generate
from src.result_manager import save_sample_payload  # noqa: F401

cfg = load_config("configs/longbench_qwen35_4b_v01.yaml")
samples = _load_task_data(cfg, "multifieldqa_en")[:3]

from src.longbench import _load_model_and_tokenizer
model, tokenizer = _load_model_and_tokenizer(cfg)

for i, sample in enumerate(samples, start=1):
    prompt = _build_prompt(sample)
    tail = str(sample.get("input", "") or "").strip()
    text = _truncate_middle(prompt, tail, tokenizer, int(cfg.evaluation.max_length))
    tok = tokenizer(text, truncation=False)["input_ids"]
    print(f"===== sample {i} | prompt_tokens={len(tok)} | answers={sample['answers'][:2]}")
    print("--- INPUT 前 500 字符 ---")
    print(text[:500])
    print("--- INPUT 后 300 字符 ---")
    print(text[-300:])
    pred = _generate(model, tokenizer, cfg, prompt, 64, tail_text=tail)
    print("--- PRED ---")
    print(repr(pred))
    print("--- PRED 原始(未 strip, 含特殊token保留) ---")
    print(repr(pred[:500]))
    print()
