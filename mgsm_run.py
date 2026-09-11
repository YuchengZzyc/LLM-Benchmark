"""MGSM 中文原生 CoT（多语言推理维度，官方对标 88.9 系 GSM8K 的扩展）。

与 gsm8k_run.py 同法：生成式任务，关闭 thinking。
lm-eval 任务名 mgsm_native_cot_zh（juletxara/mgsm 中文子集，0-shot native CoT，
生成 逐步解答: + 正则抽取 "答案是 (-?[0-9.,]+)。"）。
关 thinking 后模型直接输出逐步解答，避免 400-6000 字 thinking 前导撑爆 max_gen_toks。
"""
from __future__ import annotations

import json
import os

from lm_eval import simple_evaluate


def main() -> None:
    tasks = [t.strip() for t in os.environ.get(
        "MGSM_TASKS", "mgsm_native_cot_zh").split(",") if t.strip()]
    limit = int(os.environ.get("MGSM_LIMIT", "0")) or None

    model_args = ("pretrained=/data/yucheng/madm-llm/models/Qwen3.5-4B,"
                  "dtype=bfloat16,enable_thinking=False,think_end_token=</think>")

    print(f"### MGSM 中文原生CoT·关thinking limit={limit or '全量'}", flush=True)
    res = simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        batch_size=8,
        device="cuda",
        apply_chat_template=True,
        limit=limit,
        gen_kwargs={"max_gen_toks": 1024, "temperature": 0.0},
        random_seed=42,
        numpy_random_seed=42,
        torch_random_seed=42,
        fewshot_random_seed=42,
    )
    out = {}
    for t in tasks:
        tr = (res or {}).get("results", {}).get(t) or {}
        out[t] = {k: v for k, v in tr.items() if isinstance(v, (int, float))}
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
