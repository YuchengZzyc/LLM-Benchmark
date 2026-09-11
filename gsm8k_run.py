"""GSM8K 复现官方 88.9（待 mmlu_prox 大采样跑完后启动）。

默认模板（thinking 开）在 2048 tok 截断下会伤生成式抽取（mmlu_prox 已验证）。
GSM8K 同样先跑 enable_thinking=False 对照；若想同时出 thinking 开/关对比，
用 A_B_GROUP 或直接改下方 THINKING。
"""
from __future__ import annotations

import json
import os

from lm_eval import simple_evaluate


def main() -> None:
    tasks = ["gsm8k"]  # lm-eval 0.4.13 叶任务名（cot + strict/flexible filter）
    limit = int(os.environ.get("GSM8K_LIMIT", "0")) or None
    thinking = os.environ.get("GSM8K_THINKING", "false").lower() == "true"

    model_args = ("pretrained=/data/yucheng/madm-llm/models/Qwen3.5-4B,"
                  "dtype=bfloat16")
    if not thinking:
        model_args += ",enable_thinking=False,think_end_token=</think>"

    print(f"### GSM8K thinking={'开' if thinking else '关'} "
          f"limit={limit or '全量'}", flush=True)
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
