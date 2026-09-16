"""IFEval 全量（指令遵循维度，官方 89.8）。

与 mmlu_prox 同法：生成式任务，关闭 thinking。
IFEval 输出为 verbose 模式，不依赖正则抽取，但 thinking 前导会稀释
prompt 指令内容（strict 匹配对指令跟随质量敏感），故关 thinking 后
模型更专注执行指令本身。max_gen_toks 用 2048 对齐其他任务。
"""
from __future__ import annotations

import json
import os

from lm_eval import simple_evaluate


def main() -> None:
    tasks = [t.strip() for t in os.environ.get(
        "IFEVAL_TASKS", "ifeval").split(",") if t.strip()]
    limit = int(os.environ.get("IFEVAL_LIMIT", "0")) or None

    model_args = ("pretrained=/data/yucheng/madm-llm/models/Qwen3.5-4B,"
                  "dtype=bfloat16,enable_thinking=False,think_end_token=</think>")
    if os.environ.get("LORA_ADAPTER"):
        model_args += f",peft={os.environ['LORA_ADAPTER']}"

    print(f"### IFEval 生成式·关thinking limit={limit or '全量'}", flush=True)
    res = simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        batch_size=8,
        device="cuda",
        apply_chat_template=True,
        limit=limit,
        gen_kwargs={"max_gen_toks": 2048, "temperature": 0.0},
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
