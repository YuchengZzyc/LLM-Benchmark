"""C-Eval 复现官方 85.1（纯文本 MCQ 口径，thinking 天然无污染）。

global_mmlu_zh 已验证：chat 模板会让 loglikelihood 选择题掉到 40%，
纯文本 `apply_chat_template=False` 恢复到 71.75。C-Eval 同是 loglikelihood
选择题，官方 85.1 应走纯文本口径。

可选 CEVAL_LIMIT=n 按科采样（默认全量 52 科）；CEVAL_TASKS 覆盖任务名。
"""
from __future__ import annotations

import json
import os

from lm_eval import simple_evaluate


def main() -> None:
    tasks = [t.strip() for t in os.environ.get(
        "CEVAL_TASKS", "ceval-valid").split(",") if t.strip()]
    limit = int(os.environ.get("CEVAL_LIMIT", "0")) or None
    nshot = int(os.environ.get("CEVAL_FEWSHOT", "5"))

    model_args = ("pretrained=/data/yucheng/madm-llm/models/Qwen3.5-4B,"
                  "dtype=bfloat16")

    print(f"### C-Eval 纯文本 MCQ {nshot}-shot limit={limit or '全量'}", flush=True)
    res = simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        batch_size=8,
        device="cuda",
        apply_chat_template=False,
        num_fewshot=nshot,
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
