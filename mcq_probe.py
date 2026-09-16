"""global_mmlu_zh(MMMLU-zh) MCQ 口径探针。

官方 76.1 是 loglikelihood 选字母还是生成式抽取？两种假设都便宜可测：
  A) 纯文本 MCQ：apply_chat_template=False（不带 thinking 块）
  B) chat + 关 thinking：apply_chat_template=True + enable_thinking=False
本轮 A/B 对照，2 分钟出数。若 A 接近 76 → 官方是纯文本 MCQ；若都 40 左右 →
loglikelihood 口径本身不行，需做生成式变体（与 mmlu_prox 同法）。
"""
from __future__ import annotations

import json
import os


def run(tasks, extra="", apply_ct=False, limit=None):
    from lm_eval import simple_evaluate

    model_args = "pretrained=/data/yucheng/madm-llm/models/Qwen3.5-4B,dtype=bfloat16"
    if extra:
        model_args += "," + extra
    if os.environ.get("LORA_ADAPTER"):
        model_args += f",peft={os.environ['LORA_ADAPTER']}"
    res = simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        batch_size=8,
        device="cuda",
        apply_chat_template=apply_ct,
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
    return out


def main() -> None:
    tasks = ["global_mmlu_zh"]
    print("### A) 纯文本 MCQ（apply_chat_template=False）", flush=True)
    a = run(tasks, apply_ct=False)
    print(json.dumps(a, ensure_ascii=False, indent=2), flush=True)

    print("### B) chat + 关 thinking（当前 v0.1 口径）", flush=True)
    b = run(tasks, extra="enable_thinking=False,think_end_token=</think>",
            apply_ct=True)
    print(json.dumps(b, ensure_ascii=False, indent=2), flush=True)

    print("### 对照", flush=True)
    for t in tasks:
        aa = a.get(t, {}).get("acc,none")
        bb = b.get(t, {}).get("acc,none")
        print(f"  {t}: A(纯文本)={aa}  B(chat关thinking)={bb}", flush=True)


if __name__ == "__main__":
    main()
