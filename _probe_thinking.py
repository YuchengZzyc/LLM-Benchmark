# -*- coding: utf-8 -*-
"""诊断探针：Qwen3.5-4B 的 thinking 模板对评测分数的影响。

GSM8K 2 条样本，对比：
  A. 默认 chat template（Qwen3 系默认可能开启 thinking 模式）
  B. enable_thinking=False（若模板支持）
输出原始生成内容的开头/结尾与 strict/flexible 抽取判定，用于实证：
模型会做 vs 评分器抽不到。
"""
import os
import re
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.stdout.reconfigure(encoding="utf-8")

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

M = "/data/yucheng/madm-llm/models/Qwen3.5-4B"
tok = AutoTokenizer.from_pretrained(M)
tpl = tok.chat_template or ""
print("[chat_template 中含 'think' 关键字]:", "think" in tpl.lower())

model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16).to("cuda").eval()
ds = load_dataset("openai/gsm8k", "main", split="test")


def gen(prompt, enable_thinking):
    msgs = [{"role": "user", "content": prompt}]
    kw = dict(tokenize=False, add_generation_prompt=True)
    if enable_thinking is not None:
        kw["enable_thinking"] = enable_thinking
    try:
        text = tok.apply_chat_template(msgs, **kw)
    except Exception as exc:  # 模板不支持该参数则退回默认
        print("  (模板不接受 enable_thinking:", exc, ")")
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=2048, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def judge(text, gold):
    m = re.search(r"####\s*\$?(-?[\d,\.]+)", text)
    nums = re.findall(r"-?\d[\d,\.]*", text)
    strict = m.group(1).replace(",", "").replace("$", "") if m else None
    flex = nums[-1].replace(",", "").replace("$", "") if nums else None
    g = gold.replace(",", "").replace("$", "")
    return strict, (strict == g), flex, (flex == g)


for i in [0, 2]:
    q = ds[i]["question"]
    gold = ds[i]["answer"].split("####")[-1].strip()
    prompt = f"Question: {q}\nAnswer:"   # lm-eval gsm8k 的 doc_to_text 原样
    print(f"\n===== 样本{i} | gold={gold}")
    for mode, et in [("A 默认模板", None), ("B enable_thinking=False", False)]:
        g = gen(prompt, et)
        strict, ok_s, flex, ok_f = judge(g, gold)
        head = re.sub(r"\s+", " ", g[:260])
        tail = re.sub(r"\s+", " ", g[-160:])
        print(f"[{mode}] 长度={len(g)}字符 strict抽取={strict}({'对' if ok_s else '错/未抽到'})"
              f" flexible抽取={flex}({'对' if ok_f else '错'})")
        print(f"   开头: {head}")
        print(f"   结尾: ...{tail}")

print("\n[探针完成]")
