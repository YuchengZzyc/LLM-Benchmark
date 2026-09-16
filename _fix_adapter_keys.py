"""_fix_adapter_keys.py — 修正 adapter checkpoint 键名（验证后可删）。

问题：qwen35_4b_lora_v3.2 训练时模型经多模态壳类加载，LoRA 键名主干为
  base_model.model.model.language_model.layers...lora_A.weight
而评测侧 AutoModelForCausalLM + PEFT 期望
  base_model.model.model.layers...lora_A.weight
（PEFT 保存格式本就不含 .default 适配器名段，加载时由 set_peft_model_state_dict
内部插入；唯一错位是 language_model. 主干段。）
PEFT 按名加载全部 miss → adapter 不生效（logit diff = 0）。

本脚本把权重重映射到新目录 <src>-peft（不改动原始训练产物），并在 GPU 上
自验：挂载后 base 与 base+adapter 的 logits 必须有差异。
"""
import json
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SRC = Path("/data/yucheng/madm-llm/madm-llm/output/qwen35_4b_lora_v3.2")
DST = SRC.with_name(SRC.name + "-peft")
BASE = "/data/yucheng/madm-llm/models/Qwen3.5-4B"

PAT = re.compile(r"^base_model\.model\.(.+)\.lora_(A|B)\.weight$")

sd = load_file(str(SRC / "adapter_model.safetensors"))
out, bad = {}, []
for k, v in sd.items():
    m = PAT.match(k)
    if not m:
        bad.append(k)
        continue
    body, ab = m.groups()
    if body.startswith("model.language_model."):
        body = "model." + body[len("model.language_model."):]
    else:
        bad.append(k)
        continue
    out[f"base_model.model.{body}.lora_{ab}.weight"] = v

if bad:
    print(f"未匹配键 {len(bad)} 个，示例: {bad[:5]}")
    sys.exit(1)

DST.mkdir(exist_ok=True)
save_file(out, str(DST / "adapter_model.safetensors"))
shutil.copy(SRC / "adapter_config.json", DST / "adapter_config.json")
for f in ("README.md", "all_results.json"):
    if (SRC / f).exists():
        shutil.copy(SRC / f, DST / f)
print(f"重映射 {len(out)} 键 -> {DST}")

# ---- GPU 自验：adapter 必须真实改变前向输出 ----
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16).to("cuda")
model.eval()
ids = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda")
with torch.no_grad():
    base_logits = model(ids).logits[0, -1, :].float()

model = PeftModel.from_pretrained(model, str(DST)).to("cuda")
model.eval()
loaded = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n and "default" in n)
with torch.no_grad():
    ada_logits = model(ids).logits[0, -1, :].float()

diff = (base_logits - ada_logits).abs().max().item()
print(f"lora 参数量: {loaded/1e6:.2f}M | max |logit diff|: {diff:.4f}")
print("top tok base:", tok.decode(base_logits.argmax()),
      "| adapter:", tok.decode(ada_logits.argmax()))
out_g = model.generate(ids, max_new_tokens=16, do_sample=False,
                       pad_token_id=tok.eos_token_id)
print("GEN:", repr(tok.decode(out_g[0][ids.shape[1]:], skip_special_tokens=True)))
print("FIX_OK" if diff > 1e-3 else "STILL_NOT_EFFECTIVE")
