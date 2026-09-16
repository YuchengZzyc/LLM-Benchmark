"""_probe_adapter.py — base+adapter 加载探针（验证后可删）。

验证三件事：
1. PeftModel 能把 qwen35_4b_lora_v3.2 挂到服务器 base 上（模块名对得上）；
2. adapter 确实改变前向输出（同 prompt 下 base 与 base+adapter logits 不同
   —— 完全相同说明 adapter 没生效）；
3. 能正常生成一段文本。
"""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "/data/yucheng/madm-llm/models/Qwen3.5-4B"
ADAPTER = "/data/yucheng/madm-llm/madm-llm/output/qwen35_4b_lora_v3.2"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16).to("cuda")
model.eval()

ids = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda")
with torch.no_grad():
    base_logits = model(ids).logits[0, -1, :].float()

model = PeftModel.from_pretrained(model, ADAPTER).to("cuda")
model.eval()
n_adapter = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
print(f"adapter params (lora_*): {n_adapter/1e6:.2f}M")

with torch.no_grad():
    ada_logits = model(ids).logits[0, -1, :].float()

diff = (base_logits - ada_logits).abs().max().item()
print(f"max |logit diff| base vs base+adapter: {diff:.4f}")
print("top tok base:", tok.decode(base_logits.argmax()),
      "| adapter:", tok.decode(ada_logits.argmax()))

out = model.generate(ids, max_new_tokens=16, do_sample=False,
                     pad_token_id=tok.eos_token_id)
print("GEN:", repr(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)))
print("ADAPTER_PROBE_OK" if diff > 1e-3 else "ADAPTER_NOT_EFFECTIVE")
