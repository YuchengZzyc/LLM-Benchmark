# Evaluation Report — Qwen3.5-4B v0.1（口径修复基线）

> Generated at: 2026-09-11T14:40:00Z
> Config: `configs/baseline_qwen35_4b.yaml` + 口径修复（enable_thinking=False + 纯文本 MCQ）
> 版本：`qwen35-4b-base-v0.1`（未微调基座，当前基线）
> 详细分析见 `reports/results_qwen35-4b-base-v0.1_2026-09-11.md`

---

## Model Information

| Field | Value |
| --- | --- |
| Model | Qwen3.5-4B |
| Version | `qwen35-4b-base-v0.1` |
| Type | baseline |
| Path | `/data/yucheng/madm-llm/models/Qwen3.5-4B` |
| Dtype | bfloat16 |

## Environment

| Field | Value |
| --- | --- |
| OS | Linux |
| Python | 3.11.16 |
| PyTorch | 2.11.0+cu128 |
| transformers | 5.6.0 |
| lm_eval | 0.4.13 |
| GPU | NVIDIA GeForce RTX 5090 |
| HF Endpoint | hf-mirror.com |

## Method（口径修复）

| 项 | 值 |
| --- | --- |
| thinking | 关闭：`enable_thinking=False, think_end_token=</think>` |
| 生成式任务 | 生成 + 正则抽取 `答案是\(?([ABCDEFGHIJ])\)?` |
| loglikelihood 任务 | `apply_chat_template=False` 纯文本 MCQ（官方口径） |
| 种子 | 42（四路） |

## 实测结果 vs 官方

| Dataset | Samples | Metric | v0.1 | Official | Gap |
| --- | --- | --- | --- | --- | --- |
| GSM8K | 1319 | exact_match (strict) | **0.90296** | 0.889 | +0.014 |
| C-Eval (valid) | 1346 | acc | **0.74963** | 0.851 | −0.101 |
| MMMLU-zh | 400 | acc | **0.71750** | 0.761 | −0.044 |
| MMLU-ProX (zh) | 840 | exact_match (custom-extract) | **0.63452** | 0.715 | −0.080 |
| IFEval | 541 | inst_level_loose_acc | **0.89928** | 0.898 | +0.001 |
| MGSM (zh) | 250 | exact_match (flexible-extract) | **0.764** | — | — |
| AA-LCR v1.1 | 100 | accuracy（gpt-5.6-luna 判题） | **0.590** | 0.570 | +0.020 |

长上下文参考分：LongBench v1 × 12（30 样本/任务，不作验收口径）见
`result.json` `metrics.dimensions.long_context` 与正式报告 §3.2。

## 可选后续（非协议内，全流程已跑通）

- MMMLU-zh 全量 57 科（确认 400 采样差）
- C-Eval 官方 description 模板验证（差 10 点假设）
- MMLU-Redux（4115）
