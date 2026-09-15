# Benchmark Framework — Qwen3.5-4B 通用能力评测框架

> **当前阶段：v0.1 基线评测完成（五维全闭环，全流程已跑通）**
> 权威评测口径见 [docs/evaluation_spec.md](docs/evaluation_spec.md)（冻结 v1.0）；
> 正式报告见 [reports/results_qwen35-4b-base-v0.1_2026-09-11.md](reports/results_qwen35-4b-base-v0.1_2026-09-11.md)。

本仓库是 Qwen3.5-4B 系列模型通用能力评测的可持续维护工程框架：
配置系统、自动化脚本、结果管理（registry / 版本化目录）与文档齐备，
v0.0（污染基线）与 v0.1（口径修复基线）两轮全量评测已完成。

---

## 1. 当前状态（2026-09-15 检查确认）

| 能力维度 | 数据集 | v0.1 实测 | 官方 | 状态 |
| --- | --- | --- | --- | --- |
| 知识与理解 | C-Eval valid（1346，5-shot 纯文本 MCQ） | 0.74963 | 0.851 | ✅ |
| | MMLU-ProX-zh（840 = 60/科×14） | 0.63452 | 0.715 | ✅ |
| 专业推理 | GSM8K（1319 全量，strict） | **0.90296** | 0.889（+1.4） | ✅ |
| 指令遵循 | IFEval（541 全量，inst_level_loose） | **0.89928** | 0.898（+0.1） | ✅ |
| 长上下文 | **AA-LCR v1.1**（100 题全量，gpt-5.6-luna 判题） | **0.590** | 0.570（+2.0） | ✅ |
| | LongBench v1 × 12（30/任务，参考分，不作验收） | 见 result.json | — | ✅ |
| 多语言 | MGSM-zh（250 全量，flexible） | 0.764 | — | ✅ |
| | MMMLU-zh（400 采样，纯文本 MCQ） | 0.71750 | 0.761 | ✅ |

- **五维全部有评测数据，全流程已跑通**；GSM8K / IFEval / AA-LCR 三项与官方
  分数对齐（差距 ≤ 2 点）。
- 版本沿革：v0.0（2026-09-10，chat 模板 thinking 污染，仅作污染基线）→
  **v0.1（2026-09-11，当前基线）**。后续微调版本（LoRA / DPO）在同一冻结协议下追加。

## 2. 评测目标

- **目标模型**：Qwen3.5-4B
- **当前基线**：`qwen35-4b-base-v0.1`（未微调基座，口径修复版）
- **未来版本**：`qwen35-4b-lora-v1.0`、`qwen35-4b-dpo-v2.0` 及其他实验版本

所有版本经由 `results/registry.yaml` 登记，使用**版本化结果目录**，互不覆盖，可横向比较。

## 3. 能力维度与 Benchmark（冻结口径）

| 能力维度 | 数据集（lm-eval 任务 / 独立接口） | 指标 |
| --- | --- | --- |
| 知识与理解 | `ceval-valid`、`mmlu_prox_zh` | acc / exact_match |
| 专业推理 | `gsm8k` | exact_match（strict/flexible） |
| 指令遵循 | `ifeval` | strict / loose accuracy |
| 长上下文 | `scripts/aa_lcr_run.py`（AA-LCR 官方协议）；LongBench v1 参考分 | LLM 判题 accuracy |
| 多语言 | `mgsm_native_cot_zh`、`global_mmlu_zh` | exact_match / acc |

两条核心口径（v0.1 修复结论，详见 spec §2.1）：

1. **生成式任务**（gsm8k / mmlu_prox_zh / ifeval / mgsm）：chat 模板 +
   `enable_thinking=False,think_end_token=</think>`；
2. **loglikelihood 选择题**（ceval-valid / global_mmlu_zh）：
   `apply_chat_template=False` 纯文本 MCQ。

详细说明见 [docs/evaluation_spec.md](docs/evaluation_spec.md)（唯一权威）、
[docs/benchmark_description.md](docs/benchmark_description.md)。

## 4. 技术方案

- **主框架**：lm-eval 0.4.13（`simple_evaluate`，六个数据集）
- **长上下文**：独立接口 `scripts/aa_lcr_run.py`（AA-LCR 官方协议，
  HF generate + gpt-5.6-luna 判题，不绑定 lm-eval）；
  `scripts/run_longbench.py` 供 LongBench 参考评测
- **模型加载**：本地模型路径 / chat template / batch / dtype 全部经 `configs/*.yaml` 管理

## 5. 目录结构

```
Benchmark/
├── README.md
├── requirements.txt / environment.txt
├── gsm8k_run.py / ceval_run.py / ifeval_run.py / mgsm_run.py   # 冻结口径跑测脚本
├── mcq_probe.py / a_b_tasks.py / _probe_thinking.py            # 口径探针（A/B 对照）
├── configs/
│   ├── baseline_qwen35_4b.yaml        # v0.0 主框架配置
│   ├── aa_lcr_qwen35_4b_v01.yaml      # AA-LCR（长上下文正式）
│   ├── longbench_qwen35_4b_v01.yaml   # LongBench 参考评测
│   └── template.yaml                  # 新版本模板
├── scripts/
│   ├── run_lm_eval.py / run_longbench.py / aa_lcr_run.py
│   └── generate_report.py / install_env.sh / run_eval.sh
├── src/                                # 配置加载 / 评测器 / 结果管理
├── results/
│   ├── registry.yaml                   # 版本登记
│   ├── qwen35-4b-base-v0.0/            # 污染基线（历史）
│   └── qwen35-4b-base-v0.1/            # 当前基线
├── reports/                            # 正式报告快照
└── docs/                               # spec（权威）/ runbook / protocol / environment
```

## 6. 运行评测（冻结命令，全部在评测服务器执行）

评测只在服务器 `yucheng@192.168.13.230` 的 `/data/yucheng/madm-llm/Benchmark` 下运行
（本地 Windows 仅编辑/推送）。完整从零复现流程见
[docs/evaluation_spec.md](docs/evaluation_spec.md) §4–§6：

```bash
export HF_ENDPOINT=https://hf-mirror.com
PY=/data/yucheng/madm-llm/.venv/bin/python

# 六个数据集（约 2.5h）
$PY gsm8k_run.py          # GSM8K 全量 1319
$PY ceval_run.py          # C-Eval 5-shot 全量 1346
$PY mcq_probe.py          # MMMLU-zh 纯文本 MCQ（A 组为冻结口径）
A_B_GROUP=B A_B_TASKS=mmlu_prox_zh A_B_LIMIT=60 $PY a_b_tasks.py   # MMLU-ProX-zh 840
$PY ifeval_run.py         # IFEval 全量 541
$PY mgsm_run.py           # MGSM 中文全量 250

# AA-LCR 长上下文（~2h40m 生成 + 判题；JUDGE_API_KEY 走环境变量）
nohup env JUDGE_API_KEY="$JUDGE_API_KEY" $PY scripts/aa_lcr_run.py \
    --config configs/aa_lcr_qwen35_4b_v01.yaml > /data/yucheng/aa_lcr_run.log 2>&1 &

# 生成报告并登记 registry
$PY scripts/generate_report.py --config <version 配置>
```

## 7. 结果管理规则

- 每次运行写入 `results/<version>/` 版本化目录，**不覆盖**旧版本
- `results/registry.yaml` 登记所有模型版本的元信息
- 每个版本目录包含 `config.yaml`（运行配置快照）、`logs/`、`samples/`、`result.json`、`report.md`
- 跨接口结果按维度合并进同一 `result.json`（`src/result_manager.py` `_merge_metrics`）

详见 [docs/evaluation_protocol.md](docs/evaluation_protocol.md)。

## 8. 相关文档

| 文档 | 内容 |
| --- | --- |
| [docs/evaluation_spec.md](docs/evaluation_spec.md) | **唯一权威**：冻结口径、维度映射、复现流程、验收基线 |
| [docs/aa_lcr_plan.md](docs/aa_lcr_plan.md) | AA-LCR 长上下文评测方案与 run 演进 |
| [docs/evaluation_protocol.md](docs/evaluation_protocol.md) | 评测目标、版本管理规则 |
| [docs/evaluation_runbook.md](docs/evaluation_runbook.md) | 维度↔数据集映射、环境、跑法（历史，冲突以 spec 为准） |
| [docs/benchmark_description.md](docs/benchmark_description.md) | 各 benchmark 来源 / 能力 / 指标 |
| [docs/environment_setup.md](docs/environment_setup.md) | Python / CUDA 环境与依赖安装方式 |
