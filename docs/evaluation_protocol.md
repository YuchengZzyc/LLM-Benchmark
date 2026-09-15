# 评测协议（Evaluation Protocol）

> **当前阶段：v0.1 基线评测完成（五维全闭环，2026-09-11）。**
> 本文档定义版本管理规则与运行流程的框架性协议；**冻结的评测口径（维度↔数据集↔
> 指标↔参数）以 [evaluation_spec.md](evaluation_spec.md) 为唯一权威**，
> 与本文冲突处以 spec 为准。

本文档定义 Qwen3.5-4B 系列模型的通用能力评测协议：评测目标、能力维度、
benchmark 说明、版本管理规则与后续运行流程。

---

## 1. 评测目标

对 Qwen3.5-4B 及其后续实验版本（LoRA / DPO / 其他）进行**通用能力**的
系统化、可复现、可横向比较的评测。

- **当前基线**：`qwen35-4b-base-v0.1`（未微调基座，口径修复版；v0.0 为污染基线）
- **目标**：所有版本必须在同一协议、同一配置体系、同一结果管理体系下评测，
  保证结果可比。

## 2. 能力维度（冻结映射见 [evaluation_spec.md](evaluation_spec.md) §1）

| 能力维度 | 数据集（v0.1 冻结口径） | 指标 | 状态 |
| --- | --- | --- | --- |
| 1. 知识与理解 | C-Eval valid、MMLU-ProX (zh) | acc / exact_match | ✅ 已测 |
| 2. 专业推理 | GSM8K | Exact Match (strict/flexible) | ✅ 已测 |
| 3. 指令遵循 | IFEval | strict / loose accuracy | ✅ 已测 |
| 4. 长上下文 | **AA-LCR v1.1**（独立接口，官方协议）；LongBench v1 × 12 参考分 | LLM 判题 accuracy | ✅ 已测 |
| 5. 多语言 | MGSM (zh, 原生 CoT), MMMLU (zh) | exact_match / acc | ✅ 已测 |

> 历史注记：v0.0 时代曾按下表候选集配置（MMLU-Pro / GPQA / AIME / LongBench v2 等），
> 2026-09-11 冻结时收敛为上表六集 + AA-LCR（取舍记录见 spec §1.2 与
> [aa_lcr_plan.md](aa_lcr_plan.md)）。

各 benchmark 的来源、所测能力与指标定义见
[docs/benchmark_description.md](benchmark_description.md)。

## 3. 配置系统

所有运行参数由 `configs/*.yaml` 管理。**禁止**在 Python 代码中硬编码：

- 模型路径
- benchmark 名称
- batch size
- dtype
- 输出路径

模板见 [configs/template.yaml](../configs/template.yaml)，基线示例见
[configs/baseline_qwen35_4b.yaml](../configs/baseline_qwen35_4b.yaml)。

配置通过 [src/config_loader.py](../src/config_loader.py) 加载、校验并统一为
`EvalConfig` 对象。校验规则包括：必填段（model / evaluation / tasks / output）、
`model.version` 不含路径分隔符、`model.dtype` 合法取值等。

## 4. 版本管理规则

版本由 `model.version` 唯一标识，登记于 [results/registry.yaml](../results/registry.yaml)。

### 4.1 命名规范

```
qwen35-4b-<type>-v<major>.<minor>
```

| type | 含义 | 示例 |
| --- | --- | --- |
| base | 未微调基座 | qwen35-4b-base-v0.0 |
| lora | LoRA 微调 | qwen35-4b-lora-v1.0 |
| dpo | DPO 对齐 | qwen35-4b-dpo-v2.0 |

### 4.2 结果目录

每次运行写入 **版本化目录**，互不覆盖：

```
results/<version>/
├── config.yaml      # 本次运行配置快照
├── logs/            # 运行日志
├── samples/         # 模型样本输出（可选）
├── result.json      # 结构化结果
└── report.md        # Markdown 报告
```

registry 记录的 `results_path` 指向该目录。**任何运行都不得覆盖旧版本的结果**。

### 4.3 状态机

`status` 取值：`initialized` → `pending` → `running` → `completed` | `failed`

## 5. 技术方案

- **主框架**：`lm-eval`（通用能力评测）
- **长上下文**：独立接口（`scripts/aa_lcr_run.py`，AA-LCR 官方协议 +
  LLM 判题），不强行绑定主框架；`scripts/run_longbench.py` 供 LongBench 参考评测
- **模型加载**：支持 HuggingFace 模型名 / 本地模型路径；可配置 chat template、
  batch size、dtype

## 6. 后续运行流程（服务器阶段）

### 6.0 前置条件

- Linux 服务器，GPU 可用，CUDA 环境就绪（见 [environment_setup.md](environment_setup.md)）
- 配置中 `model.path` 指向真实模型路径

### 6.1 安装环境（一次性）

```bash
bash scripts/install_env.sh
```

### 6.2 配置

```bash
cp configs/template.yaml configs/baseline_qwen35_4b.yaml
# 编辑 configs/baseline_qwen35_4b.yaml：填写真实 model.path
```

### 6.3 运行主评测

```bash
bash scripts/run_eval.sh --config configs/baseline_qwen35_4b.yaml
```

### 6.4 生成报告

```bash
python scripts/generate_report.py --config configs/baseline_qwen35_4b.yaml
```

### 6.5 长上下文独立评测（正式：AA-LCR）

长上下维度的**正式评测**走独立接口 `scripts/aa_lcr_run.py`（不绑定 lm-eval），
按 ArtificialAnalysis 官方 v1.1 协议生成并用 `gpt-5.6-luna` 判题：

```bash
nohup env JUDGE_API_KEY="$JUDGE_API_KEY" \
    python scripts/aa_lcr_run.py --config configs/aa_lcr_qwen35_4b_v01.yaml \
    > /data/yucheng/aa_lcr_run.log 2>&1 &
```

完整协议（数据部署、prompt 模板、判题器、时间预估）见
[aa_lcr_plan.md](aa_lcr_plan.md) 与 [evaluation_spec.md](evaluation_spec.md) §1.1。

### 6.6 LongBench 参考评测（非验收口径）

`scripts/run_longbench.py` 仅供 LongBench v1 参考评测（v0.1 已跑 × 12 任务、
每任务 30 样本，分数并入 result.json 但**不参与与官方对齐**）：

```bash
# 默认：按配置 tasks.long_context 运行
python scripts/run_longbench.py --config configs/longbench_qwen35_4b_v01.yaml

# 指定子任务 / 覆盖上下文长度 / 保存样本
python scripts/run_longbench.py --config configs/longbench_qwen35_4b_v01.yaml \
    --tasks hotpotqa 2wikimqa --max-length 32768 --keep-samples
```

- 任务名、数据目录（`evaluation.longbench_data_dir`）均在 yaml 中配置，代码不硬编码。
  ⚠️ **当前环境 `datasets` 为 4.0.0，已移除数据集脚本加载器**，`null`（走 HuggingFace
  在线加载 `THUDM/LongBench`）**不再可用**，会报
  `Dataset scripts are no longer supported, but found LongBench.py`；
  须设为本地官方 JSONL 目录，如 `longbench_data_dir: data/longbench`。
- 指标按官方口径：问答 F1/EM、摘要 ROUGE-L、合成/检索 accuracy；
  代码类子任务（lcc / repobench-p）需官方 code eval（单元测试执行），
  当前标记 `not_supported`，不生成虚假分数。
- 结果合并写入 `results/<version>/result.json`（与主框架共用），
  `--keep-samples` 时样本存 `samples/longbench_<task>.json`。

### 6.6 多版本横向比较

新增版本（如 `qwen35-4b-lora-v1.0`）：复制配置模板、修改 `model.version`、
运行评测。registry 将自动登记新版本，报告 Comparison 章节汇总所有版本。

## 7. 可复现性

- 固定随机种子（`evaluation.seed`，默认 42）
- 每次运行保存配置快照 `config.yaml` 与运行日志
- 环境信息记录于 `environment.txt`，并在报告中回填
