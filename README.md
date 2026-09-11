# Benchmark Framework — Qwen3.5-4B 通用能力评测框架

> **当前阶段：Framework Construction Only（仅框架搭建）**

本仓库是 Qwen3.5-4B 系列模型通用能力评测的可持续维护工程框架。
当前阶段**只**包含目录结构、配置系统、自动化脚本、结果管理机制与文档模板，
**没有执行任何模型推理、benchmark 下载或 GPU 评测**。

---

## 1. 当前阶段（Framework Construction Only）

| 已完成 | 未执行 |
| --- | --- |
| ✅ 目录结构 | ❌ 模型推理 |
| ✅ 配置系统（yaml 驱动） | ❌ benchmark 数据下载 |
| ✅ 自动化脚本（参数解析/流程占位） | ❌ GPU 资源占用 |
| ✅ 结果管理机制（registry / 版本化目录） | ❌ 任何评测任务 |
| ✅ 文档模板（protocol / environment / benchmark） | ❌ 生成任何虚假测试结果 |

代码中**不硬编码**模型路径、benchmark 名称、batch size、dtype 或输出路径，
所有运行参数一律通过 `configs/*.yaml` 管理。

## 2. 评测目标

- **目标模型**：Qwen3.5-4B
- **当前版本**：`qwen35-4b-base-v0.0`（未微调基座，baseline）
- **未来版本**：`qwen35-4b-lora-v1.0`、`qwen35-4b-dpo-v2.0` 及其他实验版本

所有版本经由 `results/registry.yaml` 登记，使用**版本化结果目录**，互不覆盖，可横向比较。

## 3. 能力维度与 Benchmark

| 能力维度 | Benchmark | 指标 | 状态 |
| --- | --- | --- | --- |
| 知识与理解 | MMLU-Pro, C-Eval | Accuracy | ✅ 配置已就绪 |
| 专业推理 | GSM8K（预留 GPQA / MATH） | Exact Match / Accuracy | ✅ 配置已就绪 |
| 指令遵循 | IFEval | strict accuracy / loose accuracy | ✅ 配置已就绪 |
| 长上下文 | LongBench（预留 RULER） | task score | ✅ 独立接口 |
| 多语言 | CMMLU, MGSM | language accuracy | ✅ 配置已就绪 |

详细说明见 [docs/benchmark_description.md](docs/benchmark_description.md)。

## 4. 技术方案

- **主框架**：`lm-eval`（LLM 通用能力评测）
- **长上下文**：保留独立接口 `scripts/run_longbench.py`，不强行绑定主框架
- **模型加载**：支持 HuggingFace 模型 / 本地模型路径 / chat template / batch / dtype 配置

## 5. 目录结构

```
Benchmark/
├── README.md
├── requirements.txt
├── environment.txt
├── configs/
│   ├── baseline_qwen35_4b.yaml
│   └── template.yaml
├── scripts/
│   ├── install_env.sh
│   ├── run_eval.sh
│   ├── run_lm_eval.py
│   ├── run_longbench.py
│   └── generate_report.py
├── src/
│   ├── config_loader.py
│   ├── evaluator.py
│   ├── result_manager.py
│   ├── report_generator.py
│   └── utils.py
├── results/
│   └── registry.yaml
├── reports/
└── docs/
    ├── evaluation_protocol.md
    ├── benchmark_description.md
    └── environment_setup.md
```

## 6. 后续运行流程（服务器阶段）

### 6.1 一次性环境安装（基于 uv）

```bash
bash scripts/install_env.sh
```

### 6.2 填写模型配置

复制 `configs/template.yaml` 为实际配置，填写真实模型路径：

```bash
cp configs/template.yaml configs/baseline_qwen35_4b.yaml
# 编辑 configs/baseline_qwen35_4b.yaml，将 model.path 改为真实路径
```

### 6.3 运行 baseline 评测

```bash
bash scripts/run_eval.sh --config configs/baseline_qwen35_4b.yaml
```

### 6.4 生成本次运行报告

```bash
python scripts/generate_report.py --config configs/baseline_qwen35_4b.yaml
```

### 6.5 长上下文独立评测

```bash
python scripts/run_longbench.py --config configs/baseline_qwen35_4b.yaml
```

> **说明**：以上命令当前仅为流程占位，不会真正加载模型或运行评测。

## 7. 结果管理规则

- 每次运行写入 `results/<version>/` 版本化目录，**不覆盖**旧版本
- `results/registry.yaml` 登记所有模型版本的元信息
- 每个版本目录包含 `config.yaml`（本次运行配置快照）、`logs/`、`samples/`、`result.json`、`report.md`

详见 [docs/evaluation_protocol.md](docs/evaluation_protocol.md)。

## 8. 相关文档

| 文档 | 内容 |
| --- | --- |
| [docs/evaluation_protocol.md](docs/evaluation_protocol.md) | 评测目标、能力维度、版本管理规则、运行流程 |
| [docs/benchmark_description.md](docs/benchmark_description.md) | 各 benchmark 来源 / 能力 / 指标 |
| [docs/environment_setup.md](docs/environment_setup.md) | Python / CUDA 环境与依赖安装方式 |
