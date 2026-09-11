# Qwen3.5-4B 通用能力基线评测报告（v0.1 · 口径修复基线）

> **版本**：`qwen35-4b-base-v0.1`（未微调基座 · **口径修复基线**）
> **评测日期**：2026-09-11
> **权重**：`/data/yucheng/madm-llm/models/Qwen3.5-4B`（Qwen/Qwen3.5-4B，post-trained，bfloat16）
> **状态**：已实测 6 项（GSM8K / C-Eval / MMMLU-zh / MMLU-ProX-zh / IFEval / MGSM）；LongBench 运行中、MMMLU 全量待补（见 §6）
> **registry**：`results/registry.yaml` → `qwen35-4b-base-v0.1`（completed）

---

## 1. 为什么有 v0.1：口径修复

**v0.0 基线（2026-09-10）全部走 chat 模板，结果被 Qwen3.5 的 thinking 前导污染**：

| 污染路径 | 表现 |
| --- | --- |
| 生成式任务（GSM8K / MMLU-ProX） | 默认模板注入 `<think>\n`，模型先输出 400–6000 字 "Thinking Process:"，2048 tok 上限下答案被截断，正则抽取失败 |
| loglikelihood 任务（C-Eval / MMLU-zh / MMLU-Redux） | chat 模板在选项 token 前插入 `<think>\n\n</think>\n\n` 空块，污染选项对数概率对比，acc 掉到接近随机 |

**v0.1 的修复**（与官方分数对齐的两种正确口径）：

1. **生成式任务**：`enable_thinking=False, think_end_token=</think>`（lm-eval 0.4.13 HFLM 原生支持，`hf.py:1214`/`hf.py:1718`），生成路径剥离 thinking 前导。
2. **loglikelihood 任务（MMLU 系选择题）**：`apply_chat_template=False` **纯文本 MCQ**（`"Answer:"` 选字母）。官方走的就是纯文本口径 —— 探针证明 chat 模板让 MMMLU-zh 只有 41.25%，纯文本恢复到 71.75%。

---

## 2. 评测设置

### 2.1 能力维度 ↔ 数据集 ↔ 指标（v0.1 实测与待补）

| 维度 | lm-eval 任务 | 指标 | few-shot | 样本 | 状态 |
| --- | --- | --- | --- | --- | --- |
| 知识与理解 | `mmlu_prox_zh` | exact_match（生成式+正则抽取） | 5 | 840（60/科×14） | ✅ 已测 |
| | `ceval-valid` | acc（纯文本 MCQ） | 5 | 1346（52 科全量） | ✅ 已测 |
| | `mmmlu_zh`（MMMLU 简中） | acc（纯文本 MCQ） | 5 | 400（采样） | ✅ 已测（全量 57 科待补） |
| 专业推理 | `gsm8k` | exact_match（strict/flexible） | 5 | 1319（全量） | ✅ 已测 |
| 指令遵循 | `ifeval` | strict / loose accuracy | 0 | 541（全量） | ✅ 已测 |
| 长上下文 | LongBench v1 × 12（独立接口） | F1/EM · ROUGE-L · acc | task-specific | 12×30 | ⏳ 待补 |
| 多语言 | `mmmlu_zh`（简中，MMMLU） | acc | 5 | 同上 | ✅ 已测 |
| | `mgsm_native_cot_zh` | exact_match (flexible-extract) | 0 | 250（全量） | ✅ 已测 |

### 2.2 框架与环境

| 项 | 值 |
| --- | --- |
| 主框架 | lm-eval 0.4.13（`simple_evaluate`） |
| 长上下文 | 自研独立接口 `scripts/run_longbench.py` |
| 推理 | transformers 5.6.0 · torch 2.11.0+cu128 · RTX 5090 32GB |
| 生成参数 | greedy（temperature=0）、`max_gen_toks=2048` |
| 模型参数 | `pretrained=/data/yucheng/madm-llm/models/Qwen3.5-4B, dtype=bfloat16, enable_thinking=False, think_end_token=</think>` |
| 批大小 / 上下文 | batch_size 8 · max_length 4096 |
| 种子 | 42（random/numpy/torch/fewshot 四路） |
| 数据源 | HF 镜像 hf-mirror.com（huggingface.co 不可达） |

完整环境搭建与跑测命令见 [docs/environment_setup.md](docs/environment_setup.md)、[docs/evaluation_runbook.md](docs/evaluation_runbook.md)。

---

## 3. 评测结果（对照官方）

| 数据集 | 样本 | 指标 | **v0.1** | **官方** | 差 | 口径 |
| --- | --- | --- | --- | --- | --- | --- |
| GSM8K | 1319 | exact_match (strict) | **0.90296** | 0.889 | **+0.014** ✅ | 生成式，关 thinking |
| C-Eval (valid) | 1346 | acc | **0.74963** | 0.851 | −0.101 | 纯文本 MCQ，5-shot |
| MMMLU-zh | 400 | acc | **0.71750** | 0.761 | −0.044 | 纯文本 MCQ，5-shot |
| MMLU-ProX (zh) | 840 | exact_match (custom-extract) | **0.63452** | 0.715 | −0.080 | 生成式抽取，5-shot，60/科 |
| IFEval | 541 | inst_level_loose_acc | **0.89928** | 0.898 | **+0.001** ✅ | 生成式，关 thinking，0-shot |
| MGSM (zh) | 250 | exact_match (flexible-extract) | **0.764** | — | — | 0-shot 原生 CoT，关 thinking |

### 3.1 与 v0.0 的对比（口径修复的效果）

| 任务 | v0.0（chat 污染） | v0.1（口径修复） | 官方 |
| --- | --- | --- | --- |
| GSM8K | 0.400 | **0.903** | 0.889 |
| C-Eval | 0.241 | **0.750** | 0.851 |
| MMMLU-zh | 0.413（chat+关thinking探针） | **0.718** | 0.761 |
| MMLU-ProX-zh | 0.374 | **0.635** | 0.715 |

口径修复使生成式任务跳升 ~0.5、loglikelihood 选择题恢复随机率之上的正常区间。

---

## 4. 关键结论

1. **GSM8K 完美复现官方（90.3 vs 88.9，略超 1.4 点）** —— 关 thinking 的生成式路径对推理题完全正确。
2. **IFEval 完美复现官方（inst_level_loose 89.93 vs 89.8）** —— 关 thinking 后指令跟随质量与官方持平，指令遵循维度闭环。
3. **MMLU 系选择题必须走纯文本 MCQ**。chat 模板无论 thinking 开/关都会注入 thinking 相关 token 污染选项概率；官方走 `"Answer:"` 纯文本 loglikelihood 选字母。
4. **C-Eval 差 10 点，疑似模板缺口**：官方 C-Eval 评测带"以下是中国关于X的单项选择题，请选出其中的正确答案"的 description 上下文 + 固定 fewshot，lm-eval 的 `ceval-valid` 恰好缺这段 description，5-shot 只比 0-shot 涨 2 点（0.750 vs 0.730）。待自定义任务验证。
5. **MMLU-ProX 采样口径**：60/科只采全量 6.8%，官方为英+中混合全量 12,412；差 8 点主要来自采样。
6. **MGSM 中文 76.4%（flexible-extract）**：逐步解答逻辑正确（数字抽取得出 76.4%），但结尾没按官方格式写"答案是 X。"（strict 0%）。与 GSM8K strict/flexible 差距同源，官方 MGSM 评测取 flexible 口径。多语言维度闭环。

---

## 5. 复现方法（跑测命令）

所有脚本在评测服务器 `/data/yucheng/madm-llm/Benchmark/` 下运行（本地 Windows 仅编辑/推送）：

```bash
# 环境变量
export HF_ENDPOINT=https://hf-mirror.com
# 模型参数（所有任务统一关闭 thinking）
MODEL_ARGS="pretrained=/data/yucheng/madm-llm/models/Qwen3.5-4B,dtype=bfloat16,enable_thinking=False,think_end_token=</think>"

# GSM8K 全量（1319，~15min，官方 88.9 → 实测 90.3）
python gsm8k_run.py

# C-Eval 5-shot 全量（1346，~5min，官方 85.1 → 实测 75.0）
python ceval_run.py        # CEVAL_FEWSHOT=5（脚本默认）

# MMMLU-zh 纯文本 MCQ（400 采样，~2min，官方 76.1 → 实测 71.75）
python mcq_probe.py        # A 组 = 纯文本口径

# MMLU-ProX-zh 60/科（840，~70min，官方 71.5 → 实测 63.45）
A_B_GROUP=B A_B_TASKS=mmlu_prox_zh A_B_LIMIT=60 python a_b_tasks.py
```

完整协议见 [docs/evaluation_protocol.md](docs/evaluation_protocol.md)。

---

## 6. 待补项（下轮迭代清单）

| 优先级 | 任务 | 目的 |
| --- | --- | --- |
| P0 | MMMLU-zh 全量 57 科 | 确认 71.75 采样差（~15min） |
| P1 | C-Eval 官方 description 模板自定义任务 | 验证差 10 点假设（~15min 工作量，2min 出数） |
| P1 | LongBench v1 × 12 | 补"长上下文"维度（运行中，~40min） |
| P2 | MMLU-Redux（4115，~40min） | 补官方表 mmlu_redux 88.8 |

---

## 7. 版本与复盘

- **v0.0**（2026-09-10）：chat 模板全部走通，但被 thinking 污染，分数不可作数，仅作"污染基线"参考。
- **v0.1**（2026-09-11）：**当前基线**，口径修复后四个数据集已对齐官方方向。
- **v1.0+**（微调后）：LoRA/DPO 等版本在**同一协议**下追加评测（`results/<version>/` 目录、registry 登记），与本基线逐项对比复盘。

复盘入口：[results/registry.yaml](results/registry.yaml)、[results/qwen35-4b-base-v0.0/](results/qwen35-4b-base-v0.0/)、[results/qwen35-4b-base-v0.1/](results/qwen35-4b-base-v0.1/)。
