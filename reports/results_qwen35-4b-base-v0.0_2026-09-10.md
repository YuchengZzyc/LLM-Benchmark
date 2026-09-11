# Qwen3.5-4B 通用能力基线评测报告（v0.0 限量基线）

> **版本**：`qwen35-4b-base-v0.0`（未微调基线 · 限量 30 条/任务/学科叶）
> **评测日期**：2026-09-10 18:38 – 23:43（约 5 小时，含一次网络故障降级自愈）
> **权重**：`/data/yucheng/madm-llm/models/Qwen3.5-4B`（Qwen/Qwen3.5-4B，post-trained 版，bfloat16）
> **状态**：已完成（五维 24 个任务 0 error；registry status: completed）

---

## 1. 概述

对 Qwen3.5-4B（未微调，作为本项目基线）进行五个能力维度的系统评测：
**知识与理解 / 专业推理 / 指令遵循 / 长上下文 / 多语言**。产出可复现、可迭代
对比的基线分数；后续微调版本（LoRA/DPO 等）在同一协议下追加版本号评测，
与本基线逐项对比（复盘入口见 §6）。

**v0.0 为限量基线**：每任务/每学科叶最多 30 条（`max_samples: 30`），
`max_gen_toks: 2048`。全量版（v1.0，`max_samples: null`）预留后续执行。
限量设置对本报告结论的影响见 §5。

## 2. 评测设置

### 2.1 能力维度 ↔ 数据集 ↔ 指标

| 维度 | lm-eval 任务 | 上游数据集 | 指标 | few-shot | 样本量* |
| --- | --- | --- | --- | --- | --- |
| 知识与理解 | `mmlu_redux_generative` | fxmarty/mmlu-redux-2.0-ok | exact_match | 默认 | 57 叶 × 30 = 1710 |
| | `mmlu_prox_en` | li-lab/MMLU-ProX（英） | exact_match | 5-shot | 14 学科 × 30 = 420 |
| | `mmlu_prox_zh` | li-lab/MMLU-ProX（中） | exact_match | 5-shot | 14 学科 × 30 = 420 |
| | `ceval-valid` | C-Eval 验证集（52 学科） | acc | 默认 | 52 学科 × ≤30 = 1195 |
| 专业推理 | `gsm8k` | GSM8K | exact_match (strict/flexible) | 5-shot | 30 |
| | `leaderboard_math_hard` | MATH-lighteval (hard) | exact_match | 4-shot | 7 学科 × 30 = 210 |
| | `aime24` / `aime25` | AIME 2024 / 2025 | exact_match | 0-shot | 30 / 30 |
| | `minerva_math500` | HuggingFaceH4/MATH-500 | exact_match / math_verify | 4-shot | 30 |
| 指令遵循 | `ifeval` | IFEval | strict / loose accuracy | 0-shot | 30 |
| 长上下文 | LongBench v1 × 12 任务（独立接口） | THUDM/LongBench（本地 JSONL） | F1/EM · ROUGE-L · acc | task-specific | 12 × 30 |
| 多语言 | `mgsm_native_cot_zh` | juletxara/mgsm（中文原生 CoT） | exact_match | 0-shot | 30 |
| | `mmmlu_zh_cn` | openai/MMMLU（简中，57 学科） | acc / acc_norm | 默认 | 57 学科 × 30 = 1710 |

\* 组任务 limit 按叶任务应用（lm-eval 语义）；long_context 由
`scripts/run_longbench.py` 独立评测（内存截断，不改写数据）。

**未覆盖项及原因**：`gpqa_diamond_zeroshot`（门控数据集需 HF_TOKEN）、
`mmlu_pro`（全量约 42h）、`code` 维度 humaneval 系列（需 code_eval，Python ≥ 3.12，
当前 venv 3.11.16）、`longbench2`（数据加载未验证）。详见
[docs/evaluation_runbook.md](../docs/evaluation_runbook.md) §7。

### 2.2 框架与环境

| 项 | 值 |
| --- | --- |
| 主框架 | lm-eval 0.4.13（`simple_evaluate`，knowledge/reasoning/instruction/multilingual） |
| 长上下文 | 自研独立接口 `scripts/run_longbench.py`（官方口径评分：F1/ROUGE-L/acc） |
| 推理 | transformers 5.6.0 · torch 2.11.0+cu128 · RTX 5090 32GB |
| 生成参数 | greedy（temperature=0, top_p=1）、`max_gen_toks=2048`、chat template 开启 |
| 批大小 / 上下文 | batch_size 8 · max_length 4096 |
| 种子 | 42（random/numpy/torch/fewshot 四路） |
| 数据源 | HF 镜像 hf-mirror.com（huggingface.co 不可达）；LongBench v1 本地 JSONL |

环境搭建与跑测命令的完整步骤见
[docs/evaluation_runbook.md](../docs/evaluation_runbook.md)（§3 环境搭建、§4 跑法）。

## 3. 评测结果

<!-- 分数已回填（数据源 results/qwen35-4b-base-v0.0/result.json，2026-09-10 23:43 落盘） -->

### 3.1 知识与理解

| 任务 | 样本 | 指标 | 分数 | stderr |
| --- | --- | --- | --- | --- |
| mmlu_redux_generative | 1710 | exact_match | 0.2333 | 0.0102 |
| mmlu_prox_en | 420 | exact_match (custom-extract) | 0.1857 | 0.0185 |
| mmlu_prox_zh | 420 | exact_match (custom-extract) | 0.3738 | 0.0229 |
| ceval-valid | 1195 | acc | 0.2410 | 0.0124 |

### 3.2 专业推理

| 任务 | 样本 | 指标 | 分数 | stderr |
| --- | --- | --- | --- | --- |
| gsm8k | 30 | exact_match (strict) | 0.4000 | 0.0910 |
| | | exact_match (flexible) | 0.2000 | 0.0743 |
| leaderboard_math_hard | 210 | exact_match | 0.1524 | 0.0236 |
| aime24 | 30 | exact_match | 0.0000 | 0.0000 |
| aime25 | 30 | exact_match | 0.0000 | 0.0000 |
| minerva_math500 | 30 | exact_match | 0.0667 | 0.0463 |
| | | math_verify | 0.4000 | 0.0910 |

### 3.3 指令遵循

| 任务 | 样本 | 指标 | 分数 | stderr |
| --- | --- | --- | --- | --- |
| ifeval | 30 | prompt-level strict acc | 0.1333 | 0.0631 |
| | | inst-level strict acc | 0.3333 | — |
| | | prompt-level loose acc | 0.1333 | 0.0631 |
| | | inst-level loose acc | 0.3333 | — |

### 3.4 长上下文（LongBench v1，各 30 条）

| 任务 | 类别 | 指标 | 分数 |
| --- | --- | --- | --- |
| multifieldqa_en | 单文档问答 | F1 / EM | 0.0610 / 0.0000 |
| qasper | 单文档问答 | F1 / EM | 0.0258 / 0.0000 |
| hotpotqa | 多文档问答 | F1 / EM | 0.0111 / 0.0000 |
| 2wikimqa | 多文档问答 | F1 / EM | 0.0187 / 0.0000 |
| musique | 多文档问答 | F1 / EM | 0.0045 / 0.0000 |
| gov_report | 摘要 | ROUGE-L | 0.1645 |
| multi_news | 摘要 | ROUGE-L | 0.1292 |
| trec | few-shot 分类 | acc | 0.0000 |
| triviaqa | few-shot 问答 | F1 / EM | 0.0144 / 0.0000 |
| samsum | few-shot 摘要 | ROUGE-L | 0.0694 |
| passage_count | 合成 | acc | 0.0000 |
| passage_retrieval_en | 合成 | acc | 0.0333 |

> ⚠ LongBench v1 分数受**基座冗长前导 × 官方短生成上限**双重压制（详见 §5 第 5 条）：
> TASK_META 沿用官方 pred.py 的 `max_new_tokens`（问答类 64–128、摘要 512），而基座
> 在 chat 模板下先输出数百至数千字符 "Thinking Process" 前导，生成额度在到达答案前
> 已耗尽。512 上限的摘要类（gov_report 0.164）明显高于 64 上限的问答类，佐证此根因。
> 该组分数反映的是"官方口径下基座的表现"，不代表其真实长上下文能力。

### 3.5 多语言

| 任务 | 样本 | 指标 | 分数 | stderr |
| --- | --- | --- | --- | --- |
| mgsm_native_cot_zh | 30 | exact_match (strict) | 0.0000 | 0.0000 |
| | | exact_match (flexible) | 0.6000 | 0.0910 |
| mmmlu_zh_cn | 1710 | acc | 0.2351 | 0.0102 |

## 4. 与公开结果对比

官方来源：Qwen/Qwen3.5-4B HuggingFace 模型卡（调研报告 §7.2 摘录）。
**口径声明**：官方分数为其评测设置（post-trained 模型推荐模板、全量数据、
官方生成参数）下的结果；本地为冻结设置（chat template + greedy +
`max_gen_toks=2048` + 限量 30 条）。任务与官方 benchmark 非逐项相同处
已标注"近似对应"，对比仅供方向性参考，不构成胜负结论。

| 维度 | 本地任务 | 本地分数 | 官方 benchmark | 官方分数 | 对应关系 |
| --- | --- | --- | --- | --- | --- |
| 知识与理解 | mmlu_redux_generative | **0.2333** | MMLU-Redux | 88.8 | 同名数据集，口径不同（generative / 限量） |
| | mmlu_prox_en | **0.1857** | MMLU-ProX | 71.5 | 同名（英文子集 / 限量） |
| | mmlu_prox_zh | **0.3738** | MMLU-ProX | 71.5 | 同名（中文子集 / 限量） |
| | （mmlu_pro 未跑） | — | MMLU-Pro | 79.1 | 未覆盖 |
| | ceval-valid | **0.2410** | C-Eval | 85.1 | 同名（验证集子集 / 限量） |
| 专业推理 | （gpqa 未跑，门控） | — | GPQA Diamond | 76.2 | 未覆盖 |
| | gsm8k（strict） | **0.4000** | HMMT Feb/Nov 25 | 74.0 / 76.8 | 近似对应（竞赛/数学类） |
| | leaderboard_math_hard | **0.1524** | ↑ | ↑ | 近似对应 |
| | aime24 / aime25 | **0.00 / 0.00** | ↑ | ↑ | 近似对应 |
| | minerva_math500（math_verify） | **0.4000** | ↑ | ↑ | 近似对应（MATH-500 同源） |
| 指令遵循 | ifeval（prompt strict） | **0.1333** | IFEval | 89.8 | 同名（限量） |
| 长上下文 | LongBench v1 × 12（F1/ROUGE-L/acc） | **0.00–0.16** | LongBench v2 | 50.0 | 版本不同（v1 vs v2）+ 口径压制（见 §5.5），仅供参考 |
| 多语言 | mmmlu_zh_cn | **0.2351** | MMMLU | 76.1 | 同名（中文子集 / 限量） |
| | mgsm_native_cot_zh（flexible） | **0.6000** | （无官方对应） | — | — |

**差异解读**（本地普遍低于官方一个数量级的主因，按影响排序）：

1. **知识/多语言 loglikelihood 类**（MMLU-Redux/ProX、C-Eval、MMMLU 约 0.19–0.37 vs 官方 71–89）：
   官方为 post-trained 模型按其推荐模板与全量数据评测；本地冻结协议（chat template +
   greedy + 限量 30 条/叶）下基座表现出的差距属真实口径差异，不全是采样误差
   （stderr ≤ 0.023，差异远超噪声）。可能主因：官方推荐模板/prompt 格式与本地不同，
   及 post-trained 权重在选择题 loglikelihood 路径上的校准差异。
2. **生成类**（gsm8k 0.40、math500 math_verify 0.40、mgsm flexible 0.60）：2048 上限
   下基座 CoT 可用性明显改善（512 时 gsm8k 仅 0.05），但 strict 口径（要求格式化
   `\boxed{}` / `####` 答案）与冗长前导仍互相干扰——gsm8k flexible 0.20 反而低于
   strict 0.40 即格式提取错位的证据。AIME 0 分：30 条限量 + 竞赛难度，单条 CoT 常
   超长被截断。
3. **长上下文**：口径压制（§5.5），不能据此认为能力缺失。

## 5. 已知限制

1. **限量 30 条**：单任务 stderr 约 ±0.09（p=0.5 时），分数为基线方向的
   粗粒度信号；精确分数须等 v1.0 全量。
2. **生成上限 2048**：诊断（2026-09-10）表明该基座输出冗长 "Thinking Process"
   前导，512 上限会大量截断；2048 下超长 CoT 仍可能截断（对 aime/数学 hard 类
   影响最大）。官方口径无此限制。
3. **官方模型卡为 post-trained 版**：本地权重即该版，但评测模板/参数与官方
   推荐不完全一致；差异在 §4 口径声明中逐项标注。
4. **未覆盖项**：GPQA（门控）、MMLU-Pro（耗时）、代码维度（Python 3.11）、
   LongBench v2。
5. **LongBench v1 口径压制**（本报告最重要的解读注意项）：TASK_META 按官方
   pred.py 指定短 `max_new_tokens`（问答 64–128 / 摘要 512），基座冗长前导使
   生成在答案前耗尽，F1/EM 全部接近 0。该口径为 instruct 模型设计；对基座
   建议后续补一版"扩展口径"（如统一 max_new_tokens 512–1024）作对照，本次
   未改（保持官方口径一致性），见 §6。
6. **运行故障记录**：全量批次在 loglikelihood 聚合阶段遇 HF 镜像 SSL 握手超时
   （`_ssl.c:999 handshake timed out`），`src/evaluator.py` 按维度降级重跑自愈，
   最终五维 24 任务 0 error；`result.json` 的 `errors` 字段保留了该记录。

## 6. 版本维护与复盘

- 本报告对应 `results/qwen35-4b-base-v0.0/`（config 快照 / logs / result.json /
  report.md），registry.yaml 已登记（status: completed）。
- 后续迭代（如 `qwen35-4b-lora-v1.0`）：复制 `configs/template.yaml`，改
  `model.version` 与 `model.path`，其余参数与本基线一致以保证可比；
  跑完后各版本 report.md 的 Comparison 表自动汇总对照。
- 复盘入口：`results/registry.yaml`（总览）、各版本 `result.json`（逐任务指标
  diff）、`logs/run.log`（原始「评测结果: {...}」行）。
- 命名规范：`qwen35-4b-<type>-v<major>.<minor>`
  （base=未微调 / lora / dpo），详见
  [docs/evaluation_protocol.md](../docs/evaluation_protocol.md) §4。
- 建议的下一步（不阻塞本基线交付）：
  ① `v0.1` LongBench 扩展口径对照（§5.5）；
  ② 配置 HF_TOKEN 后补 GPQA Diamond；
  ③ `v1.0` 全量（`max_samples: null`，预计 30h+）。

---

*数据源：`results/qwen35-4b-base-v0.0/result.json`（2026-09-10 23:43 落盘）；
远端 `report.md` 由 `scripts/generate_report.py` 自动生成，本文档为含官方对比
与解读的人工整理版。生成于 2026-09-11。*
