# Benchmark 说明（Benchmark Description）

> 当前阶段：配置就绪，数据/评测将在服务器阶段执行。

本页列出框架支持的 benchmark 来源、所测能力与指标定义。所有名称均为
`lm-eval`（及 LongBench 独立接口）中的任务名/数据集名。

> **协议差异声明（重要）**：模型卡中部分 benchmark（HMMT / LiveCodeBench /
> OJBench / IFBench）在 lm-eval 0.4.13 中不存在，本框架以协议最接近
> 的公开 benchmark 补齐（见各节「用途」标注）。**公开跑分不是本地验收**——
> 协议（few-shot、数据版本、评测框架、prompt 模板）不一致时，本地分数与
> 模型卡分数**不可直接做胜负结论**，仅能反映同协议下的趋势。
>
> **长上下文例外**：AA-LCR（§4.0）已定为长上下文维度**最终评测**——独立接口
> 原生复现官方协议（全量 100 题 + 官方 prompt 模板 + gpt-5.6-luna 判题器），
> 目标是逐位复现模型卡 57.0，**非近似替代**。
>
- **正式评测的冻结维度 / 验收基线 / 运行命令一律以 [evaluation_spec.md](evaluation_spec.md) 为准。**

---

## 1. 知识与理解（Accuracy）

### 1.1 MMLU-Pro

- **来源**：MMLU-Pro（Massive Multitask Language Understanding - Pro），
  在经典 MMLU 基础上扩展更难的选项与题目数量（官方数据集 `TIGER-Lab/MMLU-Pro`）。
- **所测能力**：跨学科世界知识与理解（涵盖 STEM、人文、社会科学等领域）。
- **指标**：Accuracy（0-100）。对每个题目在候选选项中选对的比例。

### 1.2 C-Eval

- **来源**：C-Eval（Chinese Evaluation Suite），面向中文的综合知识评测套件。
- **所测能力**：中文语境下的多领域知识与理解（涵盖计算机、数学、医学、
  法律、历史、文学等多个学科）。
- **指标**：Accuracy（0-100）。
- **lm-eval 任务名**：`ceval-valid`（0.4.13 无 `c_eval`，验证集 52 学科）。

### 1.3 MMLU-Redux

- **来源**：MMLU-Redux（`fxmarty/mmlu-redux-2.0-ok`），修复争议题目后的 MMLU。
- **指标**：Accuracy。lm-eval 0.4.13 中为 `mmlu_redux_generative`（generative
  格式，58 叶任务）。

### 1.4 MMLU-ProX

- **来源**：MMLU-ProX（`li-lab/MMLU-ProX`），带错误选项扰动与更严谨题目的
  MMLU-Pro 变体。
- **指标**：Accuracy（5-shot）。lm-eval 任务名 `mmlu_prox_en`（英文，14 学科）
  与 `mmlu_prox_zh`（中文，14 学科）。

### 1.5 MMMLU（中文子集）

- **来源**：MMMLU（`openai/MMMLU`），多语言 MMLU 扩展。
- **指标**：Accuracy。lm-eval 任务名 `mmmlu_zh_cn`（简体中文，57 学科）。

## 2. 专业推理（Exact Match / Accuracy）

### 2.1 GSM8K

- **来源**：GSM8K（Grade School Math 8K），小学数学应用题数据集（OpenAI）。
- **所测能力**：多步数学推理。
- **指标**：Exact Match / Accuracy（答案与标准答案字符串完全匹配的比例）。

### 2.2 GPQA（已接入）

- **来源**：GPQA（Graduate-Level Google-Proof Q&A），研究生级科学问答
  （`Idavidrein/gpqa`）。
- **所测能力**：高难度科学推理（生物、化学、物理）。
- **指标**：Accuracy（0-100）。
- **lm-eval 任务名**：`gpqa_diamond_zeroshot`（Diamond 子集，0-shot，
  metrics: acc / acc_norm）。

### 2.3 MATH-Hard / AIME（HMMT 近似替代）

- **来源**：MATH（Hendrycks et al.），竞赛级数学问题集。
- **lm-eval 任务名**：
  - `leaderboard_math_hard`（`DigitalLearningGmbH/MATH-lighteval`，8 学科，4-shot）
  - `aime24` / `aime25`（AIME 2024/2025，0-shot，tag: math_word_problems）
- **用途**：模型卡 HMMT 在 lm-eval 0.4.13 中无对应任务，以 MATH-Hard + AIME
  作近似替代。协议不同，**不做直接胜负结论**。

### 2.4 MATH-500

- **来源**：MATH-500（`HuggingFaceH4/MATH-500`，MATH 精选 500 题）。
- **lm-eval 任务名**：`minerva_math500`（4-shot）。

### 2.5 HumanEval / HumanEval+ / MBPP+（LiveCodeBench 近似替代）

- **来源**：OpenAI HumanEval / EvalPlus 的 HumanEval+ / MBPP+。
- **lm-eval 任务名**：`humaneval`（0-shot，pass@1）、`humaneval_plus`（0-shot）、
  `mbpp_plus`（3-shot）。
- **用途**：模型卡 LiveCodeBench 在 lm-eval 0.4.13 中无对应任务，以
  HumanEval 系列近似替代。注意 humaneval 需安装 code_eval 依赖（见
  install_env.sh）才能真实执行单元测试。
- **协议差异**：HumanEval 与 LiveCodeBench 数据不重叠、评测框架不同，
  **不做直接胜负结论**。

## 3. 指令遵循（strict / loose accuracy）

### 3.1 IFEval

- **来源**：IFEval（Instruction-Following Eval），构造性指令遵循评测集。
- **所测能力**：模型对显式指令约束（格式、长度、关键词、引用等）的遵循程度。
- **指标**：
  - **strict accuracy**：所有可检查约束均满足才算通过的比例。
  - **loose accuracy**：仅要求关键约束满足（更宽松）的比例。

## 4. 长上下文（task score）

### 4.0 AA-LCR（最终长上下文评测 · 已冻结）

- **来源**：AA-LCR（`ArtificialAnalysis/AA-LCR`，Apache-2.0，v1.1），
  真实长上下文检索/理解基准。
- **官方参考（模型卡 Qwen3.5-4B）**：**57.0**（100 题 accuracy）。
- **为什么替代 LongBench v1/v2**：LongBench v1/v2 在本模型上跑不通 /
  不可比（v1 全部卡在复读、v2 判题器不可用），AA-LCR 官方跑分可逐位复现，
  作长上下文维度的**最终评测**。前置决策见 [aa_lcr_plan.md](aa_lcr_plan.md)。
- **数据集**：`ArtificialAnalysis/AA-LCR`（v1.1），100 题全 text-only；
  每套文档平均 ~99k tokens（`input_tokens` 单题可见），30 套 / 234 文档 /
  ~2.98M tokens。
- **文档顺序**：严格按 CSV `data_source_filenames`（分号分隔）加载，
  不得乱序；token 计数用 `tiktoken` 的 `cl100k_base`（官方统计口径）。
- **所测能力**：在 ~94k 上下文内的检索 + 多文档推理（Company / Industry /
  Gov Consultation / Academia / Legal / Marketing / Survey 七类文档）。
- **指标**：accuracy（CORRECT 数 / 100），**LLM 判题器 `gpt-5.6-luna`
  按官方 v1.1 prompt 判题**（含数字等价、格式、排序等判定规则）。
- **运行方式**：独立接口 `scripts/aa_lcr_run.py`（vLLM 并发 + 2 worker
  并行，约 5-6h），**不绑定 lm-eval**。
- **数据（已部署在服务器）**：
  `AA-LCR_Dataset.csv`（145KB，100 行）+ `AA-LCR_extracted-text.zip`
  → 解压 `aa_lcr_data/lcr/{category}/{set_id}/{filename}.txt`（230 个）。
- **复现条件（官方条件全量，不改）**：100 题全量 + 每套文档全量上下文 +
  `gpt-5.6-luna` 判题。只并行化生成部分压时间。

### 4.1 LongBench v1（归档，不再用于正式评测）

- **来源**：LongBench（THUDM/LongBench），长文本理解评测基准。
- **状态**：**已取消**。v0.1 实测 11/12 任务卡在无限复读、分数不可复现，
  v1 官方跑分无法作为本模型的长上下文验收。细节见 [aa_lcr_plan.md](aa_lcr_plan.md) §1。
- 独立接口 `scripts/run_longbench.py` 保留仅供**排障/研究**（非正式评测）；
  不进入冻结评测集合。

### 4.2 RULER（预留）

- **来源**：RULER，可扩展长上下文基准（合成任务，控制上下文长度）。
- **所测能力**：在特定上下文长度下的检索/聚合/多跳推理。
- **指标**：task score。
- **状态**：预留，尚未纳入默认配置。

## 5. 多语言（language accuracy）

### 5.1 MGSM（中文，原生 CoT）

- **来源**：MGSM（Multilingual Grade School Math），GSM8K 的多语言版本
  （`juletxara/mgsm`）。
- **lm-eval 任务名**：`mgsm_native_cot_zh`（中文原生 CoT，0.4.13 无统一
  `mgsm` 根任务）。
- **所测能力**：跨语言数学推理。
- **指标**：language accuracy（exact_match）。

### 5.2 MMMLU（中文子集）

- **来源**：MMMLU（`openai/MMMLU`），多语言 MMLU。
- **lm-eval 任务名**：`mmmlu_zh_cn`（57 学科，acc / acc_norm）。
