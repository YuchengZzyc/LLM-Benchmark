# AA-LCR 长上下文评测方案（Qwen3.5-4B v0.1）

> **状态**：定稿（2026-09-11）。长上下文维度的最终方案。
> **前置决策**：LongBench v1/v2 尝试后取消（太慢/不可复现）；AA-LCR 为官方卡 57.0 的复现目标。
> **执行模式**：单卡并发（vLLM）+ 2 进程并行。
> **目标**：官方条件全量 100 题、每题 ~94k 上下文、判题器 `gpt-5.6-luna`，**~5-6h 跑完，可复现官方 57.0**。

---

## 1. 为什么选 AA-LCR

| 项 | 值 |
| --- | --- |
| 官方分数（模型卡 Qwen3.5-4B 列） | **57.0** |
| 数据集 | `ArtificialAnalysis/AA-LCR`（Apache-2.0，v1.1） |
| 题数 | 100 题（全部 text-only，无视觉） |
| 每套文档 | 平均 ~99k tokens（cl100k_base），单题 `input_tokens` 可见（如 94494） |
| 文档类型 | 7 类：Company/Industry/Gov Consultation/Academia/Legal/Marketing/Survey |
| 文档集 | 30 套、234 个文档、共 ~2.98M tokens |
| 评分 | LLM 判题器 `gpt-5.6-luna`（官方） |
| 对比 | 非 LongBench v1/v2（无判题器依赖、上下文短但跑不通/不可比） |

**为什么可复现**：官方条件 = 100 题全量 + 每套文档全量上下文 + gpt-5.6-luna 判题。我们不改这三个条件，只把**生成部分并行化**压时间。

---

## 2. 官方评测协议（逐条还原）

### 2.1 Prompt 模板（官方 README 原样）
```python
documents_text = "\n\n".join(
    f"BEGIN DOCUMENT {i + 1}:\n{doc}\nEND DOCUMENT {i + 1}"
    for i, doc in enumerate(docs)
)
prompt = f"""BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION
"""
```
- **文档顺序**：严格按 `data_source_filenames`（CSV 分号分隔的有序列表）加载，不能乱序。
- **token 计数**：用 `tiktoken` 的 `cl100k_base`（官方统计口径）。

### 2.2 数据
| 文件 | 路径 | 状态 |
| --- | --- | --- |
| `AA-LCR_Dataset.csv` | `/data/yucheng/AA-LCR_Dataset.csv` | ✅ 已下载（145KB，100 行） |
| `extracted_text/AA-LCR_extracted-text.zip` | `/data/yucheng/AA-LCR_extracted-text.zip` | ✅ 已下载 |
| 解压后文档 | `/data/yucheng/aa_lcr_data/lcr/{category}/{set_id}/{filename}.txt` | ✅ 已解压（230 个 .txt） |

CSV 列：`(unnamed)`, `document_category`, `document_set_id`, `question_id`, `question`, `answer`, `data_source_filenames`（分号分隔）, `data_source_urls`, `input_tokens`。

### 2.3 判题器（官方 v1.1 协议）
**模型**：`gpt-5.6-luna`（base URL `http://192.168.13.165:8002`，key 已配置）。

**System prompt**（官方原样）：
```
Decide whether the CANDIDATE ANSWER is correct or incorrect against the OFFICIAL ANSWER.
Note the following points when assessing correctness:

- Numbers should still match when they are the same value written differently, e.g., a
  percentage, a count of percentage points, and the equivalent decimal fraction are the same
  value: 0.675, "67.5%" and "67.5 percentage points" all match. So do different scales
  (thousand, million, bn) and different notations (thousands separators, currency symbols,
  LaTeX markup, and numbers written as words).
- Where the question asks for a particular format (e.g., a percentage, a number of decimal
  places, a unit, a rounding, or an ordering) the CANDIDATE ANSWER must meet it. If the
  question asks for no particular format, accept any equivalent form.
- In cases where the question asks for an ordered list, a title, honorific or article added
  to an entry in the CANDIDATE ANSWER can change where that entry sorts. Accept the ordering
  if it is correct either with those additions or without them.
- Grade the value the CANDIDATE ANSWER finally commits to, and it must commit to one. Values
  reached while working, and alternatives it considers and sets aside, do not count. If it
  offers several values without selecting one, it is incorrect even if one of them is right.
  Hedging is fine as long as one clearly definitive answer is given.
```

**User prompt**（官方原样）：
```
Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: START QUESTION {question}

END QUESTION

The OFFICIAL ANSWER: {official_answer}

END OFFICIAL ANSWER

BEGIN CANDIDATE ANSWER TO ASSESS

{candidate_answer}

END CANDIDATE ANSWER TO ASSESS

Reply as JSON, with a verdict of CORRECT or INCORRECT.
```
- 解析输出为 JSON `{"verdict": "CORRECT"|"INCORRECT"}`，失败重试 1 次。
- 统计：`accuracy = CORRECT 数 / 100`。

---

## 3. 执行方案：单卡并发 + 2 进程并行

### 3.1 环境
| 项 | 值 |
| --- | --- |
| GPU | 1 × RTX 5090 32GB（已确认单卡） |
| 模型 | `/data/yucheng/madm-llm/models/Qwen3.5-4B`（bf16，本地） |
| 上下文 | 原生 262k（模型卡确认），100k 单题可装 |
| 引擎 | vLLM（单卡并发吞吐最高，需安装） |
| 并行 | 2 个 worker 进程，各处理 50 题（0-49 / 50-99） |
| 判题 | 2 个 worker 共享判题 API（gpt-5.6-luna） |
| 运行时 | `HF_ENDPOINT=https://hf-mirror.com`（数据已本地化，不依赖网络） |

### 3.2 时间预估
| 步骤 | 量 | 预估 |
| --- | --- | --- |
| 数据加载 + 文档构建 prompt | 100 题 | ~5 分钟 |
| 单 worker 生成 50 题 × ~94k 上下文（vLLM 并发） | 2 worker | **~5-6h** |
| 判题 100 次（gpt-5.6-luna，max_tokens 少量） | 100 | ~5-10 分钟 |
| 聚合 + 写结果 | — | 即时 |
| **合计** | | **~5-6h** |

### 3.3 结果落盘（版本化，5 维统一）
- 写入 `results/qwen35-4b-base-v0.1/result.json` 的 `metrics.dimensions.long_context`（AA-LCR 单独键），**不动**四个已跑通维度的结果。
- registry/report 同步更新。长上下文维度闭环后，**5 个维度全部有评测数据**。

---

## 4. 脚本设计（`scripts/aa_lcr_run.py`）

```
加载配置（configs/aa_lcr_qwen35_4b_v01.yaml）
├── 读 CSV（100 题）+ 文档（按 data_source_filenames 顺序）
├── 建 prompt（官方模板）+ 按 input_tokens 截断到 90k（保险）
├── 2 worker 并行：worker_i 跑 50 题
│   ├── 生成：vLLM 并发（每 worker 内 sequential over 50 题）
│   └── 判题：gpt-5.6-luna（官方模板）
├── 聚合 accuracy + 每题保存（含 verdict/answer/官方答案）
└── 合并写 result.json（_merge_metrics，不动其他维度）
```

**失败兜底**：worker 崩溃不丢已跑题（每题落盘 samples）；重跑续跑跳过已完成题。

---

## 5. 复现方法（跑测命令）

```bash
# 1) 安装 vLLM（服务器）
export HF_ENDPOINT=https://hf-mirror.com
pip install vllm

# 2) 启动评测
cd /data/yucheng/madm-llm/Benchmark
nohup ./scripts/aa_lcr_run.py --config configs/aa_lcr_qwen35_4b_v01.yaml \
    > /data/yucheng/aa_lcr_out.txt 2>&1 &

# 3) 监控
tail -f /data/yucheng/aa_lcr_out.txt
```

---

## 6. 风险与对策

| 风险 | 影响 | 对策 |
| --- | --- | --- |
| 32GB 内存 2 进程 vLLM OOM | 中途崩溃 | vLLM 并发受 `--max-concurrent`/KV 上限控制，先测 1 题验证 |
| 判题 API 超时/限流 | 某题判不了 | 重试 3 次 + 指数退避；失败标记，不阻塞整体 |
| 生成超时/无输出 | 某题空答 | 判题器判空为 INCORRECT（官方行为），记录 |
| 官方 57.0 偏差 | 分数不可比 | 先跑 5 题冒烟验证判题器+生成路径，再全量 |

---

## 7. 完成后固化为标准

> 长上下文维度测完后，本方案 + `scripts/aa_lcr_run.py` + 配置即为**标准长上下文评测流水线**，与其他 4 维（GSM8K / MMLU-ProX-zh / C-Eval / MMMLU-zh / IFEval / MGSM）并列，五维全部有评测数据。
> 后续版本（微调后）在同一协议下追加评测，与本基线 v0.1 逐项对比。
