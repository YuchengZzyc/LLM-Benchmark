# AA-LCR 长上下文评测方案（Qwen3.5-4B v0.1）

> **状态**：定稿（2026-09-11）。长上下文维度的最终方案。
> **前置决策**：LongBench v1/v2 尝试后取消（太慢/不可复现）；AA-LCR 为官方卡 57.0 的复现目标。
> **执行模式**：单进程串行（2026-09-11 用户裁定"会OOM那就改成单进程"；冒烟实测峰值显存 25.6GB，2 进程会超 31.36GB 可用，故单进程）。
> **目标**：官方条件全量 100 题、每题 ~94k 上下文、判题器 `gpt-5.6-luna`，**~30 分钟生成 + ~10 分钟判题**，可复现官方 57.0。

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

## 3. 执行方案：单进程串行（HF generate + fla 内核）

> 原"单卡并发 vLLM + 2 进程并行"设计作废：冒烟实测 HF 生成路径（fla 0.3.0
> Triton 内核）单进程峰值显存 **25.6GB**，2 进程需 ~51GB 远超 31.36GB 可用，
> 用户裁定 **单进程**。单进程也远快于预算：单题 10-24s → 100 题 ~25-30 分钟。

### 3.1 环境
| 项 | 值 |
| --- | --- |
| GPU | 1 × RTX 5090 32GB（已确认单卡，可用 31.36GB） |
| 模型 | `/data/yucheng/madm-llm/models/Qwen3.5-4B`（bf16，本地） |
| 上下文 | 原生 262k，单题 ~94k 全量加载，不截断 |
| 引擎 | transformers generate（fla 0.3.0 Triton 内核，`chunk_gated_delta_rule`） |
| 并行 | 单进程，100 题串行（每题增量落盘，--resume 续跑） |
| 判题 | gpt-5.6-luna API（官方协议，key 走 env `JUDGE_API_KEY`；endpoint `http://192.168.13.165:8002/v1/chat/completions`，注意 `/v1/` 前缀——裸 `/chat/completions` 返回的是前端 HTML） |
| 运行时 | `HF_ENDPOINT=https://hf-mirror.com`（数据已本地化，不依赖网络） |

### 3.2 时间预估（冒烟实测 2026-09-11）
| 步骤 | 量 | 预估 |
| --- | --- | --- |
| 模型加载（fla 内核首次 Triton JIT） | 1 次 | ~1-2 分钟 |
| 生成 100 题 × ~94k 上下文（单进程） | 100 | **~25-30 分钟**（单题 10-24s） |
| 判题 100 次（gpt-5.6-luna，max_tokens 16） | 100 | ~5-10 分钟 |
| 聚合 + 写结果 | — | 即时 |
| **合计** | | **~35-40 分钟**（远超预算内） |

> **run3 全量实测（2026-09-11，最终口径）**：开启 thinking（官方口径）后单题
> 显著慢于冒烟估值——生成 100 题实际 **9484s（~2h38m，单题 ~95s）**。
> 上表"25-30 分钟"为关 thinking 冒烟的乐观估值，仅作下限参考。
> run 演进：run1（1024 截断，16 题腰斩）→ run2（关 thinking + 1024，32.7%）→
> **run3（max_new_tokens=4096 + enable_thinking=True + NFC 文件名归一化）= 59.0**。

### 3.3 结果落盘（版本化，5 维统一）
- 写入 `results/qwen35-4b-base-v0.1/result.json` 的 `metrics.dimensions.long_context`（AA-LCR 单独键），**不动**四个已跑通维度的结果。
- registry/report 同步更新。长上下文维度闭环后，**5 个维度全部有评测数据**。

---

## 4. 脚本设计（`scripts/aa_lcr_run.py`）

```
加载配置（configs/aa_lcr_qwen35_4b_v01.yaml）
├── 读 CSV（100 题）+ 文档（按 data_source_filenames 顺序）
├── 建 prompt（官方模板，全量上下文不截断）
├── 单进程串行 100 题
│   ├── 生成：HF generate + chat 模板（enable_thinking=True（run3 起官方口径；
│   │   run2 关 thinking + 1024 截断仅 32.7%）,
│   │   stop_strings=["<|im_end|>"]，bf16，fla 内核）
│   └── 判题：gpt-5.6-luna（官方 system/user 模板，JSON verdict）
├── 每题增量落盘 samples/aa_lcr.json（崩溃最多丢一题；--resume 续跑）
├── 聚合 accuracy = CORRECT / 100
└── 合并写 result.json（_merge_metrics，metrics.dimensions.long_context.aa_lcr，
    不动其他维度）
```

---

## 5. 复现方法（跑测命令）

```bash
# 1) 环境（已就位：fla 0.3.0、causal-conv1d 未装不阻塞、uv venv）
#    venv: /data/yucheng/madm-llm/.venv  ·  uv: /data/yucheng/.local/bin/uv
#    判题 key 走环境变量，不写入任何文件

# 2) 冒烟（5 题，验证生成+判题链路）
export JUDGE_API_KEY='sk-...'
cd /data/yucheng/madm-llm/Benchmark
MSYS_NO_PATHCONV=1
./.venv/bin/python scripts/aa_lcr_run.py \
    --config configs/aa_lcr_qwen35_4b_v01.yaml --limit 5 --no-judge

# 3) 全量 100 题（单进程串行，后台运行）
nohup env JUDGE_API_KEY="$JUDGE_API_KEY" \
    ./.venv/bin/python scripts/aa_lcr_run.py \
    --config configs/aa_lcr_qwen35_4b_v01.yaml \
    > /data/yucheng/aa_lcr_run.log 2>&1 &

# 4) 监控
tail -f /data/yucheng/aa_lcr_run.log
```

> 判题 key（sk-h2h87…PyM）属敏感信息：只经环境变量传递，**永不写入**脚本、配置、日志、报告或 result.json。

---

## 6. 风险与对策

| 风险 | 影响 | 对策 |
| --- | --- | --- |
| 单进程显存超 31.36GB | OOM 崩溃 | 冒烟实测峰值 25.6GB，余量 ~6GB；仍以 `--limit 5` 复验 |
| 判题 API 超时/限流 | 某题判不了 | 每题最多 2 次重试（指数退避 2s/8s）；仍失败记 error 不阻塞 |
| 生成超时/无输出 | 某题空答 | 判题器判空为 INCORRECT（官方行为），记录 |
| 官方 57.0 偏差 | 分数不可比 | 先 `--limit 5 --no-judge` 冒烟验证生成路径，再全量 |
| 中途崩溃（断电/断连） | 丢进度 | 每题增量落盘 samples/aa_lcr.json，`--resume` 续跑 |

---

## 7. 完成后固化为标准

> 长上下文维度测完后，本方案 + `scripts/aa_lcr_run.py` + 配置即为**标准长上下文评测流水线**，与其他 4 维（GSM8K / MMLU-ProX-zh / C-Eval / MMMLU-zh / IFEval / MGSM）并列，五维全部有评测数据。
> 后续版本（微调后）在同一协议下追加评测，与本基线 v0.1 逐项对比。
