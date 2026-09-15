# Qwen3.5-4B 通用能力评测标准说明书（Evaluation Spec）

> **状态：冻结 v1.0**（2026-09-11）
> 本文档是评测的**唯一权威来源**。任何智能体在没有任何其他上下文的情况下，
> 仅凭本文档即可在评测服务器上复现全部正确的测试。
> 与本文件冲突的历史文档（`evaluation_runbook.md` / `evaluation_protocol.md` 的旧口径部分）
> 一律以本文件为准。
>
> 冻结依据：`results/qwen35-4b-base-v0.1`（2026-09-11 实测完成，registry `completed`）。
> 冻结对象：**五个能力维度、六个 lm-eval 数据集、一个 AA-LCR 长上下文任务组**。

---

## 1. 冻结范围（维度 ↔ 数据集 ↔ 指标）

五个能力维度全部冻结。前四个维度由主框架 **lm-eval 0.4.13** 执行；长上下文维度由仓库
独立接口 `scripts/aa_lcr_run.py` 执行（**不绑定 lm-eval**）。

| # | 维度 | lm-eval 任务 | 上游数据源 | 指标 | few-shot | 样本量 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 知识与理解 | `ceval-valid` | C-Eval（52 学科验证集） | acc（纯文本 MCQ） | 5 | 1346（全量） |
| | | `mmlu_prox_zh` | `li-lab/MMLU-ProX`（中文） | exact_match（生成式+正则抽取） | 5 | 840（60/科×14） |
| 2 | 专业推理 | `gsm8k` | GSM8K（OpenAI） | exact_match（strict / flexible） | 5 | 1319（全量） |
| 3 | 指令遵循 | `ifeval` | IFEval | strict / loose accuracy（inst / prompt 级） | 0 | 541（全量） |
| 4 | 多语言 | `mgsm_native_cot_zh` | `juletxara/mgsm`（中文子集） | exact_match（flexible-extract） | 0（原生 CoT） | 250（全量） |
| | | `global_mmlu_zh` | openai/MMMLU（简体中文） | acc（纯文本 MCQ） | 5 | 400（采样） |

### 1.1 长上下文维度（AA-LCR，独立接口）

**2026-09-11 决策：最终长上下文评测 = AA-LCR**（前置决策与取舍见
[aa_lcr_plan.md](aa_lcr_plan.md)，LongBench v1/v2 已取消）。配置：
`configs/aa_lcr_qwen35_4b_v01.yaml`，执行器 `scripts/aa_lcr_run.py`。
评分完全按 ArtificialAnalysis 官方 v1.1 协议（LLM 判题器 `gpt-5.6-luna`）；
目标为逐位复现模型卡 **57.0**。

| 项 | 冻结值 |
| --- | --- |
| 数据集 | `ArtificialAnalysis/AA-LCR` v1.1（100 题，text-only） |
| 数据源 | `AA-LCR_Dataset.csv` + 解压 `aa_lcr_data/lcr/{category}/{set_id}/{filename}.txt`（服务器已部署；文件名须 NFC 归一化） |
| 生成引擎 | HF generate（fla 0.3.0 Triton 内核），**单进程串行** 100 题（vLLM 双进程设计实测显存不足而作废，见 [aa_lcr_plan.md](aa_lcr_plan.md) §3） |
| 生成参数 | greedy（temperature=0, top_p=1）、`max_new_tokens=4096`、`enable_thinking=True`（官方 57.0 为带思考口径；run2 关 thinking + 1024 截断仅 32.7%） |
| 上下文 | 每题全量文档（~94k–99k tokens）**不截断**（`max_length 262144`） |
| 文档顺序 | 严格按 CSV `data_source_filenames` 顺序，不得乱序 |
| 判题器 | `gpt-5.6-luna`（base `http://192.168.13.165:8002`），官方 v1.1 prompt |
| 指标 | accuracy = CORRECT / 100 |
| 实测耗时 | 生成 9484s（~2h38m，单题 ~95s）+ 判题（2026-09-11 run3 实测） |

> LongBench 定位说明：LongBench v1/v2 实测不可复现官方分（v1 短 max_new_tokens 下复读、
> v2 判题器不可用），**退出冻结范围，不作为验收口径**；正式长上下文分数 = AA-LCR。
> `scripts/run_longbench.py` 保留供排障/研究。2026-09-11 已按 v0.1 口径补跑一轮
> **LongBench v1 × 12 参考评测**（每任务 30 样本，max_length 32768，chat 模板），
> 分数并入 `result.json` 的 `metrics.dimensions.long_context`，仅作参考，不参与与官方对齐。

### 1.2 明确不在冻结范围（不得混入）

- `longbench2`（LongBench v2 任务组，lm-eval 组任务）——与 v1 是两条独立路径，任务名
  不可混入 `tasks.long_context`（会被 `TASK_META` 校验拒绝）。
- `mmlu_redux_generative`、`mmlu_pro`、`aime24/25`、`leaderboard_math_hard`、
  `minerva_math500`、`gpqa_*`、`humaneval` 系列——v0.0 时代候选，**未冻结**。
  后续若需追加，须先走「新增维度流程」修订本文档。
- LongBench v1/v2 —— **不作验收口径**（实测不可复现官方分，见 §1.1 定位说明；
  v1 × 12 已于 2026-09-11 以 30 样本/任务补跑一轮参考分，见 §1.1）。
- MMLU-ProX 官方全量（英+中混合 12,412）——耗时过高的全量扩展，未冻结。

---

## 2. 统一口径（冻结，不可更改）

### 2.1 两条核心口径（v0.1 修复结论，全部误差源于违反这两条）

Qwen3.5-4B 的 chat 模板默认注入 thinking 前导（`<think>\n` + 400–6000 字思考），
会以两种方式污染分数。按任务类型分两种正确口径：

| 任务类型 | 口径 | 原因 |
| --- | --- | --- |
| **生成式任务**（`gsm8k`、`mmlu_prox_zh`、`ifeval`、`mgsm_native_cot_zh`） | `apply_chat_template=True` + `model_args` 追加 `enable_thinking=False,think_end_token=</think>` | 关闭 thinking 前导，生成路径直接出答案；否则答案被截断、正则抽取失败（v0.0：gsm8k 仅 0.40） |
| **loglikelihood 选择题**（`ceval-valid`、`global_mmlu_zh`） | `apply_chat_template=False` 纯文本 MCQ（`"Answer:"` 选字母） | chat 模板在选项 token 前插入 `<think>\n\n</think>\n\n` 空块，污染选项对数概率对比（探针：chat 模板仅 0.41 vs 纯文本 0.72） |

**AA-LCR 独立接口的口径**（由 `scripts/aa_lcr_run.py` 固化）：
官方 prompt 模板（`BEGIN INPUT DOCUMENTS ... END INPUT DOCUMENTS` +
`START/END QUESTION`）逐字还原；文档按 `data_source_filenames` 顺序拼接
（`BEGIN DOCUMENT {i}:\n{doc}\nEND DOCUMENT {i}`）；每题全量上下文不截断；
生成用 HF generate 单进程串行（fla Triton 内核，`max_new_tokens=4096`、
`enable_thinking=True`、贪心）；判题用 `gpt-5.6-luna`
官方 v1.1 system/user prompt，解析 JSON `{"verdict": ...}`，失败重试，
`accuracy = CORRECT / 100`。判题器细节见 [aa_lcr_plan.md](aa_lcr_plan.md) §2.3。

### 2.2 统一参数（所有任务一致，不得更改）

| 参数 | 冻结值 | 说明 |
| --- | --- | --- |
| `model_args`（生成式） | `pretrained=/data/yucheng/madm-llm/models/Qwen3.5-4B,dtype=bfloat16,enable_thinking=False,think_end_token=</think>` | 四任务统一 |
| `model_args`（选择题） | `pretrained=/data/yucheng/madm-llm/models/Qwen3.5-4B,dtype=bfloat16` | 纯文本分支不带 thinking 参数 |
| 解码 | `temperature=0.0, top_p=1.0`（贪心） | AA-LCR 亦同 |
| 随机种子 | `random_seed=numpy_random_seed=torch_random_seed=fewshot_random_seed=42` | **lm-eval 0.4.13 不接受 `seed` 参数**，四路必须全部显式传 42 |
| 批大小 | `batch_size=8` | 实测 16 在 multilingual 维度 OOM |
| few-shot | `gsm8k=5`、`ceval-valid=5`、`global_mmlu_zh=5`、`mmlu_prox_zh=5`、`ifeval=0`、`mgsm=0`（原生 CoT） | 见 §1 表 |
| `max_gen_toks` | gsm8k / mgsm：`1024`；ceval / mmlu_prox / ifeval / mcq_probe：`2048` | 各脚本已固化 |
| 数据网络 | 命令前必须 `export HF_ENDPOINT=https://hf-mirror.com` | huggingface.co 不可达（Errno 101） |

> **禁止事项**：
> - 禁止给生成式任务用默认模板（thinking 开）跑正式分数；
> - 禁止给选择题用 `apply_chat_template=True`；
> - 禁止改 `model_args` 的 pretrained 路径 / dtype；
> - 禁止覆盖已存在的 `results/<version>/` 目录（脚本默认拒绝，须换新 version）。

---

## 3. 环境搭建

### 3.1 拓扑与路径

| 项 | 值 |
| --- | --- |
| 评测服务器 | `yucheng@192.168.13.230`（Ubuntu 26.04 LTS，RTX 5090 32GB） |
| 仓库 | `/data/yucheng/madm-llm/Benchmark` |
| venv | `/data/yucheng/madm-llm/.venv`（uv 管理，Python 3.11.16） |
| 模型权重 | `/data/yucheng/madm-llm/models/Qwen3.5-4B`（本地目录，bfloat16） |
| AA-LCR 数据 | `/data/yucheng/AA-LCR_Dataset.csv` + `aa_lcr_data/lcr/`（230 个 .txt） |
| 开发机（Windows） | 仅编辑/推送代码；**评测只在服务器运行** |

已实测版本（environment.txt）：torch 2.11.0+cu128、transformers 5.6.0、datasets 4.0.0、
lm-eval 0.4.13、rouge-score 0.1.2。

### 3.2 从零搭建（仅在服务器无环境时执行）

```bash
cd /data/yucheng/madm-llm/Benchmark
# 1) venv（uv 管理；本服务器落地 Python 3.11.16）
uv venv --python 3.11 .venv

# 2) GPU 版 torch（按服务器 CUDA 选 index；cu128 实测）
/data/yucheng/.local/bin/uv pip install --python .venv/bin/python \
    "torch" --index-url https://download.pytorch.org/whl/cu128

# 3) 主框架与依赖（PyPI 包名是 lm-eval，旧名 lm-evaluation-harness 已不存在）
/data/yucheng/.local/bin/uv pip install --python .venv/bin/python "lm-eval==0.4.13"
/data/yucheng/.local/bin/uv pip install --python .venv/bin/python -r requirements.txt

# 4) ifeval 额外依赖（不随 lm-eval 自动安装）
/data/yucheng/.local/bin/uv pip install --python .venv/bin/python langdetect immutabledict

# 5) 验证
export HF_ENDPOINT=https://hf-mirror.com
.venv/bin/python -c "import torch,lm_eval,rouge_score;print(torch.__version__,torch.cuda.is_available(),lm_eval.__version__)"
```

> **踩坑须知**：venv 内**无 `pip` 模块**，非交互 SSH 的 PATH 不含 uv → 装包必须用
> `/data/yucheng/.local/bin/uv pip install --python <venv-python>` 全路径写法。

### 3.3 AA-LCR 数据（已本地化）与 LongBench 数据（归档）

**AA-LCR 数据**（长上下文维度的正式数据，服务器已部署）：
`/data/yucheng/AA-LCR_Dataset.csv`（145KB，100 行）+ 解压后
`/data/yucheng/aa_lcr_data/lcr/{category}/{set_id}/{filename}.txt`（230 个）。
拉取失败时从 hf-mirror 的 `ArtificialAnalysis/AA-LCR` 获取 CSV + `extracted-text.zip`
（113MB）用 `curl -sL` 拉取解压，**严格按 `data_source_filenames` 顺序加载**。

**LongBench 数据**（已归档，仅供 `run_longbench.py` 排障/研究，不产生正式分数）：
12 任务的官方 JSONL 放 `data/longbench/<task>.jsonl`，配置写
`evaluation.longbench_data_dir: data/longbench`。`datasets` 4.0.0 已移除数据集
脚本加载器，在线加载 `THUDM/LongBench` 必报
`Dataset scripts are no longer supported, but found LongBench.py`。

---

## 4. 运行测试（冻结命令）

所有命令在服务器 `/data/yucheng/madm-llm/Benchmark/` 下执行。先设公共环境变量：

```bash
export HF_ENDPOINT=https://hf-mirror.com
cd /data/yucheng/madm-llm/Benchmark
PY=/data/yucheng/madm-llm/.venv/bin/python
```

### 4.1 六个数据集（主框架，lm-eval）

```bash
# ① GSM8K 全量 1319（~15min）
$PY gsm8k_run.py

# ② C-Eval 5-shot 全量 1346（~5min）
$PY ceval_run.py

# ③ MMMLU-zh 纯文本 MCQ（A/B 两组都跑，~4min）
$PY mcq_probe.py          # 冻结口径取 A 组（apply_chat_template=False）的 acc
```

> 注：冻结实测值 0.71750 对应 **400 采样**。`mcq_probe.py` 默认无 limit 时跑全量
> `global_mmlu_zh`（约 1710 条，stderr 波动 ±1%）；若需与 0.71750 严格对齐，
> 采样数须与 v0.1 相同（400 条），否则属采样差异而非口径错误。

```bash
# ④ MMLU-ProX-zh 60/科 × 14 科 = 840（~70min；A_B_GROUP=B 只跑 B 组=关 thinking）
A_B_GROUP=B A_B_TASKS=mmlu_prox_zh A_B_LIMIT=60 $PY a_b_tasks.py

# ⑤ IFEval 全量 541（~40min）
$PY ifeval_run.py

# ⑥ MGSM 中文原生 CoT 全量 250（~10min）
$PY mgsm_run.py
```

### 4.2 AA-LCR 长上下文（独立接口，~2h40m 生成 + 判题）

```bash
# 数据（服务器已部署）：AA-LCR_Dataset.csv + aa_lcr_data/lcr/ 解压文档
# 生成引擎为 HF generate + fla 0.3.0 Triton 内核（已装，无需 vLLM）
# 全量 100 题（单进程串行，增量落盘，崩溃可 --resume 续跑）：
nohup env JUDGE_API_KEY="$JUDGE_API_KEY" \
    $PY scripts/aa_lcr_run.py --config configs/aa_lcr_qwen35_4b_v01.yaml \
    > /data/yucheng/aa_lcr_run.log 2>&1 &
```

可选：`--max-questions` 冒烟（如 5 题验证判题器+生成路径）；`--keep-samples`
保存每题判题样本；`--overwrite` 允许覆盖版本目录（默认拒绝）。

### 4.3 采样/覆盖环境变量（冒烟、复测用）

| 脚本 | 环境变量 | 默认（冻结） |
| --- | --- | --- |
| `gsm8k_run.py` | `GSM8K_LIMIT` / `GSM8K_THINKING` | 全量 / `false` |
| `ceval_run.py` | `CEVAL_LIMIT` / `CEVAL_TASKS` / `CEVAL_FEWSHOT` | 全量 / `ceval-valid` / 5 |
| `ifeval_run.py` | `IFEVAL_LIMIT` / `IFEVAL_TASKS` | 全量 / `ifeval` |
| `mgsm_run.py` | `MGSM_LIMIT` / `MGSM_TASKS` | 全量 / `mgsm_native_cot_zh` |
| `mcq_probe.py` | `MCQ_LIMIT`（脚本需自行加 `limit=` 透传；v0.1 实测为 400 采样） | 全量 / 采样见上 |
| `a_b_tasks.py` | `A_B_LIMIT` / `A_B_TASKS` / `A_B_GROUP` | 30 / 注入任务 / `AB` |
### 4.4 后台运行（推荐，避免 SSH 断连）

```bash
nohup $PY gsm8k_run.py > /tmp/gsm8k.log 2>&1 &
# 监控：tail -c 600 /tmp/gsm8k.log | tr '\r' '\n' | tail -5
```

### 4.5 结果产物

```
results/<version>/
├── config.yaml      # 运行配置快照（自动保存）
├── logs/run.log     # 完整日志（run.log 的「评测结果: {...}」行是分数原始出处）
├── samples/         # 模型输出（仅 --keep-samples）
├── result.json      # 结构化结果（dimensions.<维度>.<任务> -> 指标）
└── report.md        # generate_report.py 生成
results/registry.yaml  # 全部版本登记（pending/running/completed/failed）
reports/results_<version>_<date>.md   # 正式报告快照
```

`src/result_manager.py` 按**维度合并**写入（`_merge_metrics`），主框架与 AA-LCR
（及归档的 LongBench）共用同一 `result.json`，不会互相覆盖。
每次迭代**追加新 version，旧版本永不覆盖**。

---

## 5. 验收基线（冻结的实测值 = 复现目标）

以下为 v0.1 在服务器上的实测值。**同口径 + 同种子 + 贪心解码应逐位复现**；
若偏差 > ±0.005，先按 §7 排查口径（thinking / 模板 / 批次），不要用新口径重测。

| 数据集 | 样本 | 指标 | 冻结实测值 | 官方参考 | 复现命令 |
| --- | --- | --- | --- | --- | --- |
| GSM8K | 1319 | exact_match (strict) | **0.90296** | 0.889 | ① |
| C-Eval (valid) | 1346 | acc | **0.74963** | 0.851 | ② |
| MMMLU-zh | 400 | acc | **0.71750** | 0.761 | ③（A 组） |
| MMLU-ProX-zh | 840 | exact_match (custom-extract) | **0.63452** | 0.715 | ④ |
| IFEval | 541 | inst_level_loose_acc | **0.89928** | 0.898 | ⑤ |
| MGSM (zh) | 250 | exact_match (flexible-extract) | **0.764** | — | ⑥ |
| AA-LCR | 100 | accuracy（gpt-5.6-luna 判题） | **0.590** | 57.0 | §4.2 |

> 与官方的已知差距（属数据/采样/模板差异，**不是口径错误**）：
> C-Eval 差 −10 点（lm-eval `ceval-valid` 缺官方 description 上下文模板）；
> MMLU-ProX-zh 差 −8 点（60/科采样仅官方全量 6.8%）；MMMLU-zh 差 −4 点（400 采样）；
> MGSM strict=0（官方格式"答案是 X。"正则不命中，flexible 76.4% 为冻结口径）。

---

## 6. 从零上下文复现：端到端流程

1. **连服务器**：`ssh yucheng@192.168.13.230`（Windows 端用 `_remote_check.py` +
   `REMOTE_PW` 环境变量，密码**不落盘、不写入任何文件/报告**）。
2. **验证环境**：按 §3.2 步骤 5 的验证命令跑通（`torch.cuda.is_available()==True`、
   `lm_eval.__version__=='0.4.13'`）。
3. **确认数据**：`AA-LCR_Dataset.csv`（100 行）+ `aa_lcr_data/lcr/`（230 个 .txt）
   存在；模型目录存在。
4. **跑六个数据集**：依次执行 §4.1 的 ①–⑥，共约 2.5h。
5. **跑 AA-LCR**：§4.2，约 2h40m 生成 + 判题。
6. **核对**：对照 §5 验收表逐项比对；偏差过大 → §7 排查。
7. **写报告**：`$PY scripts/generate_report.py --config <version 配置>`，结果登记
   `results/registry.yaml`（脚本自动），报告快照放 `reports/`。

---

## 7. 常见故障排查（全部为本项目实测踩坑）

| 现象 | 根因 | 处理 |
| --- | --- | --- |
| 生成式任务分低（gsm8k≈0.4） | thinking 前导污染 | 确认 model_args 含 `enable_thinking=False,think_end_token=</think>`（§2.1） |
| 选择题分≈随机（mmmlu≈0.41） | chat 模板污染选项概率 | 确认 `apply_chat_template=False` 纯文本（§2.1） |
| 下载报 Errno 101 / SSL 握手超时 | huggingface.co 不可达 | 命令前 `export HF_ENDPOINT=https://hf-mirror.com` |
| `Dataset scripts are no longer supported` | datasets 4.0.0 | 用 `data/longbench/` 本地 JSONL + `longbench_data_dir`（§3.3） |
| `simple_evaluate() got unexpected 'seed'` | 0.4.13 无 seed 参数 | 四路 `*_random_seed=42`（§2.2） |
| multilingual OOM | batch 16 超显存 | `batch_size=8` |
| venv 装包失败 | 无 pip / uv 不在 PATH | `/data/yucheng/.local/bin/uv pip install --python .venv/bin/python ...` |
| paramiko SFTP ENOENT（本地 Windows 端） | Git Bash 把 `/data/...` 当路径转换 | 命令前 `MSYS_NO_PATHCONV=1` |
| 远端中文乱码 | GBK 显示 | SFTP 二进制下载后本地查看（`_remote_pull.py`） |
| 长命令超时（>180s） | paramiko PipeTimeout | 调大 `REMOTE_TIMEOUT`，或远端 nohup + 轮询日志 |
| LongBench 全 0 分 | 中间截断切掉问题文本 / 复读 | LongBench v1 已取消；AA-LCR 用 vLLM + 官方 prompt 模板，不复用此路径 |
| ifeval ImportError | 缺 langdetect/immutabledict | 按 §3.2 步骤 4 补装 |
| 结果目录已存在被拒 | 默认拒绝覆盖 | 换新 `model.version` 或显式 `--overwrite`（慎用） |

---

## 8. 变更记录与版本规则

- **2026-09-11（本文件，v1.0 冻结）**：基于 `qwen35-4b-base-v0.1` 已验证口径
  （生成式关 thinking + 选择题纯文本）冻结上述五维六集；长上下文维度
  定为 **AA-LCR**（目标复现官方 57.0），
  LongBench v1/v2 已取消。
  取代 v0.0 时代"全走 chat 模板"的口径（其分数仅作污染基线，见
  `results/qwen35-4b-base-v0.0/`）。
- **2026-09-11 晚（run3 落盘后的更正，口径不变）**：AA-LCR 全量 100 题完成，
  实测 **0.590**（官方 57.0，+2.0，100/100 判题），五维全闭环。
  实际执行引擎为 HF generate 单进程串行（fla Triton 内核，
  `max_new_tokens=4096`、`enable_thinking=True`、全量上下文不截断），
  §1.1/§2.1/§4.2/§5 已按实跑口径更正（原 vLLM 双进程 + 90k 截断为作废设计）。
  另补跑 LongBench v1 × 12 参考分（30 样本/任务）并入 result.json，
  不作验收口径（见 §1.1 定位说明）。
- **后续迭代**（微调 / LoRA / DPO 等）：复制 `configs/template.yaml`，改
  `model.version`（命名 `qwen35-4b-<type>-v<major>.<minor>`）与 `model.path`，
  **评测参数一律沿用本文档 §2 冻结口径**，否则跨版本不可比。
- **追加新维度**（如 MMLU-Redux 全量、LongBench v2 恢复）：先在探针脚本上验证口径、
  与官方对齐，**修订本文档并提升版本号**后方可纳入正式评测。
