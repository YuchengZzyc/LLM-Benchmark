# 评测执行手册（Evaluation Runbook）

> 本文档面向复现者，记录 Qwen3.5-4B 通用能力评测的**完整可执行过程**：
> 能力维度与数据集的对应关系、所用数据、所用框架、环境搭建步骤、跑测命令与
> 结果产物位置。
>
> 关联文档：[evaluation_protocol.md](evaluation_protocol.md)（协议与版本规则）、
> [benchmark_description.md](benchmark_description.md)（各 benchmark 定义）、
> [environment_setup.md](environment_setup.md)（环境搭建说明）。
>
> **状态**：2026-09-10 在评测服务器上实际执行并验证。

---

## 1. 能力维度 ↔ 测试数据集对应关系

五个能力维度中，前四个由主框架 `lm-eval` 执行（`knowledge` / `reasoning` /
`instruction` / `multilingual`），长上下文分两条路径执行。

| # | 能力维度 | 数据集（lm-eval 任务名） | 上游数据源 | 指标 | few-shot |
| --- | --- | --- | --- | --- | --- |
| 1 | **知识与理解** | `mmlu_redux_generative` | `fxmarty/mmlu-redux-2.0-ok` | Accuracy | 任务默认 |
| | | `mmlu_prox_en` / `mmlu_prox_zh` | `li-lab/MMLU-ProX` | Accuracy | 5-shot |
| | | `ceval-valid` | C-Eval（52 学科验证集） | Accuracy | 任务默认 |
| | | ~~`mmlu_pro`~~ | `TIGER-Lab/MMLU-Pro` | Accuracy | 5-shot（**本轮跳过**，全量约 42h） |
| 2 | **专业推理** | `gsm8k` | GSM8K（OpenAI） | Exact Match | 5-shot |
| | | `leaderboard_math_hard` | `DigitalLearningGmbH/MATH-lighteval` | Accuracy | 4-shot |
| | | `aime24` / `aime25` | AIME 2024 / 2025 | Accuracy | 0-shot |
| | | `minerva_math500` | `HuggingFaceH4/MATH-500` | Accuracy | 4-shot |
| | | ~~`gpqa_diamond_zeroshot`~~ | `Idavidrein/gpqa` | Accuracy | 0-shot（**门控数据集**，需 HF_TOKEN，本轮摘除） |
| 3 | **指令遵循** | `ifeval` | IFEval | strict / loose accuracy | 0-shot |
| 4a | **长上下文（v2）** | `longbench2`（GROUP，20 叶任务） | `THUDM/LongBench_v2` | task score | 任务默认 |
| 4b | **长上下文（v1）** | `multifieldqa_en` `qasper` `hotpotqa` `2wikimqa` `musique` `gov_report` `multi_news` `trec` `triviaqa` `samsum` `passage_count` `passage_retrieval_en` | `THUDM/LongBench`（**须本地 JSONL**，见 §4.5） | 问答 F1/EM、摘要 ROUGE-L、分类/合成 accuracy | task-specific |
| 5 | **多语言** | `mgsm_native_cot_zh` | `juletxara/mgsm` | language accuracy (exact_match) | 0-shot（原生 CoT） |
| | | `mmmlu_zh_cn` | `openai/MMMLU`（简体中文，57 学科） | Accuracy | 任务默认 |
| 6 | **代码**（README 五维之外，配置中保留） | `humaneval` / `humaneval_plus` / `mbpp_plus` | OpenAI HumanEval / EvalPlus | pass@1 | 0/0/3-shot |

### 维度边界说明

- **长上下文有两条独立路径**：
  - `longbench2` 是 `lm-eval` 的**任务组**（LongBench **v2**，模型卡 50.0），由
    主框架执行，写在 `configs/*.yaml` 的 `tasks.longbench2` 段；
  - LongBench **v1** 的 12 个任务由独立接口 `scripts/run_longbench.py` 执行，
    写在 `tasks.long_context` 段。两者不可混放——把 `longbench2` 写进
    `long_context` 会被 `run_longbench.py` 的 `TASK_META` 校验拒绝。
- **近似替代项**（报告须注明，不做与模型卡的直接胜负结论）：
  HMMT → `leaderboard_math_hard` + `aime24/25`；LiveCodeBench → `humaneval` 系列；
  MATH-500 → `minerva_math500`（同源）。
- **本轮已知缺口**：`mmlu_pro`（耗时）、`gpqa_diamond_zeroshot`（门控数据）、
  `humaneval` 系列（需 `code_eval` 依赖，要求 Python ≥ 3.12，当前 venv 为 3.11.16）。

---

## 2. 框架与工具链

| 组件 | 版本 / 说明 |
| --- | --- |
| 主框架 | `lm-eval` 0.4.13（PyPI 包名 `lm-eval`；旧名 `lm-evaluation-harness` 已不存在） |
| 长上下文 | 仓库自研独立接口 `scripts/run_longbench.py` + `src/longbench.py`，按 LongBench 官方 `evaluate.py` 口径评分 |
| 模型推理 | `transformers` 5.6.0 + `torch` 2.11.0+cu128 + `accelerate` 1.11.0 |
| 数据处理 | `datasets` **4.0.0**（注意：4.x 已移除数据集脚本加载器，见 §3 踩坑与 §7） |
| 摘要评分 | `rouge-score` 0.1.2 |
| 中文指令遵循依赖 | `langdetect`、`immutabledict`（ifeval 需要，已补装） |
| 环境管理 | `uv`（非 conda） |
| 配置系统 | `configs/*.yaml`，禁止在代码中硬编码模型路径 / 任务名 / batch / dtype / 输出路径 |

---

## 3. 环境搭建

详见 [environment_setup.md](environment_setup.md)，服务器实际执行的等效步骤：

```bash
# 1) venv（uv 管理，实际落地 Python 3.11.16）
cd /data/yucheng/madm-llm/Benchmark
uv venv --python 3.11 .venv

# 2) GPU 版 torch（按服务器 CUDA 选择 index）
/data/yucheng/.local/bin/uv pip install --python .venv/bin/python \
    "torch" --index-url https://download.pytorch.org/whl/cu128

# 3) 主框架与其余依赖
/data/yucheng/.local/bin/uv pip install --python .venv/bin/python "lm-eval==0.4.13"
/data/yucheng/.local/bin/uv pip install --python .venv/bin/python -r requirements.txt

# 4) ifeval 额外依赖
/data/yucheng/.local/bin/uv pip install --python .venv/bin/python langdetect immutabledict

# 5) 验证
.venv/bin/python -c "import torch,lm_eval,rouge_score;print(torch.__version__,torch.cuda.is_available(),lm_eval.__version__)"
```

> **踩坑记录**
> - venv 内无 `pip` 模块，装包必须用 uv 全路径 + `--python <venv python>`；
>   非交互 SSH 的 `PATH` 不含 uv，故不能简写 `uv`。
> - `huggingface.co` 在本网络不可达（Errno 101），所有下载前需 `export HF_ENDPOINT=https://hf-mirror.com`。
> - lm-eval 0.4.13 的 `simple_evaluate` **不接受 `seed`** 参数，需改用
>   `random_seed`/`numpy_random_seed`/`torch_random_seed`/`fewshot_random_seed`；
>   仓库 `src/evaluator.py` 已做内省适配。
> - **`datasets` 4.0.0 已移除数据集脚本加载器**：在线加载 `THUDM/LongBench`
>   会报 `Dataset scripts are no longer supported, but found LongBench.py`，
>   导致 LongBench v1 三个任务全失败。须改为本地 JSONL —— 见 §7 与该文件
>   `docs/environment_setup.md` §5.1。
> - `humaneval` 系列需 `code_eval` 依赖，而 `code_eval` 要求 Python ≥ 3.12；
>   当前 venv 为 3.11.16，故代码维度本轮不可跑。

### 3.1 实测环境快照（2026-09-10）

```
服务器        192.168.13.230 (Ubuntu 26.04 LTS, kernel 7.0.0-29-generic)
CPU / RAM     AMD Ryzen 9 9950X 16-Core (32 线程) / 59 GB
GPU           NVIDIA GeForce RTX 5090 32GB, Driver 595.84 / CUDA 13.2
Python        3.11.16  (/data/yucheng/madm-llm/.venv)
torch         2.11.0+cu128  (torch.cuda.is_available() == True)
transformers  5.6.0
accelerate    1.11.0
datasets      4.0.0
lm_eval       0.4.13
rouge-score   0.1.2
模型          /data/yucheng/madm-llm/models/Qwen3.5-4B  (2 × safetensors 分片 + chat_template.jinja)
HF 缓存       ~/.cache/huggingface  (1.1 GB，二次运行无需联网)
```

---

## 4. 如何跑测试

### 4.1 配置选择

| 配置 | 用途 | max_samples | 适用场景 |
| --- | --- | --- | --- |
| `configs/smoke_qwen35_4b.yaml` | 冒烟：五维各覆盖 1+ 任务，每任务限量 20 条（`max_gen_toks` 512 旧口径，生成类分数偏低属已知） | 20 | 验证管线连通、数据可下载（约 2–3h） |
| `configs/baseline_qwen35_4b.yaml` | **v0.0 限量基线**：每任务/每学科叶 30 条、`max_gen_toks` 2048（2026-09-10 夜间执行） | 30 | 出具首个正式基线报告（约 6–7h） |
| （v1.0 预留） | 全量评测：`max_samples: null` + `model.version` 升 v1.0 | null | 论文级正式分数（预计 30h+，需协调时段） |
| `configs/template.yaml` | 新建版本（lora / dpo）时复制 | — | — |

### 4.2 单步运行

```bash
cd /data/yucheng/madm-llm/Benchmark
export HF_ENDPOINT=https://hf-mirror.com
PY=/data/yucheng/madm-llm/.venv/bin/python

# ① 主框架维度（knowledge / reasoning / instruction / longbench2 / code / multilingual）
$PY scripts/run_lm_eval.py --config configs/baseline_qwen35_4b.yaml

# ② 长上下文 LongBench v1（独立接口，可与 ① 串联）
$PY scripts/run_longbench.py --config configs/baseline_qwen35_4b.yaml

# ③ 汇总生成 Markdown 报告
$PY scripts/generate_report.py --config configs/baseline_qwen35_4b.yaml
```

可选项：`--tasks <任务名...>` 只跑指定任务；`--keep-samples` 落盘模型输出到
`results/<version>/samples/`；`--overwrite` 允许覆盖已存在的版本目录（默认拒绝）。

### 4.3 后台串联运行（推荐，避免 SSH 断连）

```bash
cd /data/yucheng/madm-llm/Benchmark
export HF_ENDPOINT=https://hf-mirror.com
PY=/data/yucheng/madm-llm/.venv/bin/python
nohup bash -c "$PY scripts/run_lm_eval.py --config configs/baseline_qwen35_4b.yaml \
  && $PY scripts/run_longbench.py --config configs/baseline_qwen35_4b.yaml \
  && $PY scripts/generate_report.py --config configs/baseline_qwen35_4b.yaml" \
  > /tmp/run_eval_BASELINE.log 2>&1 &
echo STARTED_PID=$!
```

监控：

```bash
tail -c 600 /tmp/run_eval_BASELINE.log | tr '\r' '\n' | tail -5   # 进度（tqdm 用 \r 刷新）
nvidia-smi                                                        # GPU 占用
ps -o pid,etime,pcpu -p <PID>                                     # 进程存活
```

### 4.4 失败降级行为

`src/evaluator.py` 的 `evaluate()` 会先把**所有主框架任务合成一批**交给
`simple_evaluate`（一次加载模型）；若整批失败，自动**按维度降级重跑**，
把坏任务隔离到单维度，避免一个任务（如门控数据集 401）拖垮全批。
`errors` 列表会记录失败原因，最终写入 `result.json`。

### 4.5 LongBench v1 数据准备（`datasets` 4.x 必需）

`datasets` 4.0.0 起不再支持数据集脚本，在线加载 `THUDM/LongBench`
（含 `LongBench.py`）会直接失败。跑 LongBench v1 前须**把官方 JSONL 本地化**：

```bash
cd /data/yucheng/madm-llm/Benchmark
mkdir -p data/longbench
# 从 LongBench 官方仓库下载所需任务的 jsonl（如 multifieldqa_en / hotpotqa / trec）
# 放入 data/longbench/<task>.jsonl
```

全量 12 任务对应的 JSONL 清单（`baseline_qwen35_4b.yaml` 的 `tasks.long_context`）：

```
multifieldqa_en  qasper  hotpotqa  2wikimqa  musique
gov_report  multi_news  trec  triviaqa  samsum
passage_count  passage_retrieval_en
```

并在配置中指向该目录（**非 null 即走本地直读，绕开数据集脚本**）：

```yaml
evaluation:
  longbench_data_dir: data/longbench
```

> `configs/smoke_qwen35_4b.yaml` 与 `configs/baseline_qwen35_4b.yaml` 均已设置
> `longbench_data_dir: data/longbench`。**数据现状（2026-09-10）**：12 任务完整
> 官方 JSONL 已从 hf-mirror 的 THUDM/LongBench `data.zip`（113MB）部署，每任务
> 200 条（`multifieldqa_en` 150 条）；早期版本会在评测前把 JSONL 就地截断为前
> N 条，**该逻辑已移除**，现改为 `_run_task` 内存截断（`samples[:cap]`），不再
> 改写数据文件。

---

## 5. 结果产物

```
results/<version>/
├── config.yaml      # 本次运行配置快照（自动保存）
├── logs/run.log     # 运行日志
├── samples/         # 模型样本输出（仅 --keep-samples 时）
├── result.json      # 结构化结果：dimensions.<维度>.<任务> -> 指标
└── report.md        # generate_report.py 生成的 Markdown 报告
results/registry.yaml  # 所有模型版本登记（status: pending/running/completed/failed）
```

`result.json` 结构示例：

```json
{
  "version": "qwen35-4b-base-v0.0",
  "model": "Qwen3.5-4B",
  "metrics": {
    "framework": "lm-eval",
    "dimensions": {
      "reasoning": {
        "gsm8k": {"sample_len": 20.0, "exact_match,strict-match": 0.25}
      },
      "long_context": {"status": "skipped", "message": "未配置 long_context 任务"}
    },
    "errors": []
  }
}
```

### 5.1 结果维护与版本复盘（基线 → 迭代）

**基线已就位**：`qwen35-4b-smoke-v0.0`（冒烟，2026-09-10 completed）已作为
第一条完整记录写入 `results/registry.yaml`；正式基线为 `qwen35-4b-base-v0.0`
（`baseline_qwen35_4b.yaml`，全量）。后续每次迭代**追加新版本号，旧版本永不覆盖**。

**新增迭代版本的标准流程**（以 LoRA 微调为例）：

```bash
cp configs/template.yaml configs/lora_r1_qwen35_4b.yaml
# 编辑：model.version: qwen35-4b-lora-v1.0（命名规范见 evaluation_protocol.md §4.1：
#   qwen35-4b-<type>-v<major>.<minor>）、model.path 指向微调后权重
# 其余评测参数（batch / dtype / seed / gen_kwargs / 任务清单）与 baseline 保持一致，
# 否则跨版本不可比
$PY scripts/run_lm_eval.py  --config configs/lora_r1_qwen35_4b.yaml
$PY scripts/run_longbench.py --config configs/lora_r1_qwen35_4b.yaml
$PY scripts/generate_report.py --config configs/lora_r1_qwen35_4b.yaml
```

**复盘入口**（三处，均随运行自动产生，无需手工维护）：

| 入口 | 内容 | 用途 |
| --- | --- | --- |
| `results/registry.yaml` | 所有版本的 status / results_path / registered_at / last_update | 总览：跑过哪些版本、各自状态、结果在哪 |
| `results/<version>/report.md` | 末尾 **Comparison 表**汇总 registry 中全部版本 | 跨版本状态对照 |
| `results/<version>/config.yaml` + `logs/run.log` | 当次配置快照与完整日志 | 复现该版本环境与参数；run.log 中的「评测结果: {...}」行是分数的原始出处 |

**状态机**：`initialized → pending → running → completed | failed`（由脚本自动流转）。
失败版本保留 `failed` 状态与错误日志，直接改 version 重跑即可，不污染旧记录。

**版本对比**：同 `output.root` 下各版本 `report.md` 的 Comparison 表 + 各自
`result.json` 逐任务指标 diff；正式基线出分后建议固定一份对比快照到
`reports/`（如 `reports/results_<version>_<date>.md`）。

---

## 6. 复现性

- 随机种子：`evaluation.seed`（默认 42），经内省映射到 harness 实际支持的 seed 参数。
- 每次运行保存 `config.yaml` 快照与 `logs/run.log`。
- 生成参数固定：`temperature=0.0, top_p=1.0, max_new_tokens=1024`（贪心解码）。
- `evaluation.apply_chat_template: true`，`dtype: bfloat16`，`batch_size: 8`
  （5090 32GB 实测：`16` 会在 multilingual 维度 OOM，已降为 8）。
- 数据缓存于 `~/.cache/huggingface`，二次运行无需联网。

---

## 7. 已知问题与注意事项

| 问题 | 影响 | 处理 |
| --- | --- | --- |
| `huggingface.co` 不可达 | 数据集首次下载失败 | 前缀 `HF_ENDPOINT=https://hf-mirror.com` |
| **`datasets` 4.0.0 移除数据集脚本加载** | LongBench v1 三任务报 `Dataset scripts are no longer supported, but found LongBench.py` 全失败 | 下载官方 JSONL 到 `data/longbench/`，配置 `evaluation.longbench_data_dir: data/longbench`（见 §4.5） |
| **LongBench 推理 `ones_like(): ... must be Tensor, not BatchEncoding`** | transformers 5.6.0 下 `apply_chat_template(..., return_tensors="pt")` 返回 `BatchEncoding`（dict 子类）而非 Tensor，直接喂 `torch.ones_like` 报错，LongBench v1 三任务全失败 | `src/longbench.py` `_generate()` 改为按键取 `input_ids` / `attention_mask` 再各自 `.to(device)`（2026-09-10 已修，重跑 3/3 通过） |
| **`save_result_json` 整体替换 `metrics`** | `run_longbench` 落盘时把 `run_lm_eval` 写好的主框架分数整个冲掉，`result.json` 只剩 `long_context`，报告缺 4 维度 | `src/result_manager.py` 改为**按维度合并**（`_merge_metrics`：`dimensions` 逐维度 update，其余顶层键覆盖；2026-09-10 已修，单测覆盖"先主框架后长上下文"与"同维度重跑"两场景） |
| `multilingual` 维度 OOM（`batch_size: 16`） | 评测中断 | `batch_size` 降为 `8`（5090 32GB 实测值），`baseline_qwen35_4b.yaml` 已同步 |
| `gpqa` 为门控数据集（401） | 曾连坐整批 `simple_evaluate` 失败 | 已从配置摘除，待配置 HF_TOKEN 后补跑 |
| `humaneval` 需 `code_eval`（Python≥3.12） | 代码维度不可跑 | venv 为 3.11.16，`baseline_qwen35_4b.yaml` 已将 `tasks.code` 置空，不阻塞主线 |
| `aime24` / `aime25` 自带 `max_gen_toks=32768` | 全量 AIME 单条生成上限极大，推理极慢（冒烟 20 条单条约 100s+） | 全链统一 `gen_kwargs.max_gen_toks=2048` 覆盖任务级 32768，控制单条耗时 |
| **基座生成冗长前导（v0.0 诊断 2026-09-10）** | 未微调基座在 chat 模板下输出 "Thinking Process" 式长前导（400–6000 字符）后才给答案；`max_gen_toks=512`（冒烟值）大量截断致生成类任务接近 0 分（gsm8k 0.05） | `max_gen_toks` 提到 2048（诊断实测 gsm8k 3 条 strict 1/3）；知识与 loglikelihood 类任务（mmmlu 等）不受影响，其低分属权重/口径本身差异 |
| **LongBench 冒烟就地截断会改写数据文件** | 早期版本把 `data/longbench/*.jsonl` 就地截断为前 N 条，毁掉本地完整数据 | 已改为 `_run_task` 内**内存截断**（`samples[:cap]`），不再落盘改写（2026-09-10 移除）；gen_kwargs 中 lm-eval 特有键（max_gen_toks/until）在 `_generate` 中过滤，生成长度按 TASK_META 官方口径 |
| lm-eval 0.4.13 无 `c_eval` / `mgsm` 根任务 | 任务名失效 | 正确名：`ceval-valid` / `mgsm_native_cot_zh` |
| 远端终端中文 GBK 乱码 | 无法直接读数 | SFTP 二进制下载后本地 diff / 查看 |
| 长命令 >180s paramiko 超时 | 远程调用中断 | 调大 `REMOTE_TIMEOUT`，或远端 `nohup` 落盘后轮询日志 |
| 冒烟分数不可用于结论 | `max_samples=20`，`sample_len` 仅 20 | 只作管线验证；正式分数须跑 `baseline_qwen35_4b.yaml` 全量 |
