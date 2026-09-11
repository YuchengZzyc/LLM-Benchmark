# Qwen3.5-4B 评测运行状态报告

**采集时间**：2026-09-10 14:16–14:17 CST（初次）／ 15:30 CST（更新，见 §8）
**采集方式**：只读巡检（SSH + `nvidia-smi` / `ps` / 日志读取），未改动任何代码或配置

> ⚠️ **§1–§7 为 14:17 时点的快照，部分内容已被 §8 更新覆盖**：13:03 那次冒烟运行
> 已在 14:45 结束且 LongBench v1 三任务全失败，15:11 已带修复重启。请以 §8 为准。

---

## 1. 结论速览（14:17 时点）

| 项目 | 状态 |
| --- | --- |
| 评测进程 | ✅ **正在运行**（PID 1417604，已跑 1h 13m） |
| GPU | ✅ 正常占用 27.9 / 32.6 GB，利用率 11–23%，52°C，约 200–230 W |
| 当前阶段 | ⏳ 冒烟评测 `smoke_qwen35_4b.yaml`，生成进度 **1517 / 1700 ≈ 89%** |
| 结果文件 | ⚠️ 尚未落盘 `result.json`（评测结束才写入，属预期行为） |
| 服务器负载 | ⚠️ load average ≈ 106，**非本任务引起**（见 §5） |

---

## 2. 在跑的是什么

后台串联任务（13:03 启动，PID 1417548 拉起）：

```bash
cd /data/yucheng/madm-llm/Benchmark
export HF_ENDPOINT=https://hf-mirror.com
PY=/data/yucheng/madm-llm/.venv/bin/python
nohup bash -c "$PY scripts/run_lm_eval.py --config configs/smoke_qwen35_4b.yaml \
  && $PY scripts/run_longbench.py --config configs/smoke_qwen35_4b.yaml \
  && $PY scripts/generate_report.py --config configs/smoke_qwen35_4b.yaml" \
  > /tmp/run_eval_SMOKE.log 2>&1 &
```

- 版本：`qwen35-4b-smoke-v0.0`（冒烟，**非正式分数**）
- 覆盖维度：knowledge / reasoning / instruction / multilingual（主框架）+
  long_context（LongBench v1 三个轻量任务）
- 限量：`max_samples: 20`，每任务最多 20 条，只为验证管线与数据可下载
- 阶段 ①（lm-eval）→ ②（LongBench v1）→ ③（生成报告）**顺序执行**，
  当前仍在阶段 ①

### 实时指标（14:17:15）

```
进程  PID 1417604   ELAPSED 01:13:46   CPU 1290%   RSS ≈ 2.7 GB
GPU   27926 MiB / 32607 MiB   util 23%   201 W   52 °C
日志  /tmp/run_eval_SMOKE.log  (22 KB)
进度  Running generate_until requests: 89% | 1517/1700 [1:04:56<25:38, 8.41s/it]
```

按 8.4 s/it 估算，**该批次剩余约 26 分钟（≈14:43 CST 完成）**；
后续还需依次完成其余任务批次、LongBench v1 与报告生成，整体完成时间以实际为准。

### 为什么日志前面有一堆 `20/20 [1300 it/s]`

那些是各任务的数据集**构建/加载**进度条（限量 20 条，CPU 侧，瞬间完成）；
真正的模型前向在后半段的 `generate_until requests` 大进度条里，
速度从早期的 1.5 s/it 逐步涨到 8.4 s/it，属于长序列生成的正常波动
（首轮出现过 `Multiple max token args provided: {'max_gen_toks': 2048, 'max_new_tokens': 1024}`，
harness 按优先级取了 `max_gen_toks=2048`）。

---

## 3. 已产出的结果

### 3.1 `results/qwen35-4b-smoke-v0.0/`（当前运行）

```
13:03   854 B   config.yaml        # 配置快照已存
13:03   684 B   logs/run.log       # 仅有启动头部
（尚无 result.json —— 评测完成时才写入）
```

### 3.2 `results/qwen35-4b-base-v0.0/`（此前调试图）

`result.json`（更新于 10:35）**只有一个 20 条样本的 gsm8k**：

| 任务 | sample_len | exact_match,strict-match | exact_match,flexible-extract |
| --- | --- | --- | --- |
| gsm8k | 20 | 0.25 | 0.1 |

> ⚠️ **这组数字不可用于任何结论**：样本量仅 20，且 95% 置信区间宽度约 ±0.2。
> `results/qwen35-4b-base-v0.0/logs/run.log` 显示 09:45–12:58 之间经历了
> 4 次短跑（含一次 `simple_evaluate() got an unexpected keyword argument 'seed'`
> 失败），均为**管线调试**，12:58 那次在全量配置下启动后于 13:03 被冒烟运行取代。
> 当前 `registry.yaml` 中两个版本 `status: pending`——冒烟跑完才会更新。

---

## 4. 阻塞项（已确认，不阻塞当前冒烟）

| 项 | 原因 | 状态 |
| --- | --- | --- |
| `mmlu_pro`（知识与理解） | 全量 40880 请求约 42h | 本轮跳过，知识维度由 MMLU-Redux / ProX / C-Eval 覆盖 |
| `gpqa_diamond_zeroshot`（推理） | `Idavidrein/gpqa` 为门控数据集，无 HF_TOKEN 匿名拉取 401，且曾连坐整批失败 | 已摘除，待配置 HF_TOKEN 补跑 |
| `humaneval` / `humaneval_plus` / `mbpp_plus`（代码） | 需 `code_eval` 依赖，要求 Python ≥ 3.12；venv 为 3.11.16 | 冒烟跳过，不阻塞主线 |
| `longbench2`（长上下文 v2） | 20 叶任务，冒烟阶段跳过 | 正式跑 `baseline_qwen35_4b.yaml` 时纳入 |

---

## 5. 值得注意：服务器 CPU 被他人占满

`load average: 106.33, 103.01, 76.36` 远超 CPU 核数，但**不是本评测造成**——
另一账号 `akuvox*` 正以 8 个进程各占 ~99% CPU 跑
`nn/scripts/eval_asr_grade.py`（ASR 评测，13:59 启动）：

```
akuvox2+ 3088543 ... 99% ... eval_asr_grade.py --ckpt nn/products/exp/R25/ckpt/x_cond.pt
akuvox2+ 3088545 ... 99% ...
（共 8 个子进程）
```

**影响**：GPU 不受影响（本任务独占 27.9 GB 显存）；但 lm-eval 的数据预处理、
指标计算（尤其 IFEval 的正则解析、MMLU 类多进程加载）在 CPU 侧会明显变慢。
当前观测到的 8.4 s/it 里，有一部分可能来自 CPU 争抢。**建议**：正式全量评测前
与对方协调 CPU 时段，或把 `batch_size` 调小以减少 CPU 侧队列压力。

磁盘无压力：`/` 已用 56 G / 3.7 T（2%）；HF 缓存 1.1 GB。

---

## 6. 下一步建议

1. **等冒烟跑完**（预计 14:43 后继续 LongBench v1）——观察 `result.json`
   是否五个维度齐全、有无 `errors`。
2. 冒烟通过后，用**全量配置**出正式分数：
   ```bash
   nohup bash -c "$PY scripts/run_lm_eval.py --config configs/baseline_qwen35_4b.yaml \
     && $PY scripts/run_longbench.py --config configs/baseline_qwen35_4b.yaml \
     && $PY scripts/generate_report.py --config configs/baseline_qwen35_4b.yaml" \
     > /tmp/run_eval_BASELINE.log 2>&1 &
   ```
   注意全量含 `mmlu_pro` 会大幅拉长时长（知识维度建议仍按配置注释跳过）。
3. 补齐三项缺口：申请 HF_TOKEN 跑 GPQA；升 Python 到 3.12 装 `code_eval` 跑代码维度；
   确认是否值得为 `mmlu_pro` 单独排 42h 档期。
4. 完成后核对 `reports/*/report.md`，**不要**引用 `sample_len=20` 的旧数字。

---

## 7. 附：巡检命令

```bash
# 进程与 GPU
ps -ef | grep -E "run_lm_eval|run_longbench" | grep -v grep
nvidia-smi
# 进度（tqdm 用 \r 刷新，需先转换）
tail -c 600 /tmp/run_eval_SMOKE.log | tr '\r' '\n' | tail -5
# 产物
find results/qwen35-4b-smoke-v0.0 -printf "%TH:%TM %10s %p\n"
```

> 关联文档：[docs/evaluation_runbook.md](../docs/evaluation_runbook.md)（完整评测执行手册）

---

## 8. 事件跟进（15:30 CST 更新）

### 8.1 时间线

| 时间 | 事件 |
| --- | --- |
| 13:03 | 冒烟链启动（PID 1417548），跑 lm-eval 主框架 → LongBench v1 → 报告 |
| ~14:17 | lm-eval 主框架阶段进度 89%（1517/1700），GPU 27.9 GB，CPU 1.5→8.4 s/it |
| ~14:43 | 阶段① （lm-eval）收尾 |
| **14:45** | **阶段② LongBench v1 三任务全失败**（见 8.2），链路在此结束并写入 `result.json` |
| 15:06 | 官方 JSONL 落地远端 `data/longbench/`（multifieldqa_en / hotpotqa / trec） |
| **15:11** | **带两项修复重启**（PID 3128069/3128070，`--overwrite`），日志 `/tmp/run_eval_SMOKE2.log` |

### 8.2 故障与修复（两处，均已由远端就地修复）

**① LongBench v1 三任务失败**（14:45）

```
[ERROR] run_longbench: [multifieldqa_en] 失败: Dataset scripts are no longer supported, but found LongBench.py
[ERROR] run_longbench: [hotpotqa]         失败: Dataset scripts are no longer supported, but found LongBench.py
[ERROR] run_longbench: [trec]             失败: Dataset scripts are no longer supported, but found LongBench.py
[WARNING] run_longbench: 存在失败任务: ['multifieldqa_en', 'hotpotqa', 'trec']
```

原因：远端 `datasets` **4.0.0 已移除数据集脚本加载器**，无法在线加载 `THUDM/LongBench`
（其含 `LongBench.py`）。修复：把官方 JSONL 下载到 `data/longbench/`，配置
`evaluation.longbench_data_dir: data/longbench` 改走本地直读。

> 已确认 15:06 三个 JSONL 就位（`hotpotqa.jsonl` 11.5 MB / `multifieldqa_en.jsonl` 4.5 MB /
> `trec.jsonl` 6.4 MB），`smoke_qwen35_4b.yaml` 已设 `longbench_data_dir: data/longbench`。

**② `multilingual` 维度 OOM**

`batch_size: 16` 在 multilingual 维度触发显存不足，配置降为 **`batch_size: 8`**。

### 8.3 15:11 重启后的实测（15:30 快照）

```
进程    PID 3128069(bash 链) / 3128070(run_lm_eval)  起始 15:11
GPU     11403 MiB / 32607 MiB   util 81%   57°C
进度    generate_until requests: 35% | 625/1800 [11:01<21:29, 1.10 s/it]
错误    grep -c ERROR = 0（新链至今无报错）
load    average 19.04, 8.97, 11.55（较 14:17 的 106 大幅回落）
```

- 新链 `smoke_qwen35_4b.yaml`：`batch_size: 8`、`longbench_data_dir: data/longbench`，
  总请求量 **1800**（旧轮 1700）。
- 吞吐明显改善：**1.10 s/it**（旧轮 CPU 争抢时最高 8.4 s/it）。
- `results/qwen35-4b-smoke-v0.0/` 已由 `--overwrite` 重建（15:11，`config.yaml` 862 B、
  `logs/run.log` 684 B、`samples/` 空）；上一轮的 `result.json` 随目录一并清除，
  但旧日志 `/tmp/run_eval_SMOKE.log`（42 KB，截至 14:45:33）**仍保留**，可供复盘。
- `results/registry.yaml` 中两个版本仍为 `status: pending`。

### 8.4 需要留意（供全量跑前处理）

| 项 | 现状 | 建议 |
| --- | --- | --- |
| `configs/baseline_qwen35_4b.yaml` | 仍为 `batch_size: 16`、`longbench_data_dir: null` | 正式跑之前须同步改为 `8` 与 `data/longbench`，否则会重演 LongBench 全失败 + OOM |
| LongBench v1 修复验证 | 15:11 链尚未跑到阶段② | 等阶段① 完成后确认 `result.json` 中 `multifieldqa_en`/`hotpotqa`/`trec` 有分数、`errors` 为空 |
| GPQA / 代码维度缺口 | 未变（见 §4） | 申请 HF_TOKEN；或将 Python 升至 3.12 装 `code_eval` |
