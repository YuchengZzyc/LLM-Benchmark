# 冒烟评测结果汇总 — Qwen3.5-4B（`qwen35-4b-smoke-v0.0`）

> **采集时间**：2026-09-10 18:03–18:05 CST（只读巡检）
> **配置**：`configs/smoke_qwen35_4b.yaml`（`max_samples: 20`）
> **数据来源**：远端 `/data/yucheng/madm-llm/Benchmark/results/qwen35-4b-smoke-v0.0/` 的
> `result.json`、`logs/run.log`、`logs/longbench.log`，以及 `/tmp/run_eval_SMOKE2.log`
> **用途**：管线连通性验证。**冒烟分数不构成任何模型结论**（详见文末免责）。

---

## 1. 总览

| 阶段 | 内容 | 起止 | 耗时 | 结果 |
| --- | --- | --- | --- | --- |
| ① 主框架 | `run_lm_eval.py`，9 个任务（4 维度） | 15:11:56 → 17:47:45 | **2 h 35 m 49 s** | ✅ 9/9 |
| ② 长上下文 v1 首次 | `run_longbench.py` | 17:47:46 → 17:47:50 | 4 s | ❌ 3/3 报错（见 §4.1） |
| ②′ 长上下文 v1 重跑 | `run_longbench.py`（修复后重启动） | 18:01:37 → 18:03:06 | 1 m 29 s | ✅ 3/3 |
| ③ 出报告 | `generate_report.py` | 18:03:08 | <1 s | ✅ 已生成 |

- registry：`qwen35-4b-smoke-v0.0` → **`status: completed`**（18:03:06）
- 全链总耗时：15:11:56 → 18:03:06 ≈ **2 h 51 m**
- 日志中 `ERROR` 计数：修复重跑后 **0**

---

## 2. 主框架结果（`run_lm_eval.py`，9/9）

**注意**：这 9 组分数的**唯一留存处是 `logs/run.log` 的一行 JSON**——`result.json` 里已经没有了，
原因见 §4.2。下表即从该日志行还原。

### 2.1 知识与理解

| 任务 | 样本数 | 指标 | 分数 | stderr |
| --- | --- | --- | --- | --- |
| `mmlu_redux_generative` | 1140（57 叶 × 20） | `exact_match` | **0.2289** | 0.0125 |
| `mmlu_prox_en` | 280（14 叶 × 20） | `exact_match`（custom-extract） | **0.2286** | 0.0242 |
| `mmlu_prox_zh` | 280（14 叶 × 20） | `exact_match`（custom-extract） | **0.3500** | 0.0273 |

### 2.2 专业推理

| 任务 | 样本数 | 指标 | 分数 | stderr |
| --- | --- | --- | --- | --- |
| `gsm8k` | 20 | `exact_match`（strict） | **0.0500** | 0.0500 |
| | | `exact_match`（flexible） | **0.1000** | 0.0688 |
| `aime24` | 20 | `exact_match` | **0.0000** | 0.0000 |
| `minerva_math500` | 20 | `exact_match` | **0.0000** | 0.0000 |
| | | `math_verify` | **0.0500** | 0.0500 |

### 2.3 指令遵循

| 任务 | 样本数 | 指标 | 分数 |
| --- | --- | --- | --- |
| `ifeval` | 20 | `prompt_level_strict_acc` | **0.1000**（stderr 0.0688） |
| | | `inst_level_strict_acc` | **0.3000** |
| | | `prompt_level_loose_acc` | **0.1000**（stderr 0.0688） |
| | | `inst_level_loose_acc` | **0.3000** |

### 2.4 多语言

| 任务 | 样本数 | 指标 | 分数 | stderr |
| --- | --- | --- | --- | --- |
| `mgsm_native_cot_zh` | 20 | `exact_match`（strict / flexible） | **0.0000** | 0.0000 |
| `mmmlu_zh_cn` | 1140（57 叶 × 20） | `acc` | **0.2404** | 0.0127 |
| | | `acc_norm` | **0.2404** | 0.0127 |

---

## 3. 长上下文 v1 结果（`run_longbench.py`，3/3）

数据源：本地 JSONL（`data/longbench/`），已绕开 `datasets` 4.x 的脚本加载器限制。

| 任务 | 类别 | 样本数 | 指标 | 分数 |
| --- | --- | --- | --- | --- |
| `multifieldqa_en` | single_doc_qa | 20 | `em` / `f1` | **0.0** / **0.0673** |
| `hotpotqa` | multi_doc_qa | 20 | `em` / `f1` | **0.0** / **0.0034** |
| `trec` | few_shot | 20 | `accuracy` | **0.0** |

单条推理耗时约 1.15–1.2 s（20 条/任务，`max_length: 4096`）。

---

## 4. 本轮暴露的两个问题（**均未修改代码**，按约定只报告）

### 4.1 `datasets` 4.x 问题已解决；新暴露 tokenizer 调用 bug（已由远端修复）

第一次跑 LongBench（17:47:46）时，三个任务**瞬间全失败**：

```
[ERROR] run_longbench: [multifieldqa_en] 失败: ones_like(): argument 'input' (position 1) must be Tensor, not BatchEncoding
[ERROR] run_longbench: [hotpotqa]         失败: ones_like(): argument 'input' (position 1) must be Tensor, not BatchEncoding
[ERROR] run_longbench: [trec]             失败: ones_like(): argument 'input' (position 1) must be Tensor, not BatchEncoding
```

- **数据侧已 OK**：日志显示 `冒烟截断: multifieldqa_en -> 20 条` 等三行都在报错**之前**成功打印，
  说明本地 JSONL 读取正常——**`datasets` 4.0.0 的本地化修复有效**。
- **失败点在推理侧**：tokenizer 返回的 `BatchEncoding` 被直接喂给了 `torch.ones_like`，
  在 `transformers` 5.6.0 下不再隐式解包（老版本可容忍）。属 `src/longbench.py` 的 API 兼容问题。
- 18:01:37 的第二次运行（`HF_HOME=/data/yucheng/hf_cache`）三任务全部跑通，**该问题已由远端修复**。

### 4.2 ⚠️ `result.json` 丢失主框架 9 组指标（**落盘缺陷，待确认是否修**）

`result.json` 最终内容**只有 `long_context` 一个维度**，主框架的 knowledge / reasoning /
instruction / multilingual 全部不见了。

根因在 [src/result_manager.py:84-103](src/result_manager.py#L84-L103)：

```python
merged = dict(existing)
merged["_updated"] = utc_now_iso()
merged["version"] = cfg.version
merged["model"] = cfg.model.name
if isinstance(metrics, dict):
    merged["metrics"] = metrics          # ← 整体替换，不是按维度合并
```

- 17:47:45 `run_lm_eval.py` 写入含 9 组指标的 `metrics`；
- 17:47:50 `run_longbench.py` 调用同一函数，`metrics` 被**整体覆盖**成只含 `long_context` 的 dict；
- 18:03:06 第二次 longbench 再覆盖一次，最终只剩长上下文。

**影响**：紧随其后生成的 `report.md` 的 Results 章节**只有 long_context**，主框架 9 组分未进报告；
registry 与目录结构正常，只是 `result.json` 的指标不完整。

**可恢复性**：9 组指标完整保留在 `results/qwen35-4b-smoke-v0.0/logs/run.log` 的
`[INFO] run_lm_eval: 评测结果: {...}` 一行里（本文 §2 即由此还原），**数据没丢**。

修复方向（一行级）：把 `merged["metrics"] = metrics` 改为按 `metrics["dimensions"]` 逐维度
合并进 `merged.setdefault("metrics", {}).setdefault("dimensions", {})`。**我没有改动任何代码**，
需要的话请明确授权。

---

## 5. 另一处已落盘的旧结果（`qwen35-4b-base-v0.0`）

这是本轮之前的一次**单任务**尝试，与本次冒烟链不是同一批，勿混用：

| 任务 | 样本数 | 指标 | 分数 | stderr |
| --- | --- | --- | --- | --- |
| `gsm8k` | 20 | `exact_match`（strict） | **0.2500** | 0.0993 |
| | | `exact_match`（flexible） | **0.1000** | 0.0688 |
| `long_context` | — | status | `skipped`（未配置） | — |

样本明细：`results/qwen35-4b-base-v0.0/samples/gsm8k.json`（120 KB）。

---

## 6. 跑全量 baseline 前的必改项（**尚未改，仅记录**）

`configs/baseline_qwen35_4b.yaml` 仍与实测环境不一致：

| 项 | 当前值 | 应改为 | 原因 |
| --- | --- | --- | --- |
| `evaluation.batch_size` | `16` | **`8`** | 16 在 multilingual 维度 OOM |
| `evaluation.longbench_data_dir` | `null` | **`data/longbench`** | datasets 4.x 已移除脚本加载器 |

另两处耗时/口径提示：

- `aime24` 自带 `max_gen_toks=32768`，优先级高于配置的 `max_new_tokens`，
  本轮 20 条实测约 184 s/条（8 条耗时 24 m 35 s），是全链最大耗时段；
  全量 AIME 需预留大量时间。
- `max_samples: 20` 是**按叶任务**截断，故 `mmlu_redux_generative` 与 `mmmlu_zh_cn`
  各为 57 × 20 = 1140 条，而非 20 条。

---

## 7. 免责声明

本轮 `max_samples: 20`，除两个 1140 条的 MMLU 类任务外，其余任务样本量仅 20，
**置信区间极宽（多数 stderr ≈ 0.05–0.07）**，且 `aime24` / `trec` / `mgsm` / `multifieldqa`
等出现多处 0 分，主要是样本量与解码参数（`max_new_tokens` 被任务默认值压制）所致，
**不能据此对模型能力下任何结论**。正式分数须以 `baseline_qwen35_4b.yaml` 全量重跑为准。
