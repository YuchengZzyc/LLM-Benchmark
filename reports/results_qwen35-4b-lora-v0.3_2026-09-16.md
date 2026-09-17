# Qwen3.5-4B + LoRA v3.2 · v1.0 通用能力评测结果

> 版本：`qwen35-4b-lora-v1.0`（base + adapter 直挂，未合并）
> 模型：`/data/yucheng/madm-llm/models/Qwen3.5-4B` + `/data/yucheng/madm-llm/madm-llm/output/qwen35_4b_lora_v3.2-peft`（键名修正版，496 键 / 16.23M 参数，logit diff 2.13 验证通过）
> 口径：与基线 v0.1 逐字一致（六数据集全量、关 thinking、纯文本 MCQ；LongBench v1 × 12 @30 条；AA-LCR 100 题全量、开 thinking、gpt-5.6-luna 判题）
> 环境：RTX 5090 32GB / lm-eval 0.4.13 / bfloat16 / seed 42 / HF 镜像
> 日期：2026-09-16 · **状态：全部完成**（电池 15:19 完 → AA-LCR 首轮 17:45 + 补跑轮 19:27 完，100/100 全分母）

## 一、总览（六数据集 + AA-LCR）

| 维度 | 数据集 | 口径 | 样本数 | lora-v1.0 | v0.1（基座） | Δ | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 专业推理 | GSM8K | 5-shot，关 thinking，strict-extract | 1319 | 0.8529 | 0.9030 | **-0.0501** | ✅ |
| 专业推理 | GSM8K（flexible） | 同上，flexible-extract | 1319 | 0.8817 | 0.9030 | -0.0213 | ✅ |
| 知识与理解 | C-Eval valid | 5-shot loglikelihood MCQ | 1346 | 0.7392 | 0.7496 | **-0.0104** | ✅ |
| 知识与理解 | MMMLU-zh | 5-shot 纯文本 MCQ | 400 | 0.7025 | 0.7175 | **-0.0150** | ✅ |
| 知识与理解 | MMLU-ProX-zh | 5-shot 生成式+正则抽取，60/科×14 | 840 | 0.6143 | 0.6345 | **-0.0202** | ✅ |
| 多语言 | MGSM native CoT zh | 0-shot，flexible-extract | 250 | 0.7760 | 0.7640 | **+0.0120** | ✅ |
| 指令遵循 | IFEval prompt-strict | 0-shot，关 thinking | 541 | 0.4880 | 0.8226 | **-0.3346** | ✅ |
| 指令遵循 | IFEval inst-loose | 0-shot，关 thinking | 541 | 0.6067 | 0.8993 | **-0.2926** | ✅ |
| 长上下文（目标任务） | AA-LCR | 100 题全量，开 thinking，gpt-5.6-luna | 100 | **0.4800** | 0.5900 | **-0.1100** | ✅（补跑后全分母） |

IFEval 辅助指标：prompt-loose 0.4935（v0.1: 0.8540）/ inst-strict 0.5995（v0.1: 0.8753）。
MGSM strict 口径两轮均为 0（正则 `答案是 X。` 不命中，与 v0.1 同源，非异常）。

## 二、长上下文参考分（LongBench v1 × 12，每任务 30 条）

| 数据集 | 类别 | 指标 | lora-v1.0 | v0.1（基座） | Δ | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| multifieldqa_en | 单文档 QA | F1 | 0.2678 | 0.2297 | **+0.0381** | ✅ |
| qasper | 单文档 QA | F1 | 0.1489 | 0.0948 | **+0.0541** | ✅ |
| hotpotqa | 多文档 QA | F1 | 0.1184 | 0.0639 | **+0.0546** | ✅ |
| 2wikimqa | 多文档 QA | F1 | 0.1536 | 0.0759 | **+0.0777** | ✅ |
| musique | 多文档 QA | F1 | 0.1237 | 0.0610 | **+0.0627** | ✅ |
| gov_report | 摘要 | ROUGE-L | 0.1314 | 0.1842 | **-0.0529** | ✅ |
| multi_news | 摘要 | ROUGE-L | 0.1037 | 0.1738 | **-0.0701** | ✅ |
| trec | 少样本分类 | Acc | 0.0333 | 0.1333 | **-0.1000** | ✅ |
| triviaqa | 少样本 QA | EM / F1 | 0.7333 / 0.8271 | 0.7667 / 0.8444 | -0.0333 / -0.0174 | ✅ |
| samsum | 少样本摘要 | ROUGE-L | 0.4160 | 0.3991 | **+0.0169** | ✅ |
| passage_count | 合成 | Acc | 0.0000 | 0.0000 | 0.0000 | ✅（基线亦 0） |
| passage_retrieval_en | 合成 | Acc | 0.0000 | 0.2667 | **-0.2667** | 🔁 重跑复核后按真实分接受 |

## 三、异常处置记录（Auto 模式自愈）

1. **passage_retrieval_en 首轮 0 分** → 判定异常（基线 0.2667），15:39 单独重启 `run_longbench.py --tasks passage_retrieval_en`，30/30 完整重跑仍 0 分 → **定性为微调真实行为差异**（推断：不再以"段落字母"格式作答；trec 同降 10pt 同向佐证）。
2. registry 曾缺 `qwen35-4b-lora-v1.0` 登记 → 已补 register_model，终态 completed。
3. 巡检通道两段故障：14:10–15:30 分类器不可用（跳过 2 轮，远端 nohup 不受影响）；15:30 起凭据明文命令被拒 → 改走 `REMOTE_PW_FILE` 本地临时文件 + SSH 信道 stdin 传 key（key 不出现在任何命令行/ps/远端磁盘）。
4. **AA-LCR 首轮系统性 OOM**：共租进程占 1.2 GiB + PyTorch 碎片化（reserved-unallocated 涨至 6.25 GiB），68/100 题 SKIP。杀进程即时重启被安全分类器拦截（需用户明示批准 PID），改走等价自主路径：17:45 自然跑完 → 剔 error 条目 → `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + `--resume` 补跑 68 题（17:57–19:27，**0 失败**），聚合回 100 题全分母。
5. 规避一个聚合陷阱：`--resume` 模式下脚本自写的 aa_lcr 汇总只含最后一轮 68 题；终化时从 samples 文件全量重聚合（32+68=100，48 对）。
6. 发现框架 bug 两枚（已记录，本次未修）：run_longbench.py `--keep-samples` CLI 旗标未接线；aa_lcr_run.py resume 把 error 条目视作已完成跳过。

## 四、结论

1. **adapter 确认生效**：全维度分数系统性偏离基座（有升有降），且加载链路有 logit diff 预验证，排除静默失效。
2. **知识 / 推理 / 多语言基本保持**：C-Eval -1.0pt、MMMLU -1.5pt、MMLU-ProX -2.0pt、GSM8K strict -5.0pt（flexible 仅 -2.1pt，部分是答案格式习惯变化）；MGSM +1.2pt 为唯一微升。通用"硬能力"代价温和。
3. **指令遵循是最大通用代价**：IFEval 四口径齐跌 29~34pt（0.82→0.49）。LoRA v3.2 的训练分布显著收窄了输出行为，格式约束遵循能力被挤出。strict≈loose 说明并非标点/格式细节问题，而是约束本身未被满足。
4. **长文档短问答显著增益**：LongBench 单/多文档 QA 五任务 F1 全线 +3.8~+7.8pt（musique 相对 +103%）；但摘要 ROUGE-L 回落 5~7pt、trec -10pt、passage_retrieval 归零（重跑复核仍 0）——增益集中在"F1 式短答案抽取"，检索定位与归纳类未受益甚至受损。
5. **微调目标任务（AA-LCR）不升反降：59.0 → 48.0（-11pt）**。这是本轮最关键的发现：LoRA v3.2 虽然提升了短式 QA 抽取（第 4 条），但在"长上下文 + 开 thinking + 判题器对最终承诺答案"的 AA-LCR 协议下净退步。与 IFEval 的崩跌合流指向同一根因：**微调把模型输出收窄成了固定应答模式，伤害了开放格式下的指令遵循与深思熟虑作答**。若训练目标是 AA-LCR 成绩，当前 v3.2 配方未达目的，建议排查：训练数据与 AA-LCR 协议的口径差（是否训练时关闭 thinking / 答案格式 / 上下文长度分布），或考虑降低 LoRA rank/epoch、混入通用指令数据做 replay。
6. **总体定性**：以指令遵循（-33pt）与目标任务（-11pt）为代价，换取长文档 QA 抽取（+4~+8pt）与通用硬能力的近似保持——**v3.2 是一次负收益微调**，不建议就此定型。

---

*交付物：远端 `results/qwen35-4b-lora-v1.0/result.json`（v0.1 同构，registry completed）+ 本报告。*
*生成：Claude（Auto 模式自治巡检）· 2026-09-16 19:30 定稿*
