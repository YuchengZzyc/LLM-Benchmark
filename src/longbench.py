r"""LongBench 长上下文真实评测适配器（独立于 lm-eval）。

设计原则
--------
- 长上下文**不绑定**主框架（调研报告：长上下文与长期记忆分开测）。
- 本模块直接加载 LongBench 官方数据集（``THUDM/LongBench``），按官方
  评分口径计算指标（问答 F1 / 摘要 ROUGE-L / 合成与检索 accuracy），
  不引入 lm-eval。
- 运行哪些子任务由配置 ``tasks.long_context`` 决定（禁止在代码中硬编码
  benchmark 清单）；任务名 -> 指标类型的元数据是框架知识，见 :data:`TASK_META`。
- 代码类子任务（lcc / repobench-p）需要 LongBench 官方 code eval
  （单元测试执行），当前版本标记为 ``not_supported``，不生成虚假分数。

依赖（服务器阶段安装）：transformers / torch / datasets / rouge-score。
缺失时抛出 :class:`NotConfiguredError`，与主框架的未安装提示一致。

>>> from src.config_loader import load_config
>>> cfg = load_config('configs/template.yaml')
>>> # 需 GPU 与真实模型，仅演示调用方式
>>> run_longbench(cfg)  # doctest: +SKIP
"""
from __future__ import annotations

import json
import logging
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Optional

try:
    from src.config_loader import EvalConfig
    from src.result_manager import save_sample_payload
except ImportError:  # 兼容直接以脚本运行
    from config_loader import EvalConfig
    from result_manager import save_sample_payload

try:
    from src.evaluator import NotConfiguredError
except ImportError:  # pragma: no cover - 独立运行兜底
    class NotConfiguredError(RuntimeError):
        """依赖未安装。"""


# =============================================================================
# 任务元数据（任务名 -> 指标类型 / 生成长度 / 类别）
# 说明：LongBench 官方 pred.py 按任务指定 max_new_tokens；此处对齐官方取值。
# =============================================================================
TASK_META: dict[str, dict[str, Any]] = {
    # 单文档问答（F1 / EM）
    "qasper":          {"metric": "f1",   "max_new_tokens": 768, "category": "single_doc_qa"},
    "multifieldqa_en": {"metric": "f1",   "max_new_tokens": 64,  "category": "single_doc_qa"},
    "multifieldqa_zh": {"metric": "f1",   "max_new_tokens": 64,  "category": "single_doc_qa"},
    # 多文档问答（F1 / EM）
    "hotpotqa":        {"metric": "f1",   "max_new_tokens": 128, "category": "multi_doc_qa"},
    "2wikimqa":        {"metric": "f1",   "max_new_tokens": 128, "category": "multi_doc_qa"},
    "musique":         {"metric": "f1",   "max_new_tokens": 128, "category": "multi_doc_qa"},
    # 摘要（ROUGE-L）
    "gov_report":      {"metric": "rouge", "max_new_tokens": 512, "category": "summarization"},
    "qmsum":           {"metric": "rouge", "max_new_tokens": 512, "category": "summarization"},
    "multi_news":      {"metric": "rouge", "max_new_tokens": 512, "category": "summarization"},
    "vcsum":           {"metric": "rouge", "max_new_tokens": 512, "category": "summarization"},
    # few-shot 少量示例
    "trec":            {"metric": "accuracy", "max_new_tokens": 64,  "category": "few_shot"},
    "triviaqa":        {"metric": "f1",        "max_new_tokens": 128, "category": "few_shot"},
    "samsum":          {"metric": "rouge",     "max_new_tokens": 128, "category": "few_shot"},
    # 合成任务
    "passage_count":        {"metric": "retrieval_count", "max_new_tokens": 64, "category": "synthetic"},
    "passage_retrieval_en": {"metric": "retrieval",       "max_new_tokens": 64, "category": "synthetic"},
    "passage_retrieval_zh": {"metric": "retrieval",       "max_new_tokens": 64, "category": "synthetic"},
    # 代码补全（需 LongBench 官方 code eval 执行单元测试，当前未接入）
    "lcc":             {"metric": "code", "max_new_tokens": 64, "category": "code"},
    "repobench-p":     {"metric": "code", "max_new_tokens": 64, "category": "code"},
}

_SUPPORTED_METRICS = ("f1", "rouge", "accuracy", "retrieval", "retrieval_count")


def supported_tasks() -> list[str]:
    """返回支持真实评分的任务名列表（不含代码类）。"""
    return [t for t, meta in TASK_META.items() if meta["metric"] in _SUPPORTED_METRICS]


# =============================================================================
# 数据加载
# =============================================================================

def _load_task_data(cfg: EvalConfig, task: str) -> list[dict[str, Any]]:
    """加载某个 LongBench 子任务的测试数据。

    - ``evaluation.longbench_data_dir`` 指定本地目录时，从
      ``<dir>/<task>.jsonl`` 或 ``<dir>/<task>/test.jsonl`` 读取；
    - 否则从 HuggingFace ``THUDM/LongBench`` 下载（服务器需可访问 HF）。
    """
    data_dir = cfg.evaluation.longbench_data_dir
    if data_dir:
        base = Path(data_dir)
        candidates = [
            base / f"{task}.jsonl", base / task / "test.jsonl",
            base / f"{task}.json", base / task / "test.json",
        ]
        for cand in candidates:
            if cand.exists():
                if cand.suffix == ".jsonl":
                    rows = [json.loads(line) for line in
                            cand.read_text(encoding="utf-8").splitlines() if line.strip()]
                else:
                    rows = json.loads(cand.read_text(encoding="utf-8"))
                return [r for r in rows if isinstance(r, dict)]
        raise FileNotFoundError(
            f"本地 LongBench 数据目录中未找到任务 {task!r}（已检查 {base}）")
    try:
        from datasets import load_dataset
    except Exception as exc:  # noqa: BLE001
        raise NotConfiguredError(
            "未安装 datasets；请先在服务器运行: bash scripts/install_env.sh"
        ) from exc
    ds = load_dataset("THUDM/LongBench", task)
    split = "test" if "test" in ds else next(iter(ds))
    return list(ds[split])


# =============================================================================
# 指标计算（对齐 THUDM/LongBench 官方 evaluate.py 口径）
# =============================================================================

def _normalize_answer(text: str) -> str:
    """规范化答案：小写、去除冠词、标点、压缩空白。"""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> list[str]:
    return _normalize_answer(text).split()


def _token_f1(pred_tokens: list[str], gold_tokens: list[str]) -> float:
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num = sum(common.values())
    if num == 0:
        return 0.0
    prec = num / len(pred_tokens)
    rec = num / len(gold_tokens)
    return 2.0 * prec * rec / (prec + rec)


def _best_f1(pred: str, answers: list[str]) -> float:
    pt = _tokenize(pred)
    return max((_token_f1(pt, _tokenize(a)) for a in answers), default=0.0)


def _best_em(pred: str, answers: list[str]) -> float:
    pn = _normalize_answer(pred)
    return 1.0 if any(pn == _normalize_answer(a) for a in answers) else 0.0


def _score_f1(pred: str, answers: list[str]) -> dict[str, float]:
    return {"f1": round(_best_f1(pred, answers), 6),
            "em": round(_best_em(pred, answers), 6)}


_rouge_scorer: Any = None


def _get_rouge_scorer():
    global _rouge_scorer
    if _rouge_scorer is None:
        try:
            from rouge_score import rouge_scorer
        except Exception as exc:  # noqa: BLE001
            raise NotConfiguredError(
                "未安装 rouge-score；请先在服务器运行: bash scripts/install_env.sh"
            ) from exc
        _rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _rouge_scorer


def _score_rouge(pred: str, answers: list[str]) -> dict[str, float]:
    scorer = _get_rouge_scorer()
    best = max((scorer.score(a, pred)["rougeL"].fmeasure for a in answers),
               default=0.0)
    return {"rougeL": round(best, 6)}


def _score_accuracy(pred: str, answers: list[str]) -> dict[str, float]:
    pn = _normalize_answer(pred)
    return {"accuracy": 1.0 if any(pn == _normalize_answer(a) for a in answers)
            else 0.0}


def _first_int(text: str) -> Optional[int]:
    m = re.search(r"-?\d+", text or "")
    return int(m.group()) if m else None


def _score_retrieval_count(pred: str, answers: list[str]) -> dict[str, float]:
    gold = _first_int(answers[0]) if answers else None
    pred_n = _first_int(pred)
    return {"accuracy": 1.0 if (gold is not None and pred_n == gold) else 0.0}


def _score_retrieval(pred: str, answers: list[str]) -> dict[str, float]:
    """检索任务：预测中是否包含目标段落文本。"""
    return {"accuracy": 1.0 if any(a and a in pred for a in answers) else 0.0}


_SCORERS = {
    "f1": _score_f1,
    "rouge": _score_rouge,
    "accuracy": _score_accuracy,
    "retrieval": _score_retrieval,
    "retrieval_count": _score_retrieval_count,
}


def _score(metric: str, pred: str, answers: list[str]) -> dict[str, Any]:
    fn = _SCORERS.get(metric)
    if fn is None:
        return {"status": "error", "message": f"未支持的指标类型: {metric!r}"}
    return fn(pred, answers)


def _aggregate(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """对单样本分数求平均，得到任务级指标。"""
    keys = sorted({k for s in scores for k in s
                   if isinstance(s.get(k), (int, float)) and not isinstance(s.get(k), bool)})
    out: dict[str, Any] = {}
    for k in keys:
        vals = [s[k] for s in scores if isinstance(s.get(k), (int, float))]
        out[k] = round(sum(vals) / len(vals), 6) if vals else 0.0
    return out


# =============================================================================
# 模型加载与推理
# =============================================================================

def _load_model_and_tokenizer(cfg: EvalConfig):
    """按配置加载 HuggingFace 模型与 tokenizer（LongBench 独立接口自用）。"""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        raise NotConfiguredError(
            "未安装 torch/transformers；请先在服务器运行: bash scripts/install_env.sh"
        ) from exc

    dtype = str(cfg.model.dtype).lower()
    load_kwargs: dict[str, Any] = {}
    if dtype in ("int8", "int4"):
        try:
            import bitsandbytes  # noqa: F401  # 量化加载依赖
        except Exception as exc:  # noqa: BLE001
            raise NotConfiguredError(
                f"model.dtype={dtype} 需要 bitsandbytes；请先在服务器安装") from exc
        load_kwargs["load_in_8bit"] = (dtype == "int8")
        load_kwargs["load_in_4bit"] = (dtype == "int4")
    else:
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": "auto",
        }
        load_kwargs["torch_dtype"] = torch_dtype.get(dtype, "auto")
    if cfg.model.trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    if cfg.model.revision:
        load_kwargs["revision"] = cfg.model.revision

    tokenizer_path = cfg.model.tokenizer_path or cfg.model.path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=bool(cfg.model.trust_remote_code),
        revision=cfg.model.revision or None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model.path, **load_kwargs)
    if cfg.model.adapter_path:
        # base+adapter 直挂：base 加载后挂载 LoRA adapter（与主框架 lm-eval
        # 的 peft 参数同一加载方式）；tokenizer 仍从 base/tokenizer_path 加载
        try:
            from peft import PeftModel
        except Exception as exc:  # noqa: BLE001
            raise NotConfiguredError(
                "未安装 peft；请在服务器运行: /data/yucheng/.local/bin/uv pip "
                "install --python /data/yucheng/madm-llm/.venv/bin/python peft"
            ) from exc
        model = PeftModel.from_pretrained(model, cfg.model.adapter_path)
    device = str(cfg.evaluation.device or "").lower()
    if device in ("cuda", "cpu"):
        model = model.to(device)
    model.eval()  # 确保 LoRA dropout 关闭（from_pretrained 默认 eval，双保险）
    torch.manual_seed(cfg.evaluation.seed)
    return model, tokenizer


def _build_prompt(sample: dict[str, Any]) -> str:
    ctx = str(sample.get("context", "") or "").strip()
    inp = str(sample.get("input", "") or "").strip()
    return (ctx + "\n" + inp) if ctx else inp


def _truncate_middle(text: str, tail_text: str, tokenizer, max_length: int) -> str:
    """中间截断：保留 prompt 头部 + 末尾问题（tail_text），对齐 LongBench 官方 pred.py 协议。

    HF 默认 ``truncation=True`` 从尾部截断，会直接切掉末尾的问题文本；
    长上下文任务必须按 head+tail 方式截断，否则 question 丢失、全部判错。
    """
    import torch
    tokens = tokenizer(text, truncation=False)["input_ids"]
    if len(tokens) <= max_length:
        return text
    tail = tokenizer(tail_text, add_special_tokens=False)["input_ids"]
    keep_tail = min(len(tail) + 8, max_length // 2)   # 尾部问题 + 分隔符余量
    head = tokens[: max_length - keep_tail]
    merged = head + tokens[len(tokens) - keep_tail:]
    return tokenizer.decode(merged, skip_special_tokens=True)


def _generate(model, tokenizer, cfg: EvalConfig, prompt: str, max_new: int,
              tail_text: str = "") -> str:
    """对单条 prompt 生成回复；支持 chat template、中间截断与停止条件。

    修复记录（2026-09-11 v0.1 LongBench 首跑 11/12 失败后）：
    - 上一版函数体在 ``enc`` 之后被截断，返回 None —— 补齐 generate/return；
    - HF ``model.generate`` 没有默认停止条件，Qwen3.5-4B 在纯文本下会无限
      复读输入问题（``\\nWho is the current manager of Urartu?`` ×10+），
      必须显式传 ``stop_strings``；
    - chat 分支原先在 ``prompt``（未截断）上套模板，tail_text 截断结果被丢弃
      —— 现在统一先截断再套模板。

    二次修复（2026-09-11 诊断后）：
    - 停止条件按分支分口径：纯文本分支用 ``eos_token_id``（天然生成到 EOS），
      chat 分支用 ``stop_strings=["<|im_end|>", "</think>"]``（关 thinking 的
      完整输出到 ``</think>``，不会被 ``<|im_end|>`` 中途截断）。
    """
    import torch

    gen_kw = dict(cfg.evaluation.gen_kwargs or {})
    temperature = float(gen_kw.pop("temperature", 0.0))
    top_p = float(gen_kw.pop("top_p", 1.0))
    # 剔除 lm-eval 特有键：生成长度以 TASK_META 官方口径为准，
    # 其余（until/max_gen_toks 等）不是 HF generate 的合法参数，混入会 TypeError
    for k in ("max_new_tokens", "max_gen_toks", "until", "do_sample"):
        gen_kw.pop(k, None)
    gen_kw.setdefault("pad_token_id", tokenizer.eos_token_id)
    gen_kw["eos_token_id"] = tokenizer.eos_token_id  # 纯文本分支：天然停到 EOS

    # 中间截断（head + 末尾问题）在套模板**之前**做 —— 问题文本绝不能被截断
    if tail_text:
        text = _truncate_middle(prompt, tail_text, tokenizer,
                                int(cfg.evaluation.max_length))
    else:
        text = prompt

    if cfg.evaluation.apply_chat_template:
        # 对齐 LongBench 官方 pred.py：Qwen 系 chat 模型走模板 + 系统提示词。
        # enable_thinking=False 与主框架 lm-eval 口径一致（模板渲染出
        # <think>\\n\\n</think>\\n\\n 空块，模型直接生成答案，不写思考前导）。
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text},
        ]
        try:
            batch = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", enable_thinking=False, truncation=False)
        except TypeError:  # 模板不支持 enable_thinking 时降级
            batch = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", truncation=False)
        enc = {k: v.to(model.device) for k, v in batch.items()}
        # chat 分支：模型以 assistant 角色结束，天然停在 <|im_end|>。
        # 不设 </think> 停止串 —— 若模型自产 think 块，靠下方 decode 后剥除前缀。
        gen_kw["stop_strings"] = ["<|im_end|>"]
        # chat 分支 stop_strings 要生效，必须带 tokenizer 进 generate
        stop_kw = {"tokenizer": tokenizer}
    else:
        enc = tokenizer(text, truncation=False, return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        stop_kw = {}

    gen_kw["do_sample"] = temperature > 0
    if temperature > 0:
        gen_kw["temperature"] = temperature
        gen_kw["top_p"] = top_p

    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, **gen_kw, **stop_kw)
    new_tokens = out[0, enc["input_ids"].shape[1]:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # chat 分支兜底：若模型仍自产 <think> 思考前导（enable_thinking=False 下
    # 个别 sample 仍可能输出），把 <think>...</think> 整块剥掉，只留正式答案。
    if cfg.evaluation.apply_chat_template and "<think" in decoded:
        m = re.search(r"</think>\s*", decoded)
        if m:
            decoded = decoded[m.end():]
    return (decoded or "").strip()


# =============================================================================
# 任务执行
# =============================================================================

def _run_task(model, tokenizer, cfg: EvalConfig, task: str,
              logger: logging.Logger) -> dict[str, Any]:
    meta = TASK_META.get(task)
    if meta is None:
        return {"status": "error",
                "message": f"未知 LongBench 任务: {task!r}（可用任务见 src/longbench.py TASK_META）"}
    if meta["metric"] not in _SUPPORTED_METRICS:
        return {
            "status": "not_supported",
            "message": "代码类任务需要 LongBench 官方 code eval（单元测试执行），"
                       "当前版本未接入；建议后续用 THUDM/LongBench 官方 evaluate.py 单独执行",
        }

    samples = _load_task_data(cfg, task)
    cap = cfg.output.max_samples
    if cap and cap > 0:
        samples = samples[:cap]

    if not samples:
        return {"status": "error", "message": "任务数据为空（请检查网络或 longbench_data_dir）"}

    user_max_new = (cfg.evaluation.gen_kwargs or {}).get("max_new_tokens")
    max_new = int(user_max_new) if user_max_new else meta["max_new_tokens"]
    max_new = min(max_new, int(meta["max_new_tokens"]))

    scores: list[dict[str, Any]] = []
    payload: list[dict[str, Any]] = []
    for i, sample in enumerate(samples, start=1):
        prompt = _build_prompt(sample)
        # 中间截断时尾部保留的是"问题文本"（input 字段），
        # 长上下文任务的问题绝不能随尾部截断被切掉
        tail_text = str(sample.get("input", "") or "").strip()
        pred = _generate(model, tokenizer, cfg, prompt, max_new, tail_text=tail_text)
        answers = sample.get("answers") or []
        if isinstance(answers, str):
            answers = [answers]
        answers = [a for a in answers if a]
        score = _score(meta["metric"], pred, answers)
        scores.append(score)
        payload.append({"index": i - 1, "prediction": pred,
                        "answers": answers, "length": sample.get("length")})
        logger.info("[%s] %d/%d 完成", task, i, len(samples))

    agg = _aggregate(scores)
    agg["samples"] = len(samples)
    agg["category"] = meta["category"]

    if cfg.output.keep_samples:
        try:
            save_sample_payload(cfg, f"longbench_{task}", payload)
        except Exception as exc:  # 样本保存失败不影响主结果
            agg["_sample_error"] = str(exc)
    return agg


# =============================================================================
# 对外入口
# =============================================================================

def run_longbench(cfg: EvalConfig, tasks: Optional[list[str]] = None,
                  logger: Optional[logging.Logger] = None) -> dict[str, Any]:
    """执行 LongBench 评测，返回 ``{"dimensions": {"long_context": {task: 指标}}}``。

    - 加载模型一次，按任务逐个推理与评分；
    - 未配置任务时返回 skipped，不报错；
    - 单个任务失败不影响其他任务（结果中记录 error）。
    """
    tasks = list(tasks if tasks is not None else cfg.tasks.get("long_context", []))
    logger = logger or logging.getLogger("longbench")
    # 说明：max_samples 限量在 _run_task 内做**内存截断**（samples[:cap]），
    # 不改写 data/longbench/*.jsonl —— 就地截断会毁掉本地完整数据
    #（2026-09-10 夜间全量前已移除该逻辑）。
    if not tasks:
        return {"dimensions": {"long_context": {
            "status": "skipped", "message": "未配置 long_context 任务"}}}

    model, tokenizer = _load_model_and_tokenizer(cfg)
    logger.info("模型加载完成: %s", cfg.model.path)

    dim: dict[str, Any] = {}
    for task in tasks:
        try:
            logger.info("==> 开始任务: %s", task)
            dim[task] = _run_task(model, tokenizer, cfg, task, logger)
        except NotConfiguredError:
            raise
        except Exception as exc:  # noqa: BLE001 - 单任务失败不阻断其余任务
            dim[task] = {"status": "error", "error": str(exc)}
            logger.error("[%s] 失败: %s", task, exc)
            logger.debug("任务失败完整堆栈", exc_info=True)
    return {"dimensions": {"long_context": dim}}


if __name__ == "__main__":
    import doctest
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    doctest.testmod()
    print(f"longbench OK（支持 {len(supported_tasks())} 个真实评分任务，"
          f"代码类 {len(TASK_META) - len(supported_tasks())} 个未接入）")
