#!/usr/bin/env python3
"""aa_lcr_run.py — AA-LCR 长上下文评测入口（Qwen3.5-4B v0.1 基线）。

复现官方 AA-LCR 57.0 的协议（100 题全量、官方 prompt 模板、
gpt-5.6-luna 判题器）。单进程串行，单卡 RTX 5090 32GB。

用法::

    python scripts/aa_lcr_run.py --config configs/aa_lcr_qwen35_4b_v01.yaml
    python scripts/aa_lcr_run.py --config configs/aa_lcr_qwen35_4b_v01.yaml --limit 5   # 冒烟
    python scripts/aa_lcr_run.py --config configs/aa_lcr_qwen35_4b_v01.yaml --resume

流程：加载配置 -> 读 CSV + 文档 -> 官方模板组 prompt -> 逐题生成
（HF generate，fla 内核）-> gpt-5.6-luna 判题 -> 聚合写 result.json。
每题一个 worker 崩溃不丢已跑题（增量落盘 samples/）；--resume 续跑。
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import load_config  # noqa: E402
from src.result_manager import (  # noqa: E402
    create_run_dir, save_config_snapshot, save_result_json, update_status,
)
from src.utils import setup_logger  # noqa: E402

try:
    from src.config_loader import EvalConfig
except ImportError:  # 兼容直接以脚本运行
    from config_loader import EvalConfig

# =============================================================================
# 官方 AA-LCR prompt 模板（README 原样）
# =============================================================================

SYSTEM_PROMPT = "You are a helpful assistant."


def build_documents_text(docs: list[str]) -> str:
    return "\n\n".join(
        f"BEGIN DOCUMENT {i + 1}:\n{doc}\nEND DOCUMENT {i + 1}"
        for i, doc in enumerate(docs)
    )


def build_input_prompt(docs: list[str], question: str) -> str:
    documents_text = build_documents_text(docs)
    return (
        "BEGIN INPUT DOCUMENTS\n\n"
        f"{documents_text}\n\n"
        "END INPUT DOCUMENTS\n\n"
        "Answer the following question using the input documents provided above.\n\n"
        "START QUESTION\n\n"
        f"{question}\n\n"
        "END QUESTION"
    )


# =============================================================================
# 官方 gpt-5.6-luna 判题器模板
# =============================================================================

JUDGE_SYSTEM_PROMPT = """Decide whether the CANDIDATE ANSWER is correct or incorrect against the OFFICIAL ANSWER.
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
  Hedging is fine as long as one clearly definitive answer is given."""


def build_judge_user_prompt(question: str, official_answer: str,
                            candidate_answer: str) -> str:
    return (
        "Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.\n"
        "For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.\n"
        "\n"
        "The question, for reference only: START QUESTION "
        f"{question}\n\n"
        "END QUESTION\n"
        "\n"
        "The OFFICIAL ANSWER: "
        f"{official_answer}\n\n"
        "END OFFICIAL ANSWER\n"
        "\n"
        "BEGIN CANDIDATE ANSWER TO ASSESS\n"
        "\n"
        f"{candidate_answer}\n"
        "\n"
        "END CANDIDATE ANSWER TO ASSESS\n"
        "\n"
        "Reply as JSON, with a verdict of CORRECT or INCORRECT."
    )


# =============================================================================
# 数据加载
# =============================================================================

def load_rows(csv_path: str) -> list[dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def load_docs(row: dict[str, str], docs_root: str) -> list[str]:
    """按 CSV ``data_source_filenames``（分号有序）加载文档，乱序会改变官方条件。

    macOS 重解压会把文件名的复合字符拆成 NFD（如 ``Başev`` -> ``Başev``），
    CSV 里是 NFC。比较/拼接前做 NFC 归一化，避免同名文件找不到。
    """
    filenames = [s.strip() for s in row["data_source_filenames"].split(";") if s.strip()]
    category, set_id = row["document_category"], row["document_set_id"]
    docs: list[str] = []
    for fn in filenames:
        p = os.path.join(docs_root, category, set_id, fn)
        if not os.path.exists(p):
            p_nfc = os.path.join(docs_root, category, set_id,
                                 unicodedata.normalize("NFC", fn))
            if os.path.exists(p_nfc):
                p = p_nfc
            else:
                # 兜底：目录内按 NFC 匹配一次（覆盖 macOS 重解压的 NFD 变体）
                d = os.path.dirname(p)
                if os.path.isdir(d):
                    match = [n for n in os.listdir(d)
                             if unicodedata.normalize("NFC", n) == fn]
                    if match:
                        p = os.path.join(d, match[0])
        if not os.path.exists(p):
            raise FileNotFoundError(f"缺少文档: {p}")
        with open(p, encoding="utf-8") as f:
            docs.append(f.read())
    return docs


# =============================================================================
# 生成
# =============================================================================

def load_model_and_tokenizer(model_path: str, dtype: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}.get(dtype, torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype).to("cuda")
    return model, tok


def generate_answer(model, tok, docs: list[str], question: str,
                    max_new: int = 1024, enable_thinking: bool = False) -> str:
    prompt = build_input_prompt(docs, question)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    import torch

    batch = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", enable_thinking=enable_thinking,
        truncation=False)
    batch = {k: v.to("cuda") for k, v in batch.items()}

    with torch.no_grad():
        out = model.generate(
            **batch, max_new_tokens=max_new, do_sample=False,
            eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id,
            stop_strings=["<|im_end|>"], tokenizer=tok)
    new_tokens = out[0, batch["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


# =============================================================================
# 判题（gpt-5.6-luna API）
# =============================================================================

def judge_with_luna(question: str, official_answer: str, candidate_answer: str,
                    api_url: str, api_key: str, model: str = "gpt-5.6-luna",
                    timeout: int = 120, max_retries: int = 2) -> tuple[bool, str]:
    """调用 gpt-5.6-luna 判题器；返回 (verdict, raw)。

    - 解析 JSON 输出；输出不含 "CORRECT"/"INCORRECT" 关键字时重试。
    - 网络异常同样重试（指数退避 2s/8s）；仍失败抛异常，由主流程记录。
    """
    import json
    import re
    import time
    import urllib.error
    import urllib.request

    def parse_verdict(raw: str) -> bool | None:
        """从判题输出提取 verdict。

        - 优先解析 JSON 的 ``verdict`` 字段；
        - 失败则按词边界匹配 CORRECT/INCORRECT。

        修复 2026-09-11 系统性误判：旧逻辑用 ``"CORRECT" in upper``
        判断，而 INCORRECT 含 CORRECT 子串，导致所有判题结果恒为 True。
        """
        try:
            obj = json.loads(raw)
            v = obj.get("verdict")
            if isinstance(v, str):
                v = v.strip().upper()
                if v == "CORRECT":
                    return True
                if v == "INCORRECT":
                    return False
        except Exception:
            pass
        m = re.search(r"\b(CORRECT|INCORRECT)\b", raw.upper())
        if m:
            return m.group(1) == "CORRECT"
        return None

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": build_judge_user_prompt(
                question, official_answer, candidate_answer)},
        ],
        "max_tokens": 16,
        "temperature": 0.0,
    }
    data = json.dumps(payload).encode("utf-8")

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                api_url.rstrip("/") + "/v1/chat/completions",
                data=data,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {api_key}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            raw = body["choices"][0]["message"]["content"].strip()
            verdict = parse_verdict(raw)
            if verdict is not None:
                return verdict, raw
            last_exc = RuntimeError(f"判题输出无法解析 verdict: {raw[:80]!r}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            last_exc = RuntimeError(f"判题响应解析失败: {exc}")
        if attempt < max_retries:
            time.sleep(2 ** (attempt + 1))  # 2s / 8s
    raise RuntimeError(f"判题重试 {max_retries} 次仍失败: {last_exc}") from last_exc


# =============================================================================
# 主流程
# =============================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AA-LCR 长上下文真实评测接口",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="评测配置 yaml")
    parser.add_argument("--limit", type=int, default=None,
                        help="只跑前 N 题（冒烟用，默认全量）")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="续跑：跳过 samples/aa_lcr.json 中已判题的条目")
    parser.add_argument("--no-judge", action="store_true", default=False,
                        help="跳过判题（仅生成样本），用于纯冒烟")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)

    logger = setup_logger(
        "aa_lcr",
        log_path=str(Path(cfg.output.root) / cfg.version / "logs" / "aa_lcr.log"))
    logger.info("=== AA-LCR 长上下文评测 ===")
    logger.info("配置: %s | 模型: %s", args.config, cfg.model.path)
    logger.info("数据: %s | 文档: %s", cfg.evaluation.aa_lcr_csv_path,
                cfg.evaluation.aa_lcr_docs_dir)
    logger.info("判题: %s (model=%s) key=%s", cfg.evaluation.judge_api_url,
                cfg.evaluation.judge_model,
                "已设置" if os.environ.get("JUDGE_API_KEY") else "缺失!")

    if not os.environ.get("JUDGE_API_KEY"):
        logger.error("缺少环境变量 JUDGE_API_KEY，判题将全部失败")
        print("错误: 请先 export JUDGE_API_KEY=... 再运行", file=sys.stderr)
        return 2
    logger.info("limit: %s | resume: %s", args.limit, args.resume)

    run_dir = create_run_dir(cfg)
    save_config_snapshot(cfg, run_dir)
    logger.info("结果目录: %s", run_dir)

    try:
        update_status(cfg, "running")
    except Exception as exc:
        logger.warning("registry 状态更新失败（不影响评测）: %s", exc)

    # 1) 数据
    rows = load_rows(cfg.evaluation.aa_lcr_csv_path)
    if args.limit:
        rows = rows[: args.limit]
    logger.info("待评测题目: %d", len(rows))
    if not rows:
        logger.warning("无题目可测")
        return 0

    # 3) 生成 + 判题
    # 生成参数：thinking 默认关闭；配置文件（evaluation.gen_kwargs）可开启
    # （enable_thinking）并调大 max_new_tokens（防 1024 token 硬截断）。
    gen_kwargs = dict(cfg.evaluation.gen_kwargs or {})
    max_new = int(gen_kwargs.pop("max_new_tokens", 1024))
    enable_thinking = bool(gen_kwargs.pop("enable_thinking", False))
    if gen_kwargs:
        logger.warning("gen_kwargs 存在未使用项: %s", gen_kwargs)

    # 2) 增量样本文件（续跑/崩溃恢复用）
    samples_path = run_dir / "samples" / "aa_lcr.json"
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if args.resume and samples_path.exists():
        try:
            done = {s["question_id"]: s for s in json.loads(
                samples_path.read_text(encoding="utf-8"))}
        except Exception:
            done = {}
        logger.info("续跑：已跳过 %d 题", len(done))

    # 3) 生成 + 判题
    model, tok = load_model_and_tokenizer(cfg.model.path, cfg.model.dtype)
    logger.info("模型加载完成")

    results: list[dict] = []
    pending = [r for r in rows if r["question_id"] not in done]
    n = len(pending)
    t_start = time.time()
    for i, row in enumerate(pending, start=1):
        qid = row["question_id"]
        sample: dict = {
            "question_id": qid,
            "document_category": row["document_category"],
            "document_set_id": row["document_set_id"],
            "question": row["question"],
            "official_answer": row["answer"],
            "input_tokens": row["input_tokens"],
        }
        try:
            docs = load_docs(row, cfg.evaluation.aa_lcr_docs_dir)
            cand = generate_answer(model, tok, docs, row["question"],
                                   max_new=max_new,
                                   enable_thinking=enable_thinking)
            sample["candidate_answer"] = cand
            if not args.no_judge:
                verdict, raw = judge_with_luna(
                    row["question"], row["answer"], cand,
                    cfg.evaluation.judge_api_url,
                    os.environ["JUDGE_API_KEY"],
                    cfg.evaluation.judge_model)
                sample["verdict"] = verdict
                sample["judge_raw"] = raw
            else:
                sample["verdict"] = None
        except Exception as exc:
            logger.error("[%d/%d] qid=%s 失败: %s", i, n, qid, exc)
            sample["error"] = str(exc)
            sample["verdict"] = None
        results.append(sample)
        done[qid] = sample

        # 增量落盘（每题都写，崩溃最多丢一题）
        try:
            samples_path.write_text(
                json.dumps(list(done.values()), ensure_ascii=False, indent=2),
                encoding="utf-8")
        except Exception as exc:
            logger.warning("样本落盘失败: %s", exc)

        v = sample.get("verdict")
        verdict_s = "CORRECT" if v is True else ("INCORRECT" if v is False else "SKIP")
        logger.info("[%d/%d] qid=%s %s (%ds 累计)",
                    i, n, qid, verdict_s, int(time.time() - t_start))

    # 4) 聚合
    # 聚合依据 = 显式写入过 verdict 的样本（含 False，排除 SKIP/error）
    judged = [s for s in results if s.get("verdict") is not None]
    correct = sum(1 for s in judged if s["verdict"])
    agg = {
        "dataset": "AA-LCR",
        "dataset_version": "v1.1",
        "n_questions": len(results),
        "n_judged": len(judged),
        "correct": correct,
        "accuracy": round(correct / len(judged), 6) if judged else None,
        "judge_model": cfg.evaluation.judge_model,
        "total_generation_seconds": round(time.time() - t_start, 1),
    }
    result = {"dimensions": {"long_context": {"aa_lcr": agg}}}
    save_result_json(cfg, result)
    logger.info("=== AA-LCR 完成 accuracy=%s (%d/%d) ===",
                agg["accuracy"], correct, len(judged))

    try:
        update_status(cfg, "completed")
    except Exception as exc:
        logger.warning("registry 状态更新失败（不影响结果）: %s", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
