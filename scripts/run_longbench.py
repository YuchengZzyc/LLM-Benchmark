#!/usr/bin/env python3
"""run_longbench.py — LongBench 长上下文真实评测入口。

与主框架（lm-eval）共用同一配置体系与版本化结果目录，
但长上下文评测**不绑定**主框架（独立加载模型、独立评分）。

用法::

    python scripts/run_longbench.py --config configs/baseline_qwen35_4b.yaml
    python scripts/run_longbench.py --config configs/baseline_qwen35_4b.yaml --tasks hotpotqa 2wikimqa
    python scripts/run_longbench.py --config configs/baseline_qwen35_4b.yaml --max-length 32768

流程：加载配置 -> 建立版本化结果目录 -> 执行 LongBench 任务
（每个任务的评分指标写入 result.json，样本可另存 samples/）。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import load_config  # noqa: E402
from src.evaluator import NotConfiguredError  # noqa: E402
from src.longbench import TASK_META, run_longbench  # noqa: E402
from src.result_manager import (  # noqa: E402
    create_run_dir, save_config_snapshot, save_result_json, update_status,
)
from src.utils import setup_logger  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LongBench 长上下文真实评测接口（独立于 lm-eval）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True,
        help="评测配置文件路径，如 configs/baseline_qwen35_4b.yaml")
    parser.add_argument(
        "--tasks", nargs="*", default=None,
        help="指定 LongBench 子任务（默认按配置 tasks.long_context 运行）")
    parser.add_argument(
        "--max-length", type=int, default=None,
        help="覆盖配置中的 evaluation.max_length（上下文截断长度）")
    parser.add_argument(
        "--keep-samples", action="store_true", default=False,
        help="保存样本输出到 results/<version>/samples/（longbench_<task>.json）")
    parser.add_argument(
        "--overwrite", action="store_true", default=False,
        help="允许覆盖已存在的版本结果目录（默认拒绝）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)

    if args.max_length:
        cfg.evaluation.max_length = args.max_length
        logger_note = f"（--max-length 覆盖为 {args.max_length}）"
    else:
        logger_note = ""

    if args.tasks:
        cfg.tasks["long_context"] = list(args.tasks)

    logger = setup_logger(
        "run_longbench",
        log_path=str(Path(cfg.output.root) / cfg.version / "logs" / "longbench.log"))

    logger.info("=== LongBench 长上下文真实评测 ===")
    logger.info("配置: %s", args.config)
    logger.info("LongBench 任务: %s", cfg.tasks.get("long_context", []))
    logger.info("max_length: %s%s", cfg.evaluation.max_length, logger_note)
    logger.info("模型: %s (dtype=%s, device=%s)", cfg.model.path,
                cfg.model.dtype, cfg.evaluation.device)

    # 建立版本化结果目录（与主框架共用；默认拒绝覆盖）
    run_dir = create_run_dir(cfg, overwrite=args.overwrite)
    save_config_snapshot(cfg, run_dir)
    logger.info("结果目录: %s", run_dir)

    # 检查任务合法性（提前给出明确错误，而不是等评测开始时才发现）
    tasks = list(cfg.tasks.get("long_context", []))
    unknown = [t for t in tasks if t not in TASK_META]
    if unknown:
        logger.error("配置了未知的 LongBench 任务: %s", unknown)
        print(f"错误: 未知 LongBench 任务 {unknown}；"
              f"可用任务见 src/longbench.py 的 TASK_META", file=sys.stderr)
        return 2
    if not tasks:
        logger.warning("tasks.long_context 为空，未执行任何 LongBench 任务。"
                       "请在配置中填入子任务名（如 hotpotqa / multifieldqa_en）。")
        return 0

    try:
        update_status(cfg, "running")
    except Exception as exc:  # registry 状态更新失败不阻断评测
        logger.warning("registry 状态更新失败（不影响评测）: %s", exc)

    # 执行真实评测（加载模型、逐任务推理与评分）
    try:
        result = run_longbench(cfg, tasks, logger=logger)
    except NotConfiguredError as exc:
        logger.error("依赖未配置: %s", exc)
        print(f"依赖未配置: {exc}", file=sys.stderr)
        try:
            update_status(cfg, "failed")
        except Exception:
            pass
        return 1
    except Exception as exc:
        logger.error("评测失败: %s", exc)
        print(f"评测失败: {exc}", file=sys.stderr)
        try:
            update_status(cfg, "failed")
        except Exception:
            pass
        return 1

    # 落盘结构化结果（合并写入 result.json；与主框架共用同一文件）
    save_result_json(cfg, result)
    logger.info("评测结果已写入: %s", run_dir / "result.json")

    dim = result.get("dimensions", {}).get("long_context", {})
    errored = [t for t, v in dim.items()
               if isinstance(v, dict) and v.get("status") == "error"]
    skipped = [t for t, v in dim.items()
               if isinstance(v, dict) and v.get("status") == "not_supported"]
    if errored:
        logger.warning("存在失败任务: %s", errored)
        try:
            update_status(cfg, "failed")
        except Exception:
            pass
        return 1
    if skipped:
        logger.info("代码类任务未接入（不生成虚假分数）: %s", skipped)

    try:
        update_status(cfg, "completed")
    except Exception as exc:
        logger.warning("registry 状态更新失败（不影响结果）: %s", exc)
    logger.info("=== LongBench 评测完成（status=completed）===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
