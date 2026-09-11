#!/usr/bin/env python3
"""run_lm_eval.py — lm-eval 评测入口（占位）。

调用示例：:

    python scripts/run_lm_eval.py --config configs/baseline_qwen35_4b.yaml

当前阶段为框架搭建：本脚本解析参数、加载配置、初始化结果目录并完成流程占位，
**不会加载模型、不会调用 lm-eval、不生成任何评测结果**。
服务器阶段在此接入真实 harness 调用（见 evaluate() 的 TODO）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import load_config  # noqa: E402
from src.evaluator import evaluate  # noqa: E402
from src.result_manager import (  # noqa: E402
    create_run_dir, save_config_snapshot, save_result_json,
)
from src.utils import setup_logger  # noqa: E402

FRAMEWORK_DIMENSIONS = ("knowledge", "reasoning", "instruction", "longbench2", "code", "multilingual")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="lm-eval 评测入口（当前阶段：流程占位）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True,
        help="评测配置文件路径，如 configs/baseline_qwen35_4b.yaml")
    parser.add_argument(
        "--tasks", nargs="*", default=None,
        help="覆盖要运行的任务列表（默认按配置 tasks 分组运行）")
    parser.add_argument(
        "--keep-samples", action="store_true", default=False,
        help="保存模型样本输出到 results/<version>/samples/")
    parser.add_argument(
        "--overwrite", action="store_true", default=False,
        help="允许覆盖已存在的版本结果目录（默认拒绝）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cfg = load_config(args.config)

    if args.tasks:
        selected: set[str] = set(args.tasks)
        for dim in FRAMEWORK_DIMENSIONS:
            cfg.tasks[dim] = [t for t in cfg.tasks.get(dim, []) if t in selected]

        cfg.tasks["long_context"] = [t for t in cfg.tasks.get("long_context", []) if t in selected]
    logger = setup_logger(
        "run_lm_eval", log_path=str(Path(cfg.output.root) / cfg.version / "logs" / "run.log"))

    logger.info("=== Qwen3.5-4B Benchmark Evaluation ===")
    logger.info("配置: %s", args.config)
    logger.info("模型版本: %s", cfg.version)
    logger.info("任务分组: %s", cfg.tasks)

    # 建立版本化结果目录并登记
    run_dir = create_run_dir(cfg, overwrite=args.overwrite)
    save_config_snapshot(cfg, run_dir)
    logger.info("结果目录: %s", run_dir)

    # 执行评测（调用 lm-eval 真实后端；未安装时给出明确提示）
    try:
        result = evaluate(cfg)
    except Exception as exc:
        logger.error("评测失败: %s", exc)
        print(f"评测失败: {exc}", file=sys.stderr)
        return 1

    logger.info("评测结果: %s", result)

    # 落盘结构化结果 result.json
    save_result_json(cfg, result)

    if result.get("errors"):
        logger.warning("评测存在错误: %s", result["errors"])
    else:
        logger.info("全部维度评测完成（状态 completed）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
