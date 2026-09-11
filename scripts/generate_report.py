#!/usr/bin/env python3
"""generate_report.py — 生成本次运行 Markdown 报告。

调用示例：:

    python scripts/generate_report.py --config configs/baseline_qwen35_4b.yaml

流程：读取配置 -> 读取本次结果（result.json，若存在）-> 收集环境信息
-> 读取 registry 汇总比较 -> 写入 results/<version>/report.md。
若尚未评测，则生成占位模板报告（README 已声明不会伪造结果）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import load_config  # noqa: E402
from src.report_generator import write_report  # noqa: E402
from src.result_manager import run_dir_for  # noqa: E402
from src.utils import (  # noqa: E402
    collect_environment_info, version_comparison_from_registry,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成本次运行 Markdown 报告",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True,
        help="评测配置文件路径，如 configs/baseline_qwen35_4b.yaml")
    parser.add_argument(
        "--out", default=None,
        help="报告输出路径（默认 results/<version>/report.md）")
    return parser.parse_args(argv)


def _load_metrics(run_dir: Path) -> dict | None:
    result_file = run_dir / "result.json"
    if not result_file.exists():
        return None
    try:
        import json
        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("metrics")
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cfg = load_config(args.config)
    run_dir = run_dir_for(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = _load_metrics(run_dir)
    env = collect_environment_info()
    comparison = version_comparison_from_registry(cfg)

    out = args.out or str(run_dir / "report.md")
    write_report(cfg, out, metrics=metrics, env=env, comparison=comparison)

    print(f"报告已生成: {out}")
    if metrics is None:
        print("注意: 该版本尚无 result.json，报告为占位模板（无真实评测数据）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
