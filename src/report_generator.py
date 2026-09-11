"""Markdown 评测报告生成器。

报告包含五个固定章节：Model Information / Environment / Benchmark Configuration /
Results / Comparison。Results 支持嵌套维度，Comparison 用于跨版本横向比较。

>>> from src.config_loader import load_config
>>> from src.report_generator import generate_report_md
>>> cfg = load_config('configs/template.yaml')
>>> md = generate_report_md(cfg)  # doctest: +SKIP
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

try:
    from src.config_loader import EvalConfig
except ImportError:  # 兼容直接以脚本运行
    from config_loader import EvalConfig

_SECTIONS = (
    "Model Information",
    "Environment",
    "Benchmark Configuration",
    "Results",
    "Comparison",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _task_table(cfg: EvalConfig) -> list[str]:
    rows = [
        "| Dimension | Benchmark | Metric |",
        "| --- | --- | --- |",
    ]
    dims = {
        "knowledge": "Accuracy",
        "reasoning": "Exact Match / Accuracy",
        "instruction": "strict / loose accuracy",
        "longbench2": "task score (lm-eval)",
        "long_context": "task score",
        "code": "pass@1 / accuracy",
        "multilingual": "language accuracy",
    }
    for dim, metric in dims.items():
        tasks = cfg.tasks.get(dim, [])
        if not tasks:
            continue
        first = True
        for t in tasks:
            if first:
                rows.append(f"| {dim} | `{t}` | {metric} |")
                first = False
            else:
                rows.append(f"|  | `{t}` |  |")
    return rows


def _metrics_block(metrics: Any, level: int = 3) -> list[str]:
    if not isinstance(metrics, dict):
        return [f"{(level + 2) * '#'} 原始结果\n\n```json\n{metrics}\n```\n"]
    lines: list[str] = []
    for k, v in metrics.items():
        if isinstance(v, dict):
            lines.append(f"{level * '#'} {k}\n")
            lines.extend(_metrics_block(v, level + 1))
        else:
            lines.append(f"- **{k}**: {v}")
    return lines


def generate_report_md(cfg: EvalConfig,
                       metrics: Any = None,
                       env: Any = None,
                       comparison: Any = None) -> str:
    """生成 Markdown 报告模板。

    - ``metrics``：结构化结果（dict，可含嵌套维度）
    - ``env``：环境信息 dict（未提供时输出占位模板）
    - ``comparison``：跨版本对比 dict（未提供时输出占位模板）
    """
    out: list[str] = []
    out.append(f"# Evaluation Report — {cfg.model.name}\n")
    out.append(f"> Generated at: {_utc_now()}  \n")
    out.append(f"> Config: `{cfg.config_path or cfg.version}`\n")
    out.append("\n---\n")

    # 1. Model Information
    out.append("## Model Information\n")
    out.append("| Field | Value |")
    out.append("| --- | --- |")
    out.append(f"| Model | {cfg.model.name} |")
    out.append(f"| Version | `{cfg.model.version}` |")
    out.append(f"| Type | `{cfg.model.type}` |")
    out.append(f"| Path | `{cfg.model.path}` |")
    out.append(f"| Dtype | `{cfg.model.dtype}` |")
    out.append(f"| Trust Remote Code | `{cfg.model.trust_remote_code}` |")
    if cfg.model.revision:
        out.append(f"| Revision | `{cfg.model.revision}` |")
    out.append("")

    # 2. Environment
    out.append("## Environment\n")
    if isinstance(env, dict):
        out.append("| Field | Value |")
        out.append("| --- | --- |")
        for k, v in env.items():
            out.append(f"| {k} | {v} |")
    else:
        out.append("_环境信息待服务器阶段回填（见 environment.txt / docs/environment_setup.md）_\n")
    out.append("")

    # 3. Benchmark Configuration
    out.append("## Benchmark Configuration\n")
    out.extend(_task_table(cfg))
    out.append("")
    out.append("| Setting | Value |")
    out.append("| --- | --- |")
    out.append(f"| Framework | `{cfg.evaluation.framework}` |")
    out.append(f"| Batch Size | `{cfg.evaluation.batch_size}` |")
    out.append(f"| Apply Chat Template | `{cfg.evaluation.apply_chat_template}` |")
    out.append(f"| Max Length | `{cfg.evaluation.max_length}` |")
    out.append(f"| Num Fewshot | `{cfg.evaluation.num_fewshot}` |")
    out.append(f"| Seed | `{cfg.evaluation.seed}` |")
    out.append(f"| Device | `{cfg.evaluation.device}` |")
    out.append("")

    # 4. Results
    out.append("## Results\n")
    if metrics:
        out.extend(_metrics_block(metrics))
    else:
        out.append("_尚未执行评测：结果为占位模板，无真实数据。_\n")
    out.append("")

    # 5. Comparison
    out.append("## Comparison\n")
    if isinstance(comparison, dict) and comparison:
        out.append("| Version | Status |")
        out.append("| --- | --- |")
        for k, v in comparison.items():
            out.append(f"| {k} | {v} |")
    else:
        out.append(
            "_跨版本横向比较：填写多个版本后自动对比（当前仅 baseline 阶段，暂无对比数据）。_\n")

    return "\n".join(out)


def write_report(cfg: EvalConfig, path: str,
                 metrics: Any = None, env: Any = None,
                 comparison: Any = None) -> None:
    """将报告写入文件（含 BOM，便于 Windows/中文编辑器正确识别 UTF-8）。"""
    md = generate_report_md(cfg, metrics, env, comparison)
    with open(path, "w", encoding="utf-8-sig", newline="\n") as f:
        f.write(md)


if __name__ == "__main__":
    import doctest
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    doctest.testmod()
    print("report_generator OK")
