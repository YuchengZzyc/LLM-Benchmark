"""结果目录与 registry 管理。

每次评测运行在 ``results/<version>/`` 下建立版本化目录并落盘：

.. code-block:: text

    results/<version>/
    ├── config.yaml        # 本次运行的配置快照
    ├── logs/              # 运行日志
    ├── samples/           # 模型样本输出
    ├── result.json        # 结构化评测结果
    └── report.md          # 生成的 Markdown 报告

同时维护 ``results/registry.yaml``（记录所有模型版本，绝不覆盖旧版本）：

.. code-block:: yaml

    models:
      - version: qwen35-4b-base-v0.0
        type: baseline
        model: Qwen3.5-4B
        status: initialized
        results_path: results/baseline/qwen35-4b-base-v0.0

>>> from src.config_loader import load_config
>>> cfg = load_config('configs/template.yaml')
>>> ensure_run_dir(cfg)  # doctest: +SKIP
"""
from __future__ import annotations

import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    from src.config_loader import EvalConfig, dump_yaml, load_yaml
except ImportError:  # 兼容直接以脚本运行
    from config_loader import EvalConfig, dump_yaml, load_yaml


def utc_now_iso() -> str:
    """UTC 时间 ISO 字符串，如 ``2026-09-09T12:34:56Z``。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_dir_for(cfg: EvalConfig, root: Optional[str] = None) -> Path:
    """解析版本化结果目录 ``<output.root>/<version>/``。

    >>> from src.config_loader import load_config
    >>> cfg = load_config('configs/template.yaml')
    >>> run_dir_for(cfg).name
    'qwen35-4b-base-v0.0'
    """
    root = str(root or cfg.output.root)
    return Path(root) / cfg.version


def create_run_dir(cfg: EvalConfig, overwrite: bool = False) -> Path:
    """创建本次运行的版本化目录（含子目录 logs/ 与 samples/）。

    - 默认拒绝覆盖已存在的版本目录，避免污染历史结果。
    - 返回目录 Path。已存在时仅校验并通过（registry 登记仍会进行）。
    """
    run_dir = run_dir_for(cfg)
    if run_dir.exists():
        if not overwrite:
            return run_dir
    for sub in ("logs", "samples"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config_snapshot(cfg: EvalConfig, run_dir: Optional[Path] = None) -> Path:
    """将配置落盘为 ``<run_dir>/config.yaml`` 快照，供结果回溯。"""
    run_dir = Path(run_dir) if run_dir else run_dir_for(cfg)
    target = run_dir / "config.yaml"
    cfg.dump(str(target))
    return target


def _merge_metrics(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """按维度合并 metrics：主框架（run_lm_eval）与长上下文（run_longbench）
    共写同一 result.json，后写的一侧只更新自己产出的维度 / 顶层键，
    不整体替换（否则会把先跑一侧的分数冲掉，冒烟 2026-09-10 实际踩坑）。

    嵌套维度的 key 做**深度合并**：同一维度下不同子任务（如 long_context 下的
    longbench_* 与 aa_lcr）必须并存，不能用浅层 ``dict.update`` 互相覆盖
    （2026-09-11 实测：AA-LCR run2 落盘时把 LongBench 的 12 个任务分数冲掉了）。
    """
    combined = dict(old)
    new_dims = new.get("dimensions")
    if isinstance(new_dims, dict):
        dims = dict(combined.get("dimensions") or {})
        for dk, dv in new_dims.items():
            if isinstance(dv, dict) and isinstance(dims.get(dk), dict):
                inner = dict(dims[dk])
                inner.update(dv)
                dims[dk] = inner
            else:
                dims[dk] = dv
        combined["dimensions"] = dims
    for k, v in new.items():
        if k != "dimensions":
            combined[k] = v
    return combined


def save_result_json(cfg: EvalConfig, metrics: dict[str, Any],
                     run_dir: Optional[Path] = None) -> Path:
    """写入结构化评测结果 ``result.json``（若已存在则按维度合并新增指标）。"""
    run_dir = Path(run_dir) if run_dir else run_dir_for(cfg)
    target = run_dir / "result.json"
    existing: dict[str, Any] = {}
    if target.exists():
        try:
            existing = load_yaml(str(target))
        except Exception:
            existing = {}
    merged = dict(existing)
    merged["_updated"] = utc_now_iso()
    merged["version"] = cfg.version
    merged["model"] = cfg.model.name
    if isinstance(metrics, dict):
        old_metrics = merged.get("metrics")
        if isinstance(old_metrics, dict):
            merged["metrics"] = _merge_metrics(old_metrics, metrics)
        else:
            merged["metrics"] = metrics
    with io.open(str(target), "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    return target


def save_sample_payload(cfg: EvalConfig, task: str, payload: Any,
                        run_dir: Optional[Path] = None) -> Path:
    """保存单个任务的样本输出到 ``samples/<task>.json``。"""
    run_dir = Path(run_dir) if run_dir else run_dir_for(cfg)
    target = run_dir / "samples" / f"{task}.json"
    with io.open(str(target), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return target


# =============================================================================
# Registry（版本登记）
# =============================================================================

# 语义合法状态：initialized=目录已建立；pending=待评测；running=评测中；
# completed=评测完成；failed=评测失败
VALID_STATUSES = ("initialized", "pending", "running", "completed", "failed")


def _ensure_registry(cfg: EvalConfig) -> None:
    reg_path = Path(cfg.output.registry)
    if not reg_path.exists():
        reg_path.parent.mkdir(parents=True, exist_ok=True)
        dump_yaml({"models": []}, str(reg_path))


def load_registry(cfg: EvalConfig) -> dict[str, Any]:
    """读取 registry.yaml；不存在时初始化为空登记表。"""
    reg_path = Path(cfg.output.registry)
    if not reg_path.exists():
        return {"models": []}
    try:
        return load_yaml(str(reg_path))
    except Exception:
        return {"models": []}


def _entry(version: str) -> dict[str, Any] | None:
    """占位辅助（保留以提示 registry 条目形状）。"""
    return None


def register_model(cfg: EvalConfig, status: str = "initialized",
                   extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """登记（或更新）模型版本。若版本已存在则保留其历史字段（不覆盖旧版本）。

    >>> from src.config_loader import load_config
    >>> cfg = load_config('configs/template.yaml')
    >>> e = register_model(cfg)  # doctest: +SKIP
    """
    if status not in VALID_STATUSES:
        raise ValueError(
            f"status 非法: {status!r}，可选: {VALID_STATUSES}")
    _ensure_registry(cfg)
    reg = load_registry(cfg)
    models = reg.setdefault("models", [])
    now = utc_now_iso()

    idx = next((i for i, m in enumerate(models)
                if m.get("version") == cfg.version), None)
    entry: dict[str, Any] = {
        "version": cfg.version,
        "type": cfg.model.type,
        "model": cfg.model.name,
        "status": status,
        "results_path": str(run_dir_for(cfg).as_posix()),
        "registered_at": now,
        "last_update": now,
    }
    if extra:
        entry.update(extra)

    if idx is None:
        models.append(entry)
    else:
        # 只更新状态/时间戳类字段，保留既有信息（版本目录与历史不覆盖）
        old = models[idx]
        for k, v in entry.items():
            if k not in ("registered_at",):
                old[k] = v
        models[idx] = old

    dump_yaml(reg, str(cfg.output.registry))
    return entry


def update_status(cfg: EvalConfig, status: str) -> dict[str, Any]:
    """更新某版本的状态（如 running -> completed）。"""
    if status not in VALID_STATUSES:
        raise ValueError(
            f"status 非法: {status!r}，可选: {VALID_STATUSES}")
    _ensure_registry(cfg)
    reg = load_registry(cfg)
    models = reg.setdefault("models", [])
    for m in models:
        if m.get("version") == cfg.version:
            m["status"] = status
            m["last_update"] = utc_now_iso()
            dump_yaml(reg, str(cfg.output.registry))
            return m
    raise KeyError(f"registry 中不存在版本: {cfg.version!r}，请先 register_model")


if __name__ == "__main__":
    import doctest
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    doctest.testmod()
    print("result_manager OK")
