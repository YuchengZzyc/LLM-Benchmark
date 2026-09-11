r"""通用工具函数：日志、环境信息收集、registry 读取。

>>> from src import utils
>>> env = utils.collect_environment_info()
>>> 'Python' in env
True
"""
from __future__ import annotations

import logging
import os
import platform
import sys
from datetime import datetime, timezone
from typing import Any, Optional

try:
    from src.config_loader import EvalConfig, load_yaml
except ImportError:  # 兼容直接以脚本运行
    from config_loader import EvalConfig, load_yaml


def setup_logger(name: str, log_path: Optional[str] = None,
                 level: int = logging.INFO) -> logging.Logger:
    """创建 logger；可选同时输出到终端与日志文件。

    >>> from src import utils
    >>> logger = utils.setup_logger('demo', None, logging.DEBUG)
    >>> logger.name
    'demo'
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def _pkg_version(modname: str) -> str:
    try:
        mod = __import__(modname)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "not installed"


def collect_environment_info() -> dict[str, str]:
    """收集当前环境的关键版本信息（用于报告 Environment 章节）。

    >>> from src import utils
    >>> info = utils.collect_environment_info()
    >>> info['OS']
    'Windows'
    """
    info: dict[str, str] = {
        "OS": platform.system(),
        "OS Release": platform.release(),
        "Python": platform.python_version(),
        "PyTorch": _pkg_version("torch"),
        "transformers": _pkg_version("transformers"),
        "accelerate": _pkg_version("accelerate"),
        "lm_eval": _pkg_version("lm_eval"),
        "PyYAML": _pkg_version("yaml"),
        "GPU": _gpu_info(),
    }
    return info


def _gpu_info() -> str:
    try:
        import torch  # 延迟导入：无 torch 环境不报错
    except Exception:
        return "not available"
    if not torch.cuda.is_available():
        return "none (cuda unavailable)"
    count = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(count)]
    return "; ".join(names) or "cuda"


def utc_now_str() -> str:
    """UTC 时间字符串。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_registry(cfg: EvalConfig) -> list[dict[str, Any]]:
    """读取 registry 中的模型版本列表（文件缺失时返回空列表）。"""
    path = cfg.output.registry
    if not os.path.exists(path):
        return []
    data = load_yaml(path)
    models = data.get("models", []) if isinstance(data, dict) else []
    return list(models)


def version_comparison_from_registry(cfg: EvalConfig) -> dict[str, str]:
    """从 registry 汇总所有版本 -> 状态的映射（供 Comparison 章节）。"""
    return {m.get("version", "?"): m.get("status", "?")
            for m in read_registry(cfg)}


if __name__ == "__main__":
    import doctest
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    doctest.testmod()
    print("utils OK")
