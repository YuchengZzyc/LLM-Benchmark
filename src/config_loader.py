"""统一配置对象：读取 yaml、校验必填字段、提供属性访问与可复现快照。

本模块不依赖 PyYAML —— 若已安装则使用快速 C 加速，否则使用极简 yaml 子集解析器
（支持本仓库 configs/ 所需的键值/列表/注释结构）。因此框架源码在任何 Python 3.8+
环境均可直接导入验证。

>>> cfg = load_config('configs/template.yaml')
>>> cfg.model.name
'Qwen3.5-4B'
>>> cfg.tasks['knowledge'][0]
'mmlu_pro'
>>> 'version' in cfg.raw['model']
True

>>> import os
>>> cfg.dump('/tmp/_cfg_check.yaml')  # doctest: +SKIP
"""
from __future__ import annotations

import copy
import dataclasses
import io
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

try:  # 优先使用系统 PyYAML
    import yaml as _yaml
    _HAS_PYYAML = True
except Exception:  # pragma: no cover - 兜底解析器路径
    _yaml = None
    _HAS_PYYAML = False


# =============================================================================
# 极简 YAML 子集解析（无 PyYAML 时的兜底，仅覆盖配置所需语法）
# =============================================================================

def _parse_scalar(token: str) -> Any:
    token = token.strip()
    if token == "":
        return ""
    if token.lower() in ("null", "~"):
        return None
    if token.lower() == "true":
        return True
    if token.lower() == "false":
        return False
    if re.fullmatch(r"-?\d+", token):
        return int(token)
    if re.fullmatch(r"-?\d+\.\d+", token):
        return float(token)
    if token.startswith('"') and token.endswith('"') and len(token) >= 2:
        return token[1:-1].replace('\\"', '"')
    if token.startswith("'") and token.endswith("'") and len(token) >= 2:
        return token[1:-1].replace("''", "'")
    if token.startswith("#"):
        return ""
    return token


def _parse_block_list(block: str) -> list[Any]:
    items: list[Any] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        items.append(_parse_scalar(stripped.lstrip("- ")))
    return items


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """解析 ``key: value`` / ``key:`` + 嵌套映射 / ``- item`` 列表的 yaml 子集。

    >>> _parse_simple_yaml('a: 1\\nb: true\\nc:\\n  d: hello')
    {'a': 1, 'b': True, 'c': {'d': 'hello'}}
    >>> _parse_simple_yaml('t:\\n  - x\\n  - y')
    {'t': ['x', 'y']}
    """
    out: dict[str, Any] = {}
    lines = text.splitlines()

    def indent_of(line: str) -> int:
        return len(line) - len(line.lstrip(" "))

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        i += 1
        if not stripped or stripped.startswith("#"):
            continue
        indent = indent_of(line)
        if stripped.startswith("- "):
            raise ValueError(f"顶层不支持列表项: {line!r}")
        if ":" not in stripped:
            continue
        key, _, rest = stripped.partition(":")
        key = key.strip()
        rest = rest.strip()
        if rest and not rest.startswith("#"):
            if rest.startswith("- "):  # 行内列表
                out[key] = _parse_block_list("- " + rest)
            else:
                out[key] = _parse_scalar(rest)
            continue
        # 嵌套块：收集缩进大于当前 key 的行
        block: list[str] = []
        while i < n:
            nxt = lines[i]
            if not nxt.strip() or nxt.strip().startswith("#"):
                i += 1
                continue
            if indent_of(nxt) > indent:
                block.append(nxt)
                i += 1
            else:
                break
        if not block:
            out[key] = None
        elif block[0].lstrip().startswith("- "):  # 嵌套块是列表
            out[key] = _parse_block_list("\n".join(block))
        else:  # 嵌套块是映射（去掉一级缩进递归解析）
            dedented = "\n".join(l[indent + 1:] for l in block)
            out[key] = _parse_simple_yaml(dedented)
    return out


def load_yaml(path: str) -> dict[str, Any]:
    """读取 yaml 文件并返回字典。"""
    with io.open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if _HAS_PYYAML:
        return _yaml.safe_load(text) or {}
    return _parse_simple_yaml(text)


def dump_yaml(data: Any, path: str) -> None:
    """将对象写为 yaml（有 PyYAML 时使用，否则以 repr 落盘作为快照）。"""
    with io.open(path, "w", encoding="utf-8") as f:
        if _HAS_PYYAML:
            _yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        else:
            f.write(_fallback_repr(data))


def _fallback_repr(data: Any, indent: int = 0) -> str:
    pad = "  " * indent
    if isinstance(data, dict):
        if not data:
            return f"{pad}{{}}\n"
        lines: list[str] = []
        for k, v in data.items():
            if isinstance(v, (dict, list)) and v:
                lines.append(f"{pad}{k}:\n{_fallback_repr(v, indent + 1)}")
            else:
                lines.append(f"{pad}{k}: {_fallback_scalar(v)}\n")
        return "".join(lines)
    if isinstance(data, list):
        if not data:
            return f"{pad}[]\n"
        return "".join(f"{pad}- {_fallback_scalar(v)}\n" for v in data)
    return f"{pad}{_fallback_scalar(data)}\n"


def _fallback_scalar(v: Any) -> str:
    if isinstance(v, str):
        return f"'{v}'" if _needs_quote(v) else v
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _needs_quote(v: str) -> bool:
    return v == "" or any(c in v for c in ":#{}[]&*!|>'\",%@`") or v.lower() in (
        "null", "true", "false")


# =============================================================================
# 配置对象
# =============================================================================

# 必填字段（存在即合法；值级校验见 validate_config）
_REQUIRED_TOP_KEYS = ("model", "evaluation", "tasks", "output")


@dataclass
class ModelConfig:
    name: str = ""
    version: str = ""
    type: str = "other"
    path: str = ""
    dtype: str = "auto"
    trust_remote_code: bool = False
    revision: Optional[str] = None
    tokenizer_path: Optional[str] = None
    # LoRA/PEFT adapter 目录（path 指向 base 模型，加载后挂载 adapter；
    # None = path 即完整模型，行为不变）
    adapter_path: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        """从 dict 构建 ModelConfig；缺失键采用 dataclass 默认值。"""
        keys = {f.name: f.default for f in dataclasses.fields(cls)}
        return cls(**{k: d.get(k, keys[k]) for k in keys})


@dataclass
class EvaluationConfig:
    framework: str = "lm-eval"
    batch_size: Any = "auto"
    apply_chat_template: bool = True
    max_length: int = 4096
    num_fewshot: Optional[int] = 0  # None = 使用各任务默认 few-shot（如 gsm8k 5-shot）
    seed: int = 42
    device: str = "cuda"
    longbench_data_dir: Optional[str] = None
    # AA-LCR 长上下文评测（scripts/aa_lcr_run.py）
    aa_lcr_csv_path: Optional[str] = None   # AA-LCR_Dataset.csv（100 行）
    aa_lcr_docs_dir: Optional[str] = None   # 解压后文档根目录（lcr/{category}/{set_id}/）
    judge_api_url: str = ""                 # OpenAI 兼容判题器 base URL
    judge_model: str = "gpt-5.6-luna"       # 判题器模型名（官方固定）
    gen_kwargs: dict[str, Any] = field(default_factory=lambda: {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 1024,
    })

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvaluationConfig":
        kwargs = dict(d)
        gen = kwargs.pop("gen_kwargs", None) or {}
        obj = cls(**kwargs)
        obj.gen_kwargs = dict(gen)
        return obj

@dataclass
class OutputConfig:
    root: str = "results/"
    registry: str = "results/registry.yaml"
    keep_samples: bool = True
    max_samples: int = 100

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OutputConfig":
        """从 dict 构建 OutputConfig；缺失键采用 dataclass 默认值。"""
        defaults = {f.name: f.default for f in dataclasses.fields(cls)}
        return cls(**{f.name: d.get(f.name, defaults[f.name])
                      for f in dataclasses.fields(cls)})


class EvalConfig:
    """统一配置对象。

    - ``cfg.model / cfg.evaluation / cfg.output``：结构化访问
    - ``cfg.tasks``：能力维度 -> benchmark 列表的映射（dict[str, list[str]]）
    - ``cfg.raw``：原始 yaml 字典（供报告/登记使用）
    - ``cfg.config_path``：来源文件路径
    - ``cfg.benchmark_list``：全量 benchmark 列表（含 long_context 独立接口任务）
    """

    def __init__(self, raw: dict[str, Any], config_path: str = ""):
        raw = copy.deepcopy(raw or {})
        for key in _REQUIRED_TOP_KEYS:
            if key not in raw or raw[key] is None:
                raise ValueError(f"配置缺少必填段: {key!r} (配置文件: {config_path})")
        self.raw = raw
        self.config_path = config_path
        self.model = ModelConfig.from_dict(raw["model"])
        self.evaluation = EvaluationConfig.from_dict(raw["evaluation"])
        self.output = OutputConfig.from_dict(raw["output"])
        self.tasks: dict[str, list[str]] = dict(raw["tasks"])

    @property
    def benchmark_list(self) -> list[str]:
        seen: list[str] = []
        for benchs in self.tasks.values():
            for b in benchs:
                if b not in seen:
                    seen.append(b)
        return seen

    @property
    def version(self) -> str:
        return self.model.version

    def dump(self, path: str) -> None:
        """将配置落盘为快照（写回时不包含 config_path）。"""
        dump_yaml(self.raw, path)


def validate_config(cfg: EvalConfig) -> None:
    """对配置做语义校验；不合法时抛出 ValueError（保持幂等，可重复调用）。

    - model.name / model.version / model.path 不能为空
    - model.version 不能包含路径分隔符或反斜杠（将作为结果目录名）
    - model.dtype 必须在合法集合内
    - tasks 各维度必须为列表
    - output.root 与 output.registry 不能为空

    >>> v = dict(load_yaml('configs/template.yaml'))
    >>> validate_config(EvalConfig(v, 'configs/template.yaml'))
    >>> b = dict(load_yaml('configs/baseline_qwen35_4b.yaml'))
    >>> validate_config(EvalConfig(b, 'configs/baseline_qwen35_4b.yaml'))
    """
    if not cfg.model.name:
        raise ValueError("model.name 不能为空")
    if not cfg.model.version:
        raise ValueError("model.version 不能为空")
    if not cfg.model.path:
        raise ValueError("model.path 不能为空（本地路径或 HuggingFace 仓库名）")
    if any(c in cfg.model.version for c in ("/", "\\", "..")):
        raise ValueError(
            f"model.version 包含非法字符（将作为结果目录名）: {cfg.model.version!r}")
    valid_dtypes = {"auto", "float16", "bfloat16", "float32", "int8", "int4"}
    if str(cfg.model.dtype).lower() not in valid_dtypes:
        raise ValueError(
            f"model.dtype 非法: {cfg.model.dtype!r}，可选: {sorted(valid_dtypes)}")
    for dim, benchs in cfg.tasks.items():
        if not isinstance(benchs, list):
            raise ValueError(f"tasks.{dim} 必须是列表")
        for b in benchs:
            if not isinstance(b, str):
                raise ValueError(f"tasks.{dim} 中存在非法 benchmark 项: {b!r}")
            if not b.strip():
                raise ValueError(
                    f"tasks.{dim} 存在空白/空 benchmark 项（如需保留空列表，请用 []）")
    if not str(cfg.output.root).strip():
        raise ValueError("output.root 不能为空")
    if not str(cfg.output.registry).strip():
        raise ValueError("output.registry 不能为空")


def load_config(path: str) -> EvalConfig:
    """加载 yaml 并返回通过校验的配置对象。

    >>> cfg = load_config('configs/template.yaml')
    >>> cfg.version
    'qwen35-4b-base-v0.0'
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")
    raw = load_yaml(path)
    cfg = EvalConfig(raw, config_path=path)
    validate_config(cfg)
    return cfg


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    print("config_loader OK")
