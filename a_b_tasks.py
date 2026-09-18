"""注入 mmlu_prox 的 A/B 对照任务（仅在本次探针用到；不覆盖原任务）。

为 mmlu_prox_zh 生成：
  - mmlu_prox_zh_nothink_math / _other：
      复制原 zh 叶任务定义，将 model_args 设为
      `enable_thinking=False,think_end_token=</think>`，
      生成路径即走 lm-eval 原生"关 thinking + 剥离前导"逻辑（与改造 chat_template
      等价），供与默认模板（thinking 开）的同一学科直接对比。
"""
from __future__ import annotations

import os

TASKS_YAML_DIR = os.path.dirname(os.path.abspath(__file__))

_NOTHINK_YAML = """\
task: {name}
group: {name}
dataset_path: li-lab/MMLU-ProX
dataset_name: zh
test_split: test
fewshot_split: validation
fewshot_config:
  sampler: first_n
  doc_to_text: !function utils.fewshot_to_text
  doc_to_target: ""
output_type: generate_until
doc_to_text: !function utils.doc_to_text
doc_to_target: answer
filter_list:
  - name: "custom-extract"
    filter:
      - function: "regex"
        regex_pattern: '答案是 \\(?([ABCDEFGHIJ])\\)?'
      - function: "take_first"
generation_kwargs:
  until:
    - "</s>"
    - "Q:"
    - "问题："
    - "<|im_end|>"
  do_sample: false
  temperature: 0.0
  max_gen_toks: 2048
num_fewshot: 5
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    ignore_punctuation: true
metadata:
  version: 0.0
"""

# 直接复用 mmlu_prox/zh/utils.py 的函数（process_* 是按学科过滤）
_NOTHINK_TASKS = {
    "mmlu_prox_zh_nothink_math": {"subject": "math"},
    "mmlu_prox_zh_nothink_other": {"subject": "other"},
}

# 覆盖（hook）：a_b_tasks 注入 lm-eval 的 task index
def a_b_tasks(register):
    """注册 A/B 对照任务。``register`` 为 lm_eval.tasks 的模块命名空间。"""
    from lm_eval.api.task import ConfigurableTask

    from lm_eval.tasks.mmlu_prox.zh import utils as zh_utils  # noqa: F401

    for name, meta in _NOTHINK_TASKS.items():
        subject = meta["subject"]

        class _Task(ConfigurableTask):
            CONFIG = ConfigurableTask._config_cls.from_dict(
                _load_yaml_config(name, subject))
            # 使相对 !function 引用解析到 mmlu_prox/zh 目录
            _RESERVED = {"output_type", "fewshot_config", "metadata"}

            def __init__(self, *a, **kw):
                # 关键：强制生成路径走"关 thinking + 剥离 think"（与官方一致）
                kw["config"] = kw.get("config") or {}
                super().__init__(*a, **kw)
                self._model_args_hook = {"enable_thinking": False,
                                         "think_end_token": "</think>"}

        register(name, _Task)


def _load_yaml_config(name: str, subject: str) -> dict:
    import yaml as _yaml

    text = _NOTHINK_YAML.format(name=name)
    return _yaml.safe_load(text)


# 让 ConfigurableTask 能从 zh 目录解析 utils（!function utils.doc_to_text）
def _zh_dir() -> str:
    import os

    from lm_eval.tasks.mmlu_prox.zh import utils as u

    return os.path.dirname(u.__file__)


import sys  # noqa: E402


def _patch_utils_path():
    sys.path.insert(0, _zh_dir())


_patch_utils_path()


# =============================================================================
# 独立运行入口：直接在远端跑 mmlu_prox_zh_math / _other A/B 对照
#   A 组：默认模板（thinking 开）       B 组：enable_thinking=False + think_end_token
# 参数对齐 v0.0（limit=30, batch=8, max_gen_toks=2048, 5-shot），输出存 /tmp。
# 用法：
#   HF_ENDPOINT=https://hf-mirror.com python a_b_tasks.py
# =============================================================================
def _main() -> None:
    import json

    import torch
    from lm_eval import simple_evaluate

    N = int(os.environ.get("A_B_LIMIT", "30"))

    def run(tasks, extra):
        model_args = (f"pretrained={os.environ.get('MODEL_PATH', '/data/yucheng/madm-llm/models/Qwen3.5-4B')},"
                      "dtype=bfloat16")
        if extra:
            model_args += "," + extra
        if os.environ.get("LORA_ADAPTER"):
            model_args += f",peft={os.environ['LORA_ADAPTER']}"
        res = simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=tasks,
            batch_size=8,
            device="cuda",
            apply_chat_template=True,
            limit=N,
            gen_kwargs={"max_gen_toks": 2048, "temperature": 0.0},
            random_seed=42,
            numpy_random_seed=42,
            torch_random_seed=42,
            fewshot_random_seed=42,
        )
        out = {}
        for task in tasks:
            tr = (res or {}).get("results", {}).get(task) or {}
            out[task] = {k: v for k, v in tr.items() if isinstance(v, (int, float))}
        return out

    base = [t.strip() for t in
            os.environ.get("A_B_TASKS",
                           "mmlu_prox_zh_math,mmlu_prox_zh_other").split(",")
            if t.strip()]
    group = os.environ.get("A_B_GROUP", "AB")  # A / B / AB，A 组已完成可只跑 B

    if group in ("A", "AB"):
        print(">>> A 组：默认模板（thinking 开）")
        a = run(base, "")
        print(json.dumps(a, ensure_ascii=False, indent=2))
    else:
        a = {}

    if group in ("B", "AB"):
        print(">>> B 组：enable_thinking=False + think_end_token=</think>")
        # 关键：任务名与 A 组相同（lm-eval 已注册），仅 model_args 不同
        b = run(base, "enable_thinking=False,think_end_token=</think>")
        print(json.dumps(b, ensure_ascii=False, indent=2))
    else:
        b = {}

    if a and b:
        print(">>> 对照汇总")
        for name in base:
            ma = a.get(name, {})
            mb = b.get(name, {})
            ea = ma.get("exact_match,custom-extract")
            eb = mb.get("exact_match,custom-extract")
            print(f"  {name}: A={ea}  B={eb}")


if __name__ == "__main__":
    _main()
