r"""评测调度接口（已接入 lm-eval 真实后端）。

- 主框架维度（knowledge / reasoning / instruction / multilingual）交由
  lm-eval 的 ``simple_evaluate`` 执行：一次性加载模型、批量跑全部任务；
  若整批失败，按维度降级重跑以隔离出错维度，避免一个坏任务拖垮全部。
- 长上下文（long_context）走独立接口（scripts/run_longbench.py），不强绑定主框架。
- 未安装 lm-eval 时抛出 :class:`NotConfiguredError`（明确的设置错误）。

>>> from src.config_loader import load_config
>>> cfg = load_config('configs/template.yaml')
>>> evaluate(cfg)  # doctest: +SKIP
"""
from __future__ import annotations

import inspect
from typing import Any, Callable

try:
    from src.config_loader import EvalConfig
    from src.result_manager import register_model, save_sample_payload, update_status
except ImportError:  # 兼容直接以脚本运行
    from config_loader import EvalConfig
    from result_manager import register_model, save_sample_payload, update_status

# 主框架负责的维度（long_context 走独立接口 run_longbench.py）
# longbench2 是 lm-eval 任务组（LongBench v2，模型卡 50.0），与 LongBench v1
# 无关，必须由主框架执行；放入 tasks.long_context 会被 run_longbench.py 的
# TASK_META 校验拒绝。
FRAMEWORK_DIMENSIONS = ("knowledge", "reasoning", "instruction", "longbench2", "code", "multilingual")


class NotConfiguredError(RuntimeError):
    """框架已就绪，但依赖（lm-eval）未安装。"""


def _model_args(cfg: EvalConfig) -> str:
    """由配置构造 harness ``model_args`` 字符串（pretrained / dtype / tokenizer 等）。"""
    parts = [f"pretrained={cfg.model.path}", f"dtype={cfg.model.dtype}"]
    if cfg.model.trust_remote_code:
        parts.append("trust_remote_code=True")
    if cfg.model.revision:
        parts.append(f"revision={cfg.model.revision}")
    if cfg.model.tokenizer_path:
        parts.append(f"tokenizer={cfg.model.tokenizer_path}")
    return ",".join(parts)


def _harness_params(simple_evaluate: Callable) -> set[str]:
    """获取 simple_evaluate 支持的关键字参数名（不同 harness 版本略有差异）。"""
    try:
        return set(inspect.signature(simple_evaluate).parameters)
    except (TypeError, ValueError):
        return set()


def _numeric_metrics(task_res: dict[str, Any]) -> dict[str, Any]:
    """保留任务结果中的数值指标（如 ``acc,none`` / ``exact_match,strict-match``）。"""
    out: dict[str, Any] = {}
    for k, v in task_res.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = round(float(v), 6)
    return out or dict(task_res)


def dispatch_main(tasks: list[str], cfg: EvalConfig) -> dict[str, Any]:
    """调用 lm-eval 执行一批任务，返回 ``{task: 指标}``。

    - 任务名须为 harness 支持的任务或组名（如 mmlu_pro / c_eval / gsm8k / ifeval / mgsm）
    - 未安装 harness 时抛出 :class:`NotConfiguredError`
    - 其余参数（batch / dtype / max_length / gen_kwargs / limit 等）全部取自配置
    """
    tasks = [t for t in tasks if t]
    if not tasks:
        return {}
    try:
        from lm_eval import simple_evaluate
    except Exception as exc:  # noqa: BLE001 - 任何导入失败都视为未安装
        raise NotConfiguredError(
            "未安装 lm-eval；请先在服务器运行: bash scripts/install_env.sh"
        ) from exc

    harness_params = _harness_params(simple_evaluate)
    kwargs: dict[str, Any] = {
        "model": "hf",
        "model_args": _model_args(cfg),
        "tasks": list(tasks),
        "batch_size": cfg.evaluation.batch_size,
        "device": cfg.evaluation.device,
        "apply_chat_template": cfg.evaluation.apply_chat_template,
        "log_samples": bool(cfg.output.keep_samples),
        "gen_kwargs": dict(cfg.evaluation.gen_kwargs or {}),
    }
    # num_fewshot 为 null 时交给各任务默认值（如 gsm8k 默认 5-shot），不覆盖
    if cfg.evaluation.num_fewshot is not None:
        kwargs["num_fewshot"] = cfg.evaluation.num_fewshot
    if "seed" in harness_params:
        kwargs["seed"] = cfg.evaluation.seed
    else:
        for seed_name in ("random_seed", "numpy_random_seed", "torch_random_seed", "fewshot_random_seed"):
            if seed_name in harness_params:
                kwargs[seed_name] = cfg.evaluation.seed
    # output.max_samples 作为每个任务的采样上限（用于冒烟/快速验证；完整评测置 null）
    if cfg.output.max_samples and cfg.output.max_samples > 0:
        kwargs["limit"] = cfg.output.max_samples
    if "max_length" in harness_params:
        kwargs["max_length"] = cfg.evaluation.max_length

    try:
        raw = simple_evaluate(**kwargs)
    except Exception as exc:
        raise RuntimeError(f"lm-eval 执行失败: {exc}") from exc

    raw_results = (raw or {}).get("results") or {}
    raw_samples = (raw or {}).get("samples") or {}

    out: dict[str, Any] = {}
    for task in tasks:
        tr = raw_results.get(task)
        if tr is None:
            out[task] = {"status": "missing",
                         "message": f"harness 结果中未找到任务 {task!r}（任务名可能不受支持）"}
            continue
        out[task] = _numeric_metrics(tr)
        samples = raw_samples.get(task)
        if samples and kwargs["log_samples"]:
            cap = cfg.output.max_samples or len(samples)
            payload = [
                {k: s for k, s in sample.items()
                 if k in ("input", "arguments", "target", "predicted", "resps")}
                for sample in list(samples[:cap])
            ]
            try:
                save_sample_payload(cfg, task, payload)
            except Exception as exc:  # 样本保存失败不影响主结果
                out[task]["_sample_error"] = str(exc)
    return out


def dispatch_long_context(cfg: EvalConfig, run_dir: str = "") -> dict[str, Any]:
    """长上下文独立接口（不绑定 lm-eval）。

    实际评测逻辑位于 src/longbench.py（数据加载 / 官方评分口径 / 独立模型加载）。
    - 未配置 long_context 任务时返回 skipped，不报错。
    - 单任务失败不影响其余任务。
    - 依赖缺失（torch/transformers/datasets/rouge-score）时抛出
      :class:`NotConfiguredError`，与主框架的未安装提示一致。
    """
    try:
        from src.longbench import run_longbench
    except Exception as exc:  # noqa: BLE001
        raise NotConfiguredError(
            "未安装 LongBench 评测依赖；请先在服务器运行: bash scripts/install_env.sh"
        ) from exc
    result = run_longbench(cfg)
    return result.get("dimensions", {}).get("long_context", {})


def evaluate(cfg: EvalConfig,
             runner: Callable[[list[str], "EvalConfig"], dict[str, Any]] = dispatch_main,
             lc_runner: Callable[["EvalConfig", str], dict[str, Any]] = dispatch_long_context,
             register: bool = True) -> dict[str, Any]:
    """评测任务调度：一次加载模型跑全部主框架任务，失败时按维度降级重跑。

    参数
    ----
    cfg :
        已校验的统一配置对象。
    runner :
        主框架执行器（默认 dispatch_main，可注入替身以便测试）。
    lc_runner :
        长上下文执行器（默认仅登记待跑任务）。
    register :
        是否在开始前登记版本状态。

    返回
    ----
    dict
        汇总的评测结果：``dimensions``（维度 -> 任务 -> 指标）与 ``errors``。
        registry 状态由本函数更新（completed / failed）。
    """
    if register:
        register_model(cfg, status="pending")

    tasks_by_dim: dict[str, list[str]] = {
        d: list(cfg.tasks.get(d, [])) for d in FRAMEWORK_DIMENSIONS}
    dim_of = {t: d for d, tasks in tasks_by_dim.items() for t in tasks}
    all_tasks = list(dim_of)

    results: dict[str, Any] = {
        "framework": cfg.evaluation.framework,
        "version": cfg.version,
        "dimensions": {},
        "errors": [],
    }
    errors = results["errors"]

    try:
        if all_tasks:
            try:
                per_task = runner(all_tasks, cfg) or {}
                for task, metrics in per_task.items():
                    results["dimensions"].setdefault(dim_of.get(task, "unknown"), {})[task] = metrics
            except NotConfiguredError:
                raise
            except Exception as exc:
                errors.append(f"全量批次失败，按维度降级重跑: {exc}")
                for dim, tasks in tasks_by_dim.items():
                    if not tasks:
                        continue
                    try:
                        per_dim = runner(tasks, cfg) or {}
                        for task, metrics in per_dim.items():
                            results["dimensions"].setdefault(dim, {})[task] = metrics
                    except NotConfiguredError:
                        raise
                    except Exception as dim_exc:
                        results["dimensions"][dim] = {"status": "error", "error": str(dim_exc)}
                        errors.append(f"{dim}: {dim_exc}")
        else:
            for dim in FRAMEWORK_DIMENSIONS:
                results["dimensions"][dim] = {}

        try:
            lc = lc_runner(cfg, "") or {}
        except NotConfiguredError:
            raise
        except Exception as exc:
            lc = {"status": "error", "error": str(exc)}
            errors.append(f"long_context: {exc}")
        results["dimensions"]["long_context"] = lc
    except NotConfiguredError:
        if register:
            update_status(cfg, "failed")
        raise

    if register:
        update_status(cfg, "failed" if errors else "completed")
    return results


if __name__ == "__main__":
    import doctest
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    doctest.testmod()
    print("evaluator OK")
