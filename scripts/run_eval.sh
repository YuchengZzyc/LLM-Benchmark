#!/usr/bin/env bash
# =============================================================================
# run_eval.sh — 统一评测入口
# =============================================================================
# 用法：
#   bash scripts/run_eval.sh --config configs/baseline_qwen35_4b.yaml
# 流程：运行 lm-eval 评测 -> 生成 Markdown 报告。
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG=""

usage() {
    echo "Usage: bash $0 --config <config.yaml> [--no-report] [--keep-samples]"
    echo ""
    echo "Options:"
    echo "  --config <path>  评测配置文件（必须，位于 configs/）"
    echo "  --no-report      不生成 Markdown 报告（默认生成）"
    echo "  --keep-samples   保存样本输出到 results/<version>/samples/"
    echo "  --help           显示本帮助"
}

# ---- 参数解析 ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="${2:-}"
            shift 2 ;;
        --no-report)
            NO_REPORT="1"
            shift ;;
        --keep-samples)
            KEEP_SAMPLES="1"
            shift ;;
        --help|-h)
            usage
            exit 0 ;;
        *)
            echo "错误: 未知参数 $1" >&2
            usage
            exit 1 ;;
    esac
done

if [[ -z "${CONFIG}" ]]; then
    echo "错误: 缺少 --config 参数" >&2
    usage
    exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "错误: 配置文件不存在: ${CONFIG}" >&2
    exit 1
fi

echo "==> run_eval.sh"
echo "    ROOT    : ${ROOT_DIR}"
echo "    CONFIG  : ${CONFIG}"

# ---- 执行评测与报告生成（用 venv 内 python，环境由 install_env.sh 建立）----
VENV_PY="${ROOT_DIR}/../.venv/bin/python"
if [[ ! -x "${VENV_PY}" ]]; then
    echo "错误: 未找到虚拟环境 python: ${VENV_PY}" >&2
    echo "    请先运行: bash scripts/install_env.sh" >&2
    exit 1
fi

"${VENV_PY}" "${SCRIPT_DIR}/run_lm_eval.py" --config "${CONFIG}" \
    ${KEEP_SAMPLES:+--keep-samples}
if [[ -z "${NO_REPORT:-}" ]]; then
    "${VENV_PY}" "${SCRIPT_DIR}/generate_report.py" --config "${CONFIG}"
fi

echo "==> 评测与报告生成完成。"
