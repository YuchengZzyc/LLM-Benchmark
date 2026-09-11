# =============================================================================
# install_env.sh — 服务器一次性环境安装脚本（uv 版）
# =============================================================================
# 使用 uv 创建虚拟环境并安装依赖，替代 conda/系统 Python。
# 当前阶段为框架搭建：脚本已编写但**未执行**。
# 后续在 Linux 服务器上按 docs/environment_setup.md 调用。
#
# 用法（Linux/macOS 服务器）：
#   bash scripts/install_env.sh [--python 3.10] [--torch 2.3.0]
#
# 前置：已安装 uv（curl -LsSf https://astral.sh/uv/install.sh | sh）
# =============================================================================

set -euo pipefail

PYTHON_VERSION="3.10"
TORCH_VERSION="2.3.0"

# ---- 参数解析 ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) PYTHON_VERSION="$2"; shift 2 ;;
        --torch)  TORCH_VERSION="$2";  shift 2 ;;
        --help|-h)
            echo "Usage: bash $0 [--python 3.10] [--torch 2.3.0]"
            exit 0 ;;
        *) echo "未知参数: $1"; echo "Usage: bash $0 [--python 3.10] [--torch 2.3.0]"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

echo "==> Benchmark Framework 环境安装（uv）"
echo "    Python  : ${PYTHON_VERSION}"
echo "    Torch   : ${TORCH_VERSION}"
echo "    VENV    : ${VENV_DIR}"

# ---- 1. 检查 uv ----
if ! command -v uv >/dev/null 2>&1; then
    echo "错误: 未找到 uv，请先安装：curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

# ---- 2. 创建虚拟环境（uv 自动下载指定 Python 版本）----
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "==> 创建虚拟环境 ${VENV_DIR} (python ${PYTHON_VERSION})"
    uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
else
    echo "==> 虚拟环境已存在: ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# ---- 3. 安装 PyYAML（配置解析必需）----
uv pip install "PyYAML>=6.0.0"

# ---- 4. 安装 GPU 版 PyTorch（按服务器 CUDA 版本选择 index）----
echo "==> 安装 GPU 版 PyTorch ${TORCH_VERSION}"
echo "    提示: 请按服务器 nvidia-smi 的 CUDA 版本选择安装源:"
echo "      CUDA 12.1+: https://download.pytorch.org/whl/cu121"
echo "      CUDA 11.8 : https://download.pytorch.org/whl/cu118"
uv pip install "torch==${TORCH_VERSION}" \
    --index-url https://download.pytorch.org/whl/cu121

# ---- 5. 安装主框架与其余依赖 ----
uv pip install "lm-eval>=0.4.2,<0.7.0"
uv pip install -r "${ROOT_DIR}/requirements.txt"

# ---- 6. 校验安装 ----
python - <<'PY'
import sys
try:
    import torch
    print("torch:", torch.__version__,
          "| cuda available:", torch.cuda.is_available())
except ImportError:
    print("警告: torch 未安装成功", file=sys.stderr)
try:
    import lm_eval
    print("lm_eval:", getattr(lm_eval, "__version__", "unknown"))
except ImportError:
    print("警告: lm_eval 未安装成功", file=sys.stderr)
try:
    import yaml
    print("pyyaml:", yaml.__version__)
except ImportError:
    print("警告: PyYAML 未安装成功", file=sys.stderr)
try:
    import rouge_score
    print("rouge_score:", getattr(rouge_score, "__version__", "unknown"))
except ImportError:
    print("警告: rouge-score 未安装成功（LongBench 摘要任务将无法评分）",
          file=sys.stderr)
PY

echo "==> 安装完成。可运行:"
echo "    bash scripts/run_eval.sh --config configs/baseline_qwen35_4b.yaml"
echo "    python scripts/run_longbench.py --config configs/baseline_qwen35_4b.yaml"
