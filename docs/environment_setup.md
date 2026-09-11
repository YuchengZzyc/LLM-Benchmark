# 环境搭建说明（Environment Setup）

> 当前阶段：**环境已在评测服务器搭建并验证**（2026-09-10），实测版本见
> [environment.txt](../environment.txt) 与本文 §6。

本文档说明如何在 **Linux 评测服务器**上搭建本框架所需环境。
开发机（Windows）仅用于编写配置与代码，不参与评测。

---

## 1. Python 环境

- 本框架使用 **uv** 管理 Python 环境（不依赖 conda / 系统 Python）。
- 推荐 **Python 3.10**（lm-eval 与当前生态兼容性最佳），
  3.9–3.12 亦可。uv 会自动下载指定版本的 Python。
- **本服务器实际落地为 Python 3.11.16**（见 [environment.txt](../environment.txt)）。
  已知副作用：`humaneval` 系列任务需要 `code_eval` 依赖，而该依赖要求
  **Python ≥ 3.12**，故在 3.11 下代码维度不可跑。

```bash
# 安装 uv（若未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境（uv 自动获取 Python 3.10）
cd <repo>/Benchmark
uv venv --python 3.10 .venv
source .venv/bin/activate
```

> **服务器踩坑**：venv 内**没有 `pip` 模块**，且非交互 SSH 的 `PATH` 不含 uv，
> 装包必须写全路径并显式指定解释器：
> `/data/yucheng/.local/bin/uv pip install --python .venv/bin/python <pkg>`

> Windows 开发机若需本地运行 Python 脚本，同样使用 uv：
> `uv venv --python 3.10 .venv` 后 `source .venv/Scripts/activate` 即可。

## 2. CUDA 环境

评测需要 GPU。请在服务器上确认：

```bash
nvidia-smi                     # 查看 GPU 型号与 Driver/CUDA 版本
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

- 需要 **NVIDIA GPU + 足够显存**（Qwen3.5-4B 以 bfloat16 评测，单卡显存建议 ≥ 24GB）。
- PyTorch 的 CUDA 版本需与 **Driver 支持的 CUDA** 匹配（向下兼容）。
- 若 torch 无法识别 GPU，先安装匹配的 GPU 版 PyTorch 再继续。

## 3. 依赖安装

依赖声明见 [requirements.txt](../requirements.txt)。推荐顺序：

### 3.1 GPU 版 PyTorch（按服务器 CUDA 版本选择 index，经 uv 安装）

```bash
# CUDA 12.1+：
uv pip install "torch>=2.1.0" --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8：
# uv pip install "torch>=2.1.0" --index-url https://download.pytorch.org/whl/cu118
```

> 注意：**不要**直接用 `uv pip install -r requirements.txt`，因为其中 torch 被注释，
> 需按 CUDA 版本单独安装。

### 3.2 主框架与其余依赖

```bash
uv pip install "lm-eval>=0.4.2,<0.7.0"
uv pip install -r requirements.txt
```

> `requirements.txt` 同时声明 LongBench 独立接口的评分依赖
> （`rouge-score`：摘要任务 ROUGE-L）。长上下文数据原设计从 HuggingFace
> （`THUDM/LongBench`）下载，但**本服务器 `datasets` 为 4.0.0，已移除数据集脚本
> 加载器，该在线路径不可用**；须预先在服务器放好官方 JSONL 并设置
> `evaluation.longbench_data_dir`（见 §5.1）。

### 3.3 一键脚本（推荐）

以上步骤已封装为 [scripts/install_env.sh](../scripts/install_env.sh)（基于 uv）：

```bash
bash scripts/install_env.sh [--python 3.10] [--torch 2.3.0]
```

## 4. 安装验证

```bash
python - <<'PY'
import torch, lm_eval, yaml, rouge_score, datasets, transformers, accelerate
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
print("lm_eval:", getattr(lm_eval, "__version__", "unknown"))
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
print("datasets:", datasets.__version__)   # 4.x：无脚本加载器，见 §5
print("pyyaml:", yaml.__version__)
print("rouge_score:", getattr(rouge_score, "__version__", "unknown"))
PY
```

真实版本信息已回填至 [environment.txt](../environment.txt)（2026-09-10 实测）。

> **ifeval 额外依赖**：`langdetect`、`immutabledict` 未随 lm-eval 自动安装，
> 需单独补装，否则 ifeval 任务会报 ImportError。

## 5. 常见问题

| 问题 | 处理 |
| --- | --- |
| `torch.cuda.is_available()` 为 False | 重装匹配 CUDA 的 GPU 版 torch；检查 `nvidia-smi` |
| lm-eval 依赖冲突 | 单独升级/降级对应包；必要时在隔离 venv 中重装 |
| benchmark 数据集首次下载失败 | 检查网络；harness 会自动从 HF Hub 下载（需可访问 HF） |
| `huggingface.co` 不可达（Errno 101） | 命令前加 `export HF_ENDPOINT=https://hf-mirror.com`（**实测必需**） |
| `Dataset scripts are no longer supported, but found LongBench.py` | `datasets` **4.0.0 已移除数据集脚本加载支持**，`THUDM/LongBench` 走不通 HF 路线；须改用官方 JSONL 本地直读（见下） |
| 显存不足 / OOM | 调小 `evaluation.batch_size`（**实测 16 在 multilingual 维度 OOM，8 稳定**）或改 dtype 为 float16 |
| venv 内无 `pip`，`uv` 不在 PATH | 用 `/data/yucheng/.local/bin/uv pip install --python .venv/bin/python <pkg>` |
| `simple_evaluate() got an unexpected keyword argument 'seed'` | lm-eval 0.4.13 无 `seed` 参数，改用 `random_seed`/`numpy_random_seed`/`torch_random_seed`/`fewshot_random_seed`（`src/evaluator.py` 已内省适配） |

### 5.1 LongBench v1 数据的本地化（datasets 4.x 必需）

`THUDM/LongBench` 依赖 `LongBench.py` 脚本加载器，`datasets 4.0.0` 已拒绝，
故长上下文 v1 **不能**再用 `evaluation.longbench_data_dir: null`（走 HF 下载）。
做法：手动取官方 JSONL 放入本地目录，并在配置中指定路径：

```bash
# 目录约定（实测位置）
mkdir -p data/longbench
# 把 <task>.jsonl 放入其中，如 multifieldqa_en.jsonl / hotpotqa.jsonl / trec.jsonl
```

```yaml
evaluation:
  longbench_data_dir: data/longbench   # 非 null = 本地 JSONL 直读，绕开数据集脚本
```

## 6. 实测环境快照（2026-09-10）

| 项 | 值 |
| --- | --- |
| OS | Ubuntu 26.04 LTS (kernel 7.0.0-29-generic) |
| CPU / RAM | AMD Ryzen 9 9950X 16-Core (32 线程) / 59 GB |
| GPU | NVIDIA GeForce RTX 5090 32GB |
| Driver / CUDA | 595.84 / 13.2（torch 编译于 cu128） |
| Python | 3.11.16 |
| torch | 2.11.0+cu128（`cuda.is_available() == True`） |
| transformers / accelerate / datasets | 5.6.0 / 1.11.0 / 4.0.0 |
| lm-eval | 0.4.13 |
| rouge-score | 0.1.2 |

## 7. 运行前 Checklist

- [ ] `model.path` 已填为真实模型路径（本地目录或 HF 仓库名）
- [ ] `torch.cuda.is_available()` 为 True
- [ ] `uv pip show lm-eval` 有输出（用 uv 全路径）
- [ ] `python -c "import yaml"` 无报错
- [ ] `python -c "import rouge_score"` 无报错（LongBench 摘要任务评分依赖）
- [ ] LongBench v1 数据可访问：`datasets` 4.0.0 已不支持数据集脚本，
      须配置 `evaluation.longbench_data_dir`（如 `data/longbench`）指向本地官方
      JSONL，**不能**再依赖 HF 在线加载 `THUDM/LongBench`
- [ ] `batch_size` 已按显存调整（5090 32GB 实测 `8`；`16` 在 multilingual 维度 OOM）
