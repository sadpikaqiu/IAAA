#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
VLLM_VERSION="${VLLM_VERSION:-0.28.0}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found. Install or expose the NVIDIA driver first." >&2
  exit 1
fi

echo "== Python, GPU, and driver =="
"${PYTHON_BIN}" --version
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "== Install vLLM and model processor =="
PIP_DEFAULT_TIMEOUT=600 "${PYTHON_BIN}" -m pip install \
  --index-url "${PIP_INDEX_URL}" \
  --retries 10 \
  "vllm==${VLLM_VERSION}" \
  "transformers>=5.8.0"

echo "== Install IAA-Agent =="
"${PYTHON_BIN}" -m pip install --editable "${REPO_ROOT}"

echo "== Verify imports =="
"${PYTHON_BIN}" -m pip check
"${PYTHON_BIN}" -c \
  'import torch, transformers, vllm; assert torch.cuda.is_available(); print("torch", torch.__version__, "cuda", torch.version.cuda); print("gpu", torch.cuda.get_device_name(0)); print("transformers", transformers.__version__); print("vllm", vllm.__version__)'

echo
echo "Environment ready in ${CONDA_DEFAULT_ENV:-the active Python environment}."
echo "Start the server with: bash ${REPO_ROOT}/scripts/serve_qwen38_vllm.sh"
