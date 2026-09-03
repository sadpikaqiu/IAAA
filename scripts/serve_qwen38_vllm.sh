#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${QWEN_MODEL_PATH:-${HOME}/Model/Qwen38-27B}"
SERVED_MODEL_NAME="${QWEN_SERVED_MODEL_NAME:-Qwen/Qwen3.8-27B-FP8}"
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.60}"
VLLM_BIN="${VLLM_BIN:-$(command -v vllm || true)}"

if [[ -z "${VLLM_BIN}" || ! -x "${VLLM_BIN}" ]]; then
  echo "vLLM is not available in the active environment. Run scripts/setup_qwen38_vllm.sh first." >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Model checkpoint was not found at ${MODEL_PATH}. Set QWEN_MODEL_PATH." >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"

exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enable-prefix-caching \
  --generation-config vllm \
  --default-chat-template-kwargs.enable_thinking false \
  --default-chat-template-kwargs.preserve_thinking false \
  --language-model-only \
  --performance-mode throughput
