#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_ROOT="${1:-/home/yzj/IAAA/outputs/experiments/autonomous_v1_20260922_fix04}"
source /home/yzj/miniconda3/etc/profile.d/conda.sh
conda activate iaaa
PYTHON_BIN="$(command -v python)"
cd "$EXPERIMENT_ROOT/code"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec > >(tee -a "$EXPERIMENT_ROOT/runner.log") 2>&1
date --iso-8601=seconds
# The runner prints only when all arms of a session finish. Show its existing
# progress.json in a separate window so long sessions still have visible updates.
MONITOR="$EXPERIMENT_ROOT/code/scripts/watch_autonomous.py"
if [[ -n "${STY:-}" && -f "$MONITOR" ]] && command -v screen >/dev/null 2>&1; then
  screen -S "$STY" -p "${WINDOW:-0}" -X title runner || true
  # A missing -p window can still make `screen -Q title` return success.
  # Inspect the actual window list instead of trusting that exit status.
  if [[ "$(screen -S "$STY" -Q windows)" != *" progress"* ]]; then
    # New windows inherit the screen server's pre-conda PATH, not this shell's.
    screen -S "$STY" -X screen -t progress "$PYTHON_BIN" -u "$MONITOR" "$EXPERIMENT_ROOT" --interval 10 || true
  fi
  screen -S "$STY" -X select progress || true
fi
"$PYTHON_BIN" scripts/evaluate_autonomous.py \
  --data-root /home/yzj/IAAA/datasets \
  --source-experiment /home/yzj/IAAA/outputs/experiments/mm_pipeline_20260914 \
  --source-ablation /home/yzj/IAAA/outputs/experiments/mm_ablation_20260917 \
  --baseline-repair-source /home/yzj/IAAA/outputs/experiments/autonomous_v1_20260918/results \
  --output-dir "$EXPERIMENT_ROOT/results" \
  --tokenizer-path /home/yzj/Model/Qwen38-27B \
  --cities NYC TKY --development-size 50 --validation-size 500 --repeat-size 100 \
  --concurrency 16 --full
