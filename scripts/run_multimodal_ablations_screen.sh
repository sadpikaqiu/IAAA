#!/usr/bin/env bash
# Invoke inside screen; the Python runner locks, validates, and resumes its output.
set -euo pipefail
EXPERIMENT_ROOT="${1:-/home/yzj/IAAA/outputs/experiments/mm_ablation_20260917}"
SOURCE_EXPERIMENT="${2:-/home/yzj/IAAA/outputs/experiments/mm_pipeline_20260914}"
DATA_ROOT="${3:-/home/yzj/IAAA/datasets}"
source /home/yzj/miniconda3/etc/profile.d/conda.sh
conda activate iaaa
cd "$EXPERIMENT_ROOT/code"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec >> "$EXPERIMENT_ROOT/runner.log" 2>&1
date --iso-8601=seconds
python scripts/run_multimodal_ablations.py \
  --source-experiment "$SOURCE_EXPERIMENT" \
  --data-root "$DATA_ROOT" \
  --output-dir "$EXPERIMENT_ROOT/results" \
  --cities NYC TKY --sample-size 500 --repeat-size 100 --concurrency 4 --pilot-size 4
