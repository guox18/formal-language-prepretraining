#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

set -a
source "$ROOT_DIR/configs/smoke.env"
set +a

"$PYTHON" "$ROOT_DIR/scripts/make_tiny_data.py" \
  --output-root "$ROOT_DIR/data/tokenized" \
  --tokenizer "$TOKENIZER" \
  --seq-length "$MAX_SEQ_LENGTH" \
  --overwrite

ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}" \
DATA_DIR="$ROOT_DIR/data/tokenized" \
OUT_DIR="$ROOT_DIR/outputs/smoke" \
MODEL_NAME="$MODEL_NAME" \
SEEDS="$SEEDS" \
NOISE_PCTS="$NOISE_PCTS" \
METHODS="$METHODS" \
BSZ="$BSZ" \
MAX_SEQ_LENGTH="$MAX_SEQ_LENGTH" \
PPT_STEPS="$PPT_STEPS" \
PT_STEPS="$PT_STEPS" \
SAVE_STEPS="$SAVE_STEPS" \
LR="$LR" \
WARMUP_STEPS="$WARMUP_STEPS" \
MIN_LR_RATE="$MIN_LR_RATE" \
EVAL_STEPS="$EVAL_STEPS" \
EVAL_SAMPLES="$EVAL_SAMPLES" \
REPORT_TO="$REPORT_TO" \
GPUS="$GPUS" \
"$ROOT_DIR/scripts/03_run_main_c4.sh"
