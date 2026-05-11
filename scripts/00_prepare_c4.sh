#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
TOKENIZER="${TOKENIZER:-EleutherAI/pythia-160m}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/tokenized}"

C4_PPT_EXAMPLES="${C4_PPT_EXAMPLES:-100000}"
C4_VAL_EXAMPLES="${C4_VAL_EXAMPLES:-10000}"
C4_SKIP_DOCS="${C4_SKIP_DOCS:-1000000}"
C4_PT_TARGET_STEPS="${C4_PT_TARGET_STEPS:-10000}"
C4_PT_CAP_STEPS="${C4_PT_CAP_STEPS:-10400}"
BSZ="${BSZ:-32}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"

mkdir -p "$DATA_DIR"

echo "[c4] Building canonical c4_train for optional C4-PPT and c4_val for eval"
"$PYTHON" "$ROOT_DIR/scripts/prepare_c4.py" \
  --output_dir "$DATA_DIR" \
  --num_examples "$C4_PPT_EXAMPLES" \
  --val_size "$C4_VAL_EXAMPLES" \
  --tokenizer "$TOKENIZER" \
  --max_length "$MAX_SEQ_LENGTH"

echo "[c4] Building canonical skip split for PT"
"$PYTHON" "$ROOT_DIR/scripts/prepare_c4_skip_train_steps.py" \
  --output-root "$DATA_DIR" \
  --output-name c4_train_skip1m_ep1_10k \
  --skip-docs "$C4_SKIP_DOCS" \
  --target-steps "$C4_PT_TARGET_STEPS" \
  --cap-steps "$C4_PT_CAP_STEPS" \
  --tokenizer "$TOKENIZER" \
  --max-length "$MAX_SEQ_LENGTH" \
  --bsz "$BSZ"
