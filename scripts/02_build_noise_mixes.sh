#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
TOKENIZER="${TOKENIZER:-EleutherAI/pythia-160m}"
TOK_DIR="${TOK_DIR:-$ROOT_DIR/data/tokenized}"

NOISE_ROWS="${NOISE_ROWS:-260000}"
NOISE_PCTS_CSV="${NOISE_PCTS_CSV:-0,10,30,50}"
BSZ="${BSZ:-32}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
PT_STEPS="${PT_STEPS:-10000}"
MIX_NAME="${MIX_NAME:-vocabfull}"

NOISE_DATA="$TOK_DIR/noise_uniform_vocab"
PT_DATA="$TOK_DIR/c4_train_skip1m_ep1_10k"

if [ ! -d "$PT_DATA" ]; then
  echo "Missing PT data: $PT_DATA"
  echo "Run scripts/00_prepare_c4.sh first."
  exit 1
fi

echo "[noise] Building uniform vocabulary noise"
"$PYTHON" "$ROOT_DIR/scripts/build_uniform_noise_dataset.py" \
  --output "$NOISE_DATA" \
  --tokenizer "$TOKENIZER" \
  --rows "$NOISE_ROWS" \
  --seq-length "$MAX_SEQ_LENGTH"

echo "[noise] Building C4+noise PT mixes"
"$PYTHON" "$ROOT_DIR/scripts/build_pt_token_budget_mixes.py" \
  --clean-data "$PT_DATA" \
  --noise-data "$NOISE_DATA" \
  --output-dir "$TOK_DIR" \
  --mix-name "$MIX_NAME" \
  --noise-pcts "$NOISE_PCTS_CSV" \
  --base-pt-steps "$PT_STEPS" \
  --bsz "$BSZ" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --allow-noise-replacement
