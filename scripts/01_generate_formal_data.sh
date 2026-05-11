#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

TOKENIZER="${TOKENIZER:-EleutherAI/pythia-160m}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data}"
RAW_DIR="$DATA_DIR/raw"
TOK_DIR="$DATA_DIR/tokenized"

MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
RNN_NUM_MODELS="${RNN_NUM_MODELS:-1000}"
RNN_NUM_SEQS="${RNN_NUM_SEQS:-100}"
RNN_HIDDEN_SIZE="${RNN_HIDDEN_SIZE:-64}"
RNN_BATCH_SIZE="${RNN_BATCH_SIZE:-500}"
RNN_MODEL_SEED="${RNN_MODEL_SEED:-300}"
RNN_SEQUENCE_SEED="${RNN_SEQUENCE_SEED:-4000}"
FORMAL_N="${FORMAL_N:-100000}"

mkdir -p "$RAW_DIR" "$TOK_DIR"

echo "[formal] Generating RNN token-id data"
"$PYTHON" -m prepretrain.formal.linear_rnn.cli generate \
  --num-models "$RNN_NUM_MODELS" \
  --num-seqs "$RNN_NUM_SEQS" \
  --hidden-size "$RNN_HIDDEN_SIZE" \
  --seq-length "$MAX_SEQ_LENGTH" \
  --batch-size "$RNN_BATCH_SIZE" \
  --tokenizer-name "$TOKENIZER" \
  --dtype float64 \
  --model-seed "$RNN_MODEL_SEED" \
  --sequence-seed "$RNN_SEQUENCE_SEED" \
  --out "$RAW_DIR/rnn_m1000.txt" \
  --format txt

echo "[formal] Generating Dyck token-id data"
"$PYTHON" -m prepretrain.formal.data_utils generate_shuff_dyck \
  --out_path "$RAW_DIR/shuffdyck_d8.txt" \
  --num_symbols 64 \
  --n "$FORMAL_N" \
  --target_length "$MAX_SEQ_LENGTH" \
  --p 0.51 \
  --min_depth 1 \
  --max_depth 8

echo "[formal] Converting to HuggingFace datasets"
"$PYTHON" "$ROOT_DIR/scripts/convert_to_dataset.py" \
  --input "$RAW_DIR/rnn_m1000.txt" \
  --output "$TOK_DIR/rnn_m1000" \
  --input_format int

"$PYTHON" "$ROOT_DIR/scripts/convert_to_dataset.py" \
  --input "$RAW_DIR/shuffdyck_d8.txt" \
  --output "$TOK_DIR/shuffdyck_d8" \
  --input_format int
