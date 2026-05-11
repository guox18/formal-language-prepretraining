#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/tokenized}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/outputs/main_c4_10k}"
MODEL_NAME="${MODEL_NAME:-EleutherAI/pythia-160m}"
METHODS="${METHODS:-baseline rnn shuffdyck}"
SEEDS="${SEEDS:-3407 3408 3409}"
NOISE_PCTS="${NOISE_PCTS:-0 10 30 50}"
GPUS="${GPUS:-0 1 2 3}"
MIX_NAME="${MIX_NAME:-vocabfull}"

BSZ="${BSZ:-32}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
PPT_STEPS="${PPT_STEPS:-500}"
PT_STEPS="${PT_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
LR="${LR:-7e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
MIN_LR_RATE="${MIN_LR_RATE:-1}"
EVAL_STEPS="${EVAL_STEPS:-100}"
EVAL_SAMPLES="${EVAL_SAMPLES:-10000}"
REPORT_TO="${REPORT_TO:-none}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
RESUME="${RESUME:-1}"

mkdir -p "$OUT_DIR/runs" "$OUT_DIR/logs" "$OUT_DIR/metrics"

read -r -a GPU_LIST <<< "$GPUS"
if (( ${#GPU_LIST[@]} == 0 )); then
  echo "GPUS is empty"
  exit 1
fi

eval_data="$DATA_DIR/c4_val"
if [ ! -d "$eval_data" ]; then
  echo "Missing eval data: $eval_data"
  exit 1
fi

pt_data_for_noise() {
  local noise="$1"
  printf "%s/ptmix_%s_p%03d" "$DATA_DIR" "$MIX_NAME" "$noise"
}

ppt_data_for_method() {
  local method="$1"
  case "$method" in
    rnn) echo "$DATA_DIR/rnn_m1000" ;;
    shuffdyck) echo "$DATA_DIR/shuffdyck_d8" ;;
    c4ppt) echo "$DATA_DIR/c4_train" ;;
    *) return 1 ;;
  esac
}

run_train() {
  local gpu="$1"
  shift
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m prepretrain.train "$@"
}

run_ppt() {
  local gpu="$1"
  local method="$2"
  local seed="$3"
  local ppt_data
  ppt_data="$(ppt_data_for_method "$method")"
  local out="$OUT_DIR/runs/${method}_sd${seed}_ppt"
  local log="$OUT_DIR/logs/${method}_sd${seed}_ppt.log"

  if [ ! -d "$ppt_data" ]; then
    echo "Missing PPT data for $method: $ppt_data"
    return 1
  fi
  if [[ "$RESUME" == "1" ]] && [ -f "$out/final/model.safetensors" ]; then
    echo "[skip] $method seed=$seed PPT"
    return 0
  fi

  echo "[run] $method seed=$seed PPT on gpu=$gpu"
  run_train "$gpu" \
    --model_name "$MODEL_NAME" \
    --data_dir "$ppt_data" \
    --reinit True \
    --max_steps "$PPT_STEPS" \
    --save_steps "$PPT_STEPS" \
    --bsz "$BSZ" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --warmup_steps "$WARMUP_STEPS" \
    --lr "$LR" \
    --min_lr_rate "$MIN_LR_RATE" \
    --eval_data_dir "$eval_data" \
    --eval_steps "$EVAL_STEPS" \
    --eval_samples "$EVAL_SAMPLES" \
    --output_dir "$out" \
    --seed "$seed" \
    --report_to "$REPORT_TO" \
    --packing_mode pack \
    --attn_implementation "$ATTN_IMPLEMENTATION" \
    > "$log" 2>&1
}

run_pt() {
  local gpu="$1"
  local method="$2"
  local seed="$3"
  local noise="$4"
  local pt_data
  pt_data="$(pt_data_for_noise "$noise")"
  local noise_tag
  printf -v noise_tag "%03d" "$noise"
  local out="$OUT_DIR/runs/${method}_n${noise_tag}_sd${seed}_pt"
  local log="$OUT_DIR/logs/${method}_n${noise_tag}_sd${seed}_pt.log"

  if [ ! -d "$pt_data" ]; then
    echo "Missing PT data for noise=$noise: $pt_data"
    return 1
  fi
  if [[ "$RESUME" == "1" ]] && [ -f "$out/final/model.safetensors" ]; then
    echo "[skip] $method noise=$noise seed=$seed PT"
    return 0
  fi

  echo "[run] $method noise=$noise seed=$seed PT on gpu=$gpu"
  if [[ "$method" == "baseline" ]]; then
    run_train "$gpu" \
      --model_name "$MODEL_NAME" \
      --data_dir "$pt_data" \
      --reinit True \
      --max_steps "$PT_STEPS" \
      --save_steps "$SAVE_STEPS" \
      --bsz "$BSZ" \
      --max_seq_length "$MAX_SEQ_LENGTH" \
      --warmup_steps "$WARMUP_STEPS" \
      --lr "$LR" \
      --min_lr_rate "$MIN_LR_RATE" \
      --eval_data_dir "$eval_data" \
      --eval_steps "$EVAL_STEPS" \
      --eval_samples "$EVAL_SAMPLES" \
      --output_dir "$out" \
      --seed "$seed" \
      --report_to "$REPORT_TO" \
      --packing_mode pack \
      --attn_implementation "$ATTN_IMPLEMENTATION" \
      > "$log" 2>&1
  else
    local ppt_out="$OUT_DIR/runs/${method}_sd${seed}_ppt/final"
    if [ ! -f "$ppt_out/model.safetensors" ]; then
      echo "Missing PPT checkpoint for $method seed=$seed: $ppt_out"
      return 1
    fi
    run_train "$gpu" \
      --model_name "$MODEL_NAME" \
      --model_path "$ppt_out" \
      --data_dir "$pt_data" \
      --max_steps "$PT_STEPS" \
      --save_steps "$SAVE_STEPS" \
      --bsz "$BSZ" \
      --max_seq_length "$MAX_SEQ_LENGTH" \
      --warmup_steps "$WARMUP_STEPS" \
      --lr "$LR" \
      --min_lr_rate "$MIN_LR_RATE" \
      --eval_data_dir "$eval_data" \
      --eval_steps "$EVAL_STEPS" \
      --eval_samples "$EVAL_SAMPLES" \
      --output_dir "$out" \
      --seed "$seed" \
      --report_to "$REPORT_TO" \
      --packing_mode pack \
      --attn_implementation "$ATTN_IMPLEMENTATION" \
      > "$log" 2>&1
  fi
}

run_batch() {
  local -n task_ref=$1
  local label="$2"
  local idx=0
  local failed=0
  local total=${#task_ref[@]}
  while (( idx < total )); do
    local pids=()
    local batch_start=$idx
    for gpu in "${GPU_LIST[@]}"; do
      if (( idx >= total )); then
        break
      fi
      IFS=' ' read -r kind method seed noise <<< "${task_ref[$idx]}"
      if [[ "$kind" == "ppt" ]]; then
        ( run_ppt "$gpu" "$method" "$seed" ) &
      else
        ( run_pt "$gpu" "$method" "$seed" "$noise" ) &
      fi
      pids+=("$!")
      idx=$((idx + 1))
    done
    echo "[batch:$label] launched tasks $batch_start..$((idx - 1)); waiting"
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        failed=1
      fi
    done
    if (( failed != 0 )); then
      echo "[batch:$label] failed; see $OUT_DIR/logs"
      return 1
    fi
  done
}

ppt_tasks=()
pt_tasks=()
for seed in $SEEDS; do
  for method in $METHODS; do
    if [[ "$method" != "baseline" ]]; then
      ppt_tasks+=("ppt $method $seed 0")
    fi
  done
done

for seed in $SEEDS; do
  for noise in $NOISE_PCTS; do
    for method in $METHODS; do
      pt_tasks+=("pt $method $seed $noise")
    done
  done
done

echo "===== main C4 10k parallel run ====="
echo "OUT_DIR=$OUT_DIR"
echo "MODEL_NAME=$MODEL_NAME"
echo "SEEDS=$SEEDS"
echo "NOISE_PCTS=$NOISE_PCTS"
echo "METHODS=$METHODS"
echo "GPUS=$GPUS"
echo "PPT tasks=${#ppt_tasks[@]} PT tasks=${#pt_tasks[@]}"

run_batch ppt_tasks ppt
run_batch pt_tasks pt

"$PYTHON" "$ROOT_DIR/scripts/analyze_main_results.py" --runs-dir "$OUT_DIR/runs" --out-dir "$OUT_DIR/metrics"
