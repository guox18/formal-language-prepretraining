#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/outputs/main_c4_10k}"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run_main_c4_10k.log"

notify() {
  local summary="$1"
  if command -v ntfy >/dev/null 2>&1; then
    ntfy echo "$summary" || true
  else
    echo "ntfy not found; summary: $summary"
  fi
}

status=0
start_ts="$(date '+%F %T')"
{
  echo "Started main C4 10k release reproduction at $start_ts"
  echo "ROOT_DIR=$ROOT_DIR"
  echo "OUT_DIR=$OUT_DIR"
  PYTHON="${PYTHON:-$ROOT_DIR/../pre-pretraining/.venv/bin/python}" \
  GPUS="${GPUS:-0 1 2 3}" \
  OUT_DIR="$OUT_DIR" \
  METHODS="${METHODS:-baseline rnn shuffdyck}" \
  SEEDS="${SEEDS:-3407 3408 3409}" \
  NOISE_PCTS="${NOISE_PCTS:-0 10 30 50}" \
  REPORT_TO="${REPORT_TO:-none}" \
  "$ROOT_DIR/scripts/05_run_main_c4_parallel.sh"
} > "$RUN_LOG" 2>&1 || status=$?

if (( status == 0 )); then
  summary="formal-language-prepretraining 10k reproduction completed successfully. Results: $OUT_DIR/metrics/summary_by_method_noise.csv"
else
  summary="formal-language-prepretraining 10k reproduction FAILED with exit code $status. Log: $RUN_LOG"
fi

notify "$summary"
echo "$summary"
exit "$status"
