#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/outputs/paperlike_1b_0_15}"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run_paperlike_1b_0_15.log"

notify() {
  local summary="$1"
  if command -v ntfy >/dev/null 2>&1; then
    ntfy echo "$summary" || true
  else
    echo "ntfy not found; summary: $summary"
  fi
}

status=0
{
  echo "Started paperlike 1B 0/15 reproduction at $(date '+%F %T')"
  echo "ROOT_DIR=$ROOT_DIR"
  echo "OUT_DIR=$OUT_DIR"
  set -a
  source "$ROOT_DIR/configs/paperlike_1b.env"
  set +a
  export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
  PYTHON_BIN="${PYTHON:-$ROOT_DIR/../pre-pretraining/.venv/bin/python}"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/07_run_paperlike_1b_0_15.py" \
    --python "$PYTHON_BIN" \
    --out-dir "$OUT_DIR" \
    --ppt-source-root "${PPT_SOURCE_ROOT:-}"
} > "$RUN_LOG" 2>&1 || status=$?

if (( status == 0 )); then
  summary="formal-language-prepretraining paperlike 1B 0/15 reproduction completed successfully. Results: $OUT_DIR/metrics/summary_1b_0_15.csv"
else
  summary="formal-language-prepretraining paperlike 1B 0/15 reproduction FAILED with exit code $status. Log: $RUN_LOG"
fi

notify "$summary"
echo "$summary"
exit "$status"
