#!/usr/bin/env bash

# Simple watchdog to auto-restart the bot if it exits.
# Usage: ./scripts/run_bot.sh [config_path]

set -u
set -o pipefail

CFG="${1:-configs/default.yaml}"
RESTART_DELAY="${RESTART_DELAY:-5}"

# Pick a Python binary: prefer venvs, fall back to system python
if [ -z "${PY_BIN:-}" ]; then
  if [ -x "venv/bin/python" ]; then
    PY_BIN="venv/bin/python"
  elif [ -x ".venv/bin/python" ]; then
    PY_BIN=".venv/bin/python"
  elif [ -x "venv/Scripts/python.exe" ]; then
    PY_BIN="venv/Scripts/python.exe"
  elif [ -x ".venv/Scripts/python.exe" ]; then
    PY_BIN=".venv/Scripts/python.exe"
  else
    PY_BIN="python"
  fi
fi

while true; do
  echo "$(date -Is) Starting bot with config=${CFG}"
  "${PY_BIN}" -m div_donchian_bot.cli --config "${CFG}"
  rc=$?
  echo "$(date -Is) Bot exited with code=${rc}. Restarting in ${RESTART_DELAY}s..."
  sleep "${RESTART_DELAY}"
done
