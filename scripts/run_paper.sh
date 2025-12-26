#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
python -m div_donchian_bot.cli --config configs/default.yaml
