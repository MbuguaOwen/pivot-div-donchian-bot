@echo off
setlocal
if not exist .venv (
  echo Create venv first: python -m venv .venv
  exit /b 1
)
call .venv\Scripts\activate
python -m div_donchian_bot.cli --config configs\default.yaml
