# Pivot Divergence + Donchian Bot (15m) — Binance Execution + Telegram Alerts

This project turns your TradingView Pine logic (“Div + Donchian”) into a real-time trading bot:
- **Signals** on **15-minute closed candles** (no intrabar repainting).
- **Telegram alert on every entry event** (even in paper mode).
- Optional **CVD divergence** (from Binance `aggTrade` stream) as a drop-in replacement for the Pine “pressure oscillator”.
- Optional **ATR-based SL/TP** per symbol (**disabled by default**).
- **Config-driven** symbol universe + per-pair overrides.

## What this implements (matches Pine semantics)
**Entry fires only when a pivot is confirmed**, i.e., `pivotLen` bars after the actual pivot candle.
That is the same “no future leakage” behavior as `ta.pivotlow/high()` used in your Pine indicator.

## Quick start
### 1) Create venv + install
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
python -m pip install -e .
```
This editable install makes the `src` layout importable so `python -m div_donchian_bot.cli ...` works from the repo root. If you cannot install locally, set `PYTHONPATH=src` when running commands.

### 2) Put secrets in `.env` (do NOT commit)
Copy and edit:
```bash
cp .env.example .env
```

### 3) Run in paper mode first
Set `run.mode: paper` in `configs/default.yaml` before this step if you want a dry run (default in this repo is live).
```bash
python -m div_donchian_bot.cli --config configs/default.yaml
```

### 4) Enable live trading
In `configs/default.yaml` set:
```yaml
run:
  mode: live   # default is live here; set to paper if you want a dry run
```
Then run the same command.

## Running live on a VM (testnet=false)
- Set `run.mode: live` (default is live in this repo) and `exchange.testnet: false`.
- Populate `.env` with real Binance keys and Telegram bot/chat IDs if alerts are enabled.
- Auto-restart on crashes/disconnects:
  - bash (Linux/macOS/git-bash): `RESTART_DELAY=5 ./scripts/run_bot.sh configs/default.yaml`
  - PowerShell (Windows): `.\scripts\run_bot.ps1 -Config configs/default.yaml -PythonPath .\venv\Scripts\python.exe -RestartDelaySeconds 5`
- Stop with Ctrl+C; the watchdog loops will relaunch the bot on any exit.

## Binance mode
Default is **USDT-M Futures**.
- Use `BINANCE_TESTNET=true` for demo endpoint.
- If you want Spot, there’s a config toggle (see `exchange.market_type`).

## CVD divergence toggle
In `configs/default.yaml`:
```yaml
strategy:
  direction: both  # both | long_only | short_only
  divergence:
    mode: pine      # pine (default) | cvd | both | either
```
⚠️ CVD requires subscribing to **aggTrades**, which is bandwidth-heavy for a huge symbol set. The bot only opens aggTrade streams for symbols whose divergence mode needs it; prefer a curated symbol list when enabling CVD.

## ATR SL/TP (disabled by default)
In `configs/default.yaml`:
```yaml
risk:
  atr:
    enabled: false
    length: 14
    sl_mult: 2.0
    tp_mult: 3.0
```

When enabled, the bot places (Futures) protective orders:
- `STOP_MARKET` for SL
- `TAKE_PROFIT_MARKET` for TP

## Configuring many pairs
You can:
1) Provide an explicit list of symbols, OR
2) Use `universe.dynamic=true` and filter by base assets (e.g. BTC/ETH/SOL/etc.)

Per-symbol overrides live in `configs/pairs/<SYMBOL>.yaml`.
`universe.pair_overrides_dir` (default `configs/pairs`) controls where overrides are loaded from, even when using dynamic universes.
- Dynamic universes are capped via `universe.max_symbols` (default 20 here) to avoid runaway stream subscriptions. Keep `allow_bases` small or use explicit symbols if you need tighter control.
- Trade direction filter (under `strategy.direction`):
  - `both` (default): allow LONG and SHORT
  - `long_only`: block SHORT signals
  - `short_only`: block LONG signals

## Corporate Telegram alerts
- Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`.
- Config toggles in `configs/default.yaml`:
  - `telegram.enabled`: turn alerts on/off.
  - `telegram.parse_mode`: defaults to `HTML` (keeps formatting stable).
  - `telegram.branding`: optional title for headers/footers.
  - `telegram.heartbeat_enabled`: enable/disable heartbeat pings (defaults to true).
  - `telegram.heartbeat_seconds`: override heartbeat interval (falls back to `run.heartbeat_seconds`).
  - `telegram.controls_enabled`: enable inline buttons to switch direction at runtime (off by default; no polling otherwise).
  - `telegram.controls_allowed_chat_id`: restrict control clicks (defaults to `TELEGRAM_CHAT_ID`).
  - `telegram.controls_state_path`: file to persist last update id + direction to avoid replay on restart.
- When controls are enabled, the startup alert includes inline buttons (LONG ONLY / SHORT ONLY / BOTH); only the allowed chat_id is honored.
- Alert layout uses `<b>` + `<pre>` for aligned fields. Example (HTML rendered by Telegram):
  ```
  <b>📌 TRADE SIGNAL — ENTRY</b>
  <pre>
  SYM: BTCUSDT     | SIDE: LONG | TIME: 2025-12-26 10:15:00Z
  MODE: PAPER | TF: 15m | NET: n/a
  </pre>

  <b>Signal</b>
  <pre>
  Entry: 105432.10 | Pivot: 105120.00 | Slip: +29.6 bps
  DivMode: PINE | Pine: True | CVD: False
  PivotOsc: 12.345600
  Loc@Pivot: 0.072 | DonLen: 120 | PivotLen: 5 | ExtBand: 0.10
  </pre>
  ```

## Files you care about
- `src/div_donchian_bot/cli.py` — entry point
- `src/div_donchian_bot/engine.py` — orchestrator
- `src/div_donchian_bot/strategy/pivot_div_donchian.py` — the strategy logic
- `src/div_donchian_bot/binance/*` — REST + WebSocket (kline + aggTrade)
- `configs/default.yaml` — global defaults
- `configs/pairs/*.yaml` — pair overrides

## Safety switches
- `run.mode: paper | live`
- `position.one_position_per_symbol: true`
- `limits.max_positions_total`
- `limits.cooldown_minutes_per_symbol` (prevents signal spam)

## Tests
- Install dev deps: `python -m pip install -r requirements-dev.txt`
- Run: `pytest`

## Deployment (systemd template)
```
[Unit]
Description=Pivot Div Donchian Bot
After=network-online.target

[Service]
Type=simple
User=YOURUSER
WorkingDirectory=/opt/pivot-div-donchian-bot
EnvironmentFile=/opt/pivot-div-donchian-bot/.env
ExecStart=/opt/pivot-div-donchian-bot/.venv/bin/python -m div_donchian_bot.cli --config /opt/pivot-div-donchian-bot/configs/default.yaml
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```
Enable and start:
```
sudo systemctl daemon-reload
sudo systemctl enable pivot-div-donchian.service
sudo systemctl start pivot-div-donchian.service
sudo journalctl -u pivot-div-donchian.service -f --no-pager
```

## Troubleshooting
- `ModuleNotFoundError: No module named 'div_donchian_bot'`: run `python -m pip install -e .` to put the package on your `PYTHONPATH`, or set `PYTHONPATH=src` when invoking the CLI.
- `WS connection error (HTTP 404)`: update to the latest code so the WebSocket base URL is correct (no duplicated `/stream`), then rerun the bot.

## Disclaimer
This code is for research/education and can lose money. Start with paper, small size, and testnet.
