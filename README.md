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

## CVD pressure confirmation (signed delta filter)

This is **not divergence**. It’s a microstructure **directional pressure gate** computed from Binance `aggTrade`:

- `delta_1m = buy_qty - sell_qty` per minute (aligned to Binance minute close `...:59.999`).
- `delta_15m = sum(delta_1m)` over the last 15 minutes ending at `confirm_time_ms`.
- `signed_delta_15m = delta_15m` for LONG, `-delta_15m` for SHORT.

Filter rule (default):
- LONG: `delta_15m >= +50`
- SHORT: `delta_15m <= -50`

Config (top-level `cvd:`):
```yaml
cvd:
  mode: "off"                 # off | shadow | filter
  delta_window_minutes: 15
  signed_delta_threshold: 50
  warmup_minutes: 60           # fail-closed in filter mode until warmup is satisfied
```

Modes:
- `off`: no computation, no gating.
- `shadow`: compute + log `CVD_FILTER,...` but **do not block** trades.
- `filter`: **block entries** unless the signed-delta threshold passes (and the store is warmed up).

Every candidate entry (BOT or TV) logs one line:
`CVD_FILTER,side=...,symbol=...,confirm_ms=...,delta_15m=...,signed=...,thresh=...,pass=...,...`
and a CSV is written to `artifacts/<run_id>/cvd_filter.csv` for reproducibility.

## HTF structure risk engine
Single risk engine (no separate ATR SL/TP module). In `configs/default.yaml`:
```yaml
risk:
  engine:
    enabled: true
    htf_interval: "1h"
    htf_lookback_bars: 72
    buffer_bps: 10
    atr_len: 24
    atr_floor_bps: 20
    k_init: 1.8
    tp_r_mult: 2.0
    be_trigger_r: 1.0
    be_buffer_bps: 10
    trailing:
      enabled: false
      trigger_r: 2.5
      k_trail: 1.6
      lock_r: 1.0
    min_stop_replace_interval_sec: 2
    min_stop_tick_improvement: 2
```
- SL0: structure low/high ± buffer, capped by `k_init * ATR(1h,24)` (closer stop wins).
- **ATR floor**: if the resulting `R = |entry − SL0|` is too small, it is expanded to `max(R, entry * atr_floor_bps / 10000)`.
- R = |entry − SL0|; TP = entry ± `tp_r_mult * R` (default 2R).
- Break-even at `be_trigger_r` with `be_buffer_bps` buffer; stop is monotonic.
- Trailing (optional): trigger at `trailing.trigger_r`, lock at `lock_r * R`, trail by `max(k_trail * ATR(1h,24), (trigger_r-lock_r)*R)`.
- Entries are refused until the HTF engine is warmed up; SL is placed before TP, and a failed SL placement triggers an immediate flatten.

## Execution modes (paper vs binance)
- Config block:
  ```yaml
  execution:
    mode: paper   # paper | binance
    paper:
      fee_bps: 4.0           # per-side fees
      slippage_bps: 1.5      # applied on fills/exits against you
      fill_policy: next_tick # next_tick | bar_close_fallback
      max_wait_ms: 3000      # fallback deadline for bar_close_fallback
      use_mark_price: false
      log_trades: true
  ```
- `paper`: uses live mainnet websockets, no signed REST, local fills (next aggTrade tick) with slippage/fees, internal SL/TP + stop-engine exits. No API keys required.
- `binance`: current live behavior (signed REST orders, positionRisk watchdog, stop cancel/replace throttled).
- Telegram execution messages show the EXEC mode and fill px/fees; exits in paper mode are announced as `(paper)`.

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

## TradingView Parity Mode
- Enable in `configs/default.yaml` (or per-env override):
  ```yaml
  parity:
    mode: true
    ema_seed_mode: "sma"        # "sma" matches Pine ta.ema seeding
    pivot_tie_break: "tv_like"  # deterministic flat-top/flat-bottom selection
    warmup_bars: 0              # 0 = auto-compute (don/pivot/ema safety margin)
    gap_heal: true              # REST backfill on missed klines
    desync_pause: true          # pause alerts + warn if a gap cannot be healed
  ```
- Matching TV chart: same symbol string (spot vs perp), identical timeframe, `fireOnClose=true` alerts only.
- Parity check harness (offline):
  ```bash
  python -m div_donchian_bot.cli.parity_check \
    --bars_csv artifacts/my_ohlcv.csv \
    --tv_signals_csv artifacts/tv_signals.csv \
    --config configs/default.yaml
  ```
  Output includes precision/recall, missing/extra signals, and the first mismatch timestamp.
- Exporting TV alerts: use your TradingView alert log export (CSV) with columns `symbol,side,confirm_time_ms` (millisecond close timestamp). For bars, use your own OHLCV export; the bot assumes close-time ordering and bar-close semantics only.
- Known limits: CVD parity is best-effort (depends on aggTrade availability); if REST backfill fails, the engine will pause alerts for that symbol until healed.
- In parity mode, cooldown is forced to 0 to avoid dropping valid TV-equivalent alerts.

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

## TradingView webhook bridge (TV = signal engine)

If you want *true production parity* with Pine, you can run a small webhook server **inside the bot** and let TradingView send the entry signals.

Modes (config: `tv_bridge.mode`):

- `tv_only` — **TV is boss**. Bot does risk + execution only. If TV says LONG, bot trades LONG (subject to safety gates).
- `tv_and_bot` — **two-man rule**. Bot trades only if TV signal and bot strategy signal share the same `(symbol, side, confirm_time_ms)`.
- Option A parity: `/tv` webhook is authoritative. `confirm_time_ms` is treated as the canonical bar close (Binance kline `T - 1ms`); entries are placed immediately on webhook receipt with SL/TP, and websocket klines are used only for trailing/BE/state—not to delay entry.

Config (see `configs/default.yaml`):

```yaml
tv_bridge:
  enabled: true
  host: "0.0.0.0"
  port: 9001
  path: "/tv"
  secret_env: "TV_WEBHOOK_SECRET"
  mode: "tv_only"          # or: tv_and_bot
  match_window_ms: 120000   # wait time for the other side (two-man + parity alerts)
  alert_on_mismatch: true
  require_tf_match: false
```

### TradingView alert payload (JSON)

Your Pine script should `alert()` a JSON payload like:

```json
{
  "secret": "<your-secret>",
  "symbol": "{{ticker}}",
  "tickerid": "{{exchange}}:{{ticker}}",
  "tf": "{{interval}}",
  "side": "LONG",
  "confirm_time_ms": 1730000000000,
  "entry_price": 105432.10,
  "pivot_price": 105120.00,
  "pivot_osc": 12.3456,
  "slip_bps": 29.6,
  "loc_at_pivot": 0.072
}
```

Webhook URL: `http://<your-vm-ip>:9001/tv`

### Notes

- `confirm_time_ms` must be the **bar close timestamp (ms)** for the confirmation bar (what Pine calls `time_close`).
- In `tv_only`, the bot will still compute its own signals and log parity mismatches (optional), but **trades are driven by TV**.
- In `tv_and_bot`, mismatches are refused and (optionally) a Telegram “parity breach” alert is sent.

## Historical Replay
- Export TradingView Strategy Tester trades to CSV (List of Trades → Export).
- Convert to canonical `tv_signals.csv` (defaults to Binance close_time_ms = period_end-1):
  ```bash
  python -m div_donchian_bot.cli.tv_export_to_signals --in exported_trades.csv --out artifacts/tv_signals.csv --symbol BTCUSDT --side_map auto
  ```
- Offline parity check (same as before):
  ```bash
  python -m div_donchian_bot.cli.parity_check --bars_csv artifacts/bars_1m.csv --tv_signals_csv artifacts/tv_signals.csv --config configs/default.yaml
  ```
- Replay strategy-only across 15m/1h/4h (1m input auto-resampled, metrics + bot_signals per tf):
  ```bash
  python -m div_donchian_bot.cli.replay --config configs/default.yaml --bars_csv artifacts/bars_1m.csv --tv_signals_csv artifacts/tv_signals.csv --timeframes 15m,1h,4h --mode strategy_only --start 1700000000000 --end 1700003600000
  ```
- Replay the full engine in paper mode (structure/ATR + paper fills, requires aggTrades when CVD enabled):
  ```bash
  python -m div_donchian_bot.cli.replay --config configs/default.yaml --bars_csv artifacts/bars_1m.csv --agg_csv artifacts/agg_trades.csv --timeframes 15m,1h,4h --mode engine_paper --out_dir artifacts/replay
  ```
- Fire signals into a running `/tv` webhook endpoint:
  ```bash
  python -m div_donchian_bot.cli.replay_tv_webhooks --tv_signals_csv artifacts/tv_signals.csv --url http://127.0.0.1:9001/tv --secret_env TV_WEBHOOK_SECRET --sleep 0
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
