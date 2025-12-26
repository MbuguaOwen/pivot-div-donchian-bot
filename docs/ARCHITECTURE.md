# Architecture

## Data sources
- **kline_15m** stream for OHLCV and bar-close events (signals only on `x=true` close).
- Optional **aggTrade** stream to build CVD per bar:
  - delta += qty when buyer-aggressor (isBuyerMaker=false)
  - delta -= qty when seller-aggressor (isBuyerMaker=true)
  - cvd[t] = cvd[t-1] + delta_bar[t]

## Strategy semantics
Matches the Pine behavior:
- A pivot is only *known* `pivotLen` bars later (confirmation).
- Divergence is evaluated at the pivot bar against the last confirmed pivot of the same type.
- Entry fires on the confirmation bar close.

## Divergence modes
- `pine`: EMA((close-open)*volume)
- `cvd`: bar-close CVD at the pivot
- `both`: require both to agree
- `either`: accept either

## Risk
Optional ATR SL/TP:
- LONG: SL = entry - sl_mult*ATR, TP = entry + tp_mult*ATR
- SHORT: SL = entry + sl_mult*ATR, TP = entry - tp_mult*ATR
