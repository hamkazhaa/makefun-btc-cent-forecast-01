# BTC cent forecast 01

Forecast Bitcoin's USD price 4 hours ahead from the last 72 hours of
1-minute closes. Entry for make.fun competition `btc-cent-forecast-01`
($250, deadline 2026-11-15T08:00:00Z).

## Program

`main.py` — Python 3.13, standard library only, no network, deterministic.

Reads every `<id>.in` in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes
`<id>.out` to `$MAKEFUN_OUTPUT_DIR` (else `/output`) as
`{"forecasts": [{"case": <id>, "price": <guess>}]}`.

## Method

Damped log-space ensemble around the checkpoint close:

```
log(guess) = log(last)
           + 0.10 * log(last / price 4h ago)
           + 0.15 * log(24h mean / last)
           + 0.15 * log(6h mean / last)
```

MAE is a proper scoring rule for the median, so the forecast is a shrunk,
honest middle estimate — never a wild directional bet. Weights were chosen
on walk-forward backtests inside the practice cases' own histories (66
checkpoints, every 4h): MAE 39,967 cents vs 40,950 for the persistence
baseline; on the five practice answers derivable from the kit's inputs it
scores 47,493 cents vs 47,055 for persistence (that window is one strong
downtrend, so the blend is deliberately kept close to persistence rather
than chasing either regime).

## Local run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Runs in ~0.06 s per case (cap is 8 s), byte-identical across runs.
