# Moving Average Crossover Backtest

A clean, minimal backtest of a long/flat trend-following strategy, benchmarked
against buy-and-hold.

## Strategy

Two moving averages, fast (e.g. 50-day) and slow (e.g. 200-day):

- **long** (position = 1) when the fast MA is above the slow MA
- **flat** (position = 0) when the fast MA is below the slow MA

A moving average is essentially a low-pass filter: it smooths out short-term
noise and leaves the slow trend. The crossover detects when the short-term
filtered signal turns relative to the long-term one.

## Run

```bash
pip install -r requirements.txt
python test_engine.py   # verify on synthetic data + lookahead-bias demo
python backtest.py      # run on real data (SPY, 2015-2024)
```

Change `ticker`, `start`, `end`, `fast`, `slow` in `main()`. The script prints
the metrics table and saves `backtest.png`.

## Transaction costs

`backtest(close, fast, slow, cost=...)` charges a cost on every entry and exit,
expressed as a fraction of traded value (e.g. `0.0005` = 5 basis points). The
position moves between 0 and 1, so `|position.diff()|` is the fraction of
capital traded on each day; the cost is subtracted from that day's return.

The lesson is in the turnover: a slow 50/200 crossover trades a handful of times
over a decade and barely feels costs, while a fast 5/20 crossover trades far
more often and can bleed double-digit percentage points to the same per-trade
cost. Note too that a cost paid early compounds away over the rest of the curve,
so the drag on *final* wealth is larger than the naive sum of per-trade costs.

## Trade-level statistics

`trade_returns()` extracts every round trip — entry at a 0→1 position change,
exit at the matching 1→0 — and compounds the daily strategy returns across it,
so both costs are included. `report()` prints the count, win rate and average
win/loss. A correctness invariant tested in `test_engine.py`: flat days
contribute nothing, so compounding the round-trip returns must rebuild the
final equity exactly.

Two lessons hide here. First, the day-level and trade-level win rates answer
different questions: trend following tends to take small whipsaw losses around
crossings and let a few large winners run, so average win is many times the
average loss — the shape of the distribution matters more than the win rate.
Second, a slow crossover produces only a handful of round trips even over
years of daily data, so any per-trade statistic carries wide uncertainty;
distrust a win rate estimated from a few trades.

## Reading the metrics

- **Total return / CAGR** — capital growth, total and annualized.
- **Sharpe ratio** — return per unit of risk (risk-free rate assumed 0).
- **Max drawdown** — worst peak-to-trough decline.

A trend-following strategy often *trails* buy-and-hold in a bull market (it
sits out part of the upside while the trend "confirms") but *cuts the drawdown*
in a bear market. That trade-off is the point — don't expect it to beat the
market, judge the risk taken for the return.

## The lookahead bias (the part that matters most)

In `backtest()` the critical line is:

```python
position = (ma_fast > ma_slow).astype(int).shift(1).fillna(0)
```

The signal on day `t` is built from `Close[t]`, which you only know once the
day has closed, so you can act on it no earlier than `t+1`. The `.shift(1)`
enforces that. Remove it and you trade on information you would not have had in
real time; results look impressive but are fake. `test_engine.py` demonstrates
this: the cheating version (no shift) consistently shows a higher return. This
is the single most common beginner mistake in backtesting.

## Limitations

Being explicit about these is what signals quant thinking:

- **A single asset.** Good performance on one ticker proves nothing — it may be
  luck or overfitting. Robustness needs many assets.
- **Fixed parameters.** Picking 50/200 by what works on the same dataset is
  overfitting. The fix is walk-forward / out-of-sample testing.
- **Costs are a flat per-trade rate.** Real slippage varies with liquidity and
  order size; this is a simplification.
- No short selling, leverage, position sizing, or survivorship-bias control.

## Extensions

In increasing difficulty — each makes a natural commit:

1. Parameter sweep over (fast, slow) with a Sharpe heatmap.
2. Walk-forward validation (choose params in-sample, test out-of-sample).
3. Run across a basket of assets and check the edge survives on average.
4. Replace the crossover with an ML-based signal.

Done: transaction-cost modeling (the `cost` parameter); trade-level statistics
(`trade_returns()`).

## Files

- `backtest.py` — the engine (load, backtest, metrics, report, plot)
- `test_engine.py` — synthetic-data check + lookahead-bias demo
- `requirements.txt` — dependencies
