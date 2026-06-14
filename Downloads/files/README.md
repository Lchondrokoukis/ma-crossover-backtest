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

## Parameter sweep

`sweep(close, fasts, slows, cost)` computes the Sharpe ratio for every valid
(fast, slow) pair and returns the grid as a DataFrame; `heatmap(table)` plots
it (invalid pairs, fast ≥ slow, are greyed out and each valid cell is
annotated). `main()` prints the grid, flags the best cell and saves
`sweep.png`.

The warning matters more than the mechanics: picking the best cell of this
table is **in-sample optimisation**. Some pair will always look great on the
data it was tuned on, by luck alone — `test_engine.py` makes this concrete by
sweeping *pure noise* (a synthetic random walk), where the Sharpe still spreads
widely across cells despite there being no real structure to find. What you
want to see in a heatmap is a *broad plateau* of similar colour: a strategy
that only shines at one isolated spike is fitted to noise. Whether the chosen
pair survives on unseen data is a separate question, answered by walk-forward
validation.

## Walk-forward validation

`walk_forward(close, fasts, slows, train=..., test=..., cost=...)` splits the
sample into rolling folds: on each train window (default 4 years) the sweep
picks the best (fast, slow) pair, and that pair alone is traded over the
following test window (default 1 year). The stitched test returns are
genuinely out-of-sample — every parameter choice is made using only data
available before the period it is traded in. The MAs for a test window are
warmed up on earlier prices: price history is known in real time, it is the
*choice* of parameters that must not peek.

`main()` prints one row per fold (the chosen pair, its train Sharpe and its
test Sharpe) and the out-of-sample metrics for the stitched series. Two things
to notice. First, the chosen pair usually *changes* from fold to fold — that
instability is itself evidence that the in-sample optimum is partly noise.
Second, the out-of-sample Sharpe is the honest number to quote;
`test_engine.py` runs the same procedure on a synthetic random walk and shows
the in-sample best flattering the out-of-sample result even where there is no
structure at all.

## Multi-asset basket

`basket(closes, fast, slow, cost)` runs one fixed pair across a mapping of
name → close series and returns a table of per-asset strategy metrics (with
buy-and-hold Sharpe alongside) plus an Average row. `main()` runs the 50/200
pair on a small ETF basket spanning US large/small caps, developed and
emerging markets, and gold — assets the parameters were never tuned on.

The lesson: good performance on a single ticker proves nothing. In
`test_engine.py` six assets drawn from the *same driftless noise process*
spread by more than a full Sharpe point — the best row always looks like an
edge, and it is pure luck. An edge worth believing survives **on average**
across assets; judge the Average row, never the best one.

## ML-based signal

`ml_backtest(close, cost, train, test)` replaces the hand-coded crossover
with a learned one: a logistic regression over simple features
(`ml_features()` — momentum at three horizons, the 50/200 MA gap, recent
volatility) is refit every `test` days on the prior `train` days and
predicts whether the *next* day's return is positive; the strategy is long
on predicted-up days. The accounting (costs, equity, trade stats) is
identical to `backtest()`.

Lookahead can sneak into an ML signal in three places, and all three are
closed: features on day t use only closes up to t; the label for day t is
the day t+1 return, so the last training row is dropped (its label is the
first test day's return); and the prediction made at the close of day t is
acted on at day t+1 — the same `.shift(1)` as the crossover.

Don't expect magic, and `test_engine.py` quantifies why: on driftless noise
the out-of-sample hit rate is ~49% — a coin flip, exactly as it should be.
Daily direction is barely predictable; the exercise is the harness, not the
alpha. Any signal, learned or hand-coded, plugs into the same honest
accounting.

## Volatility targeting

`vol_target(close, fast, slow, target, cost, vol_window, max_lev)` keeps the
same 0/1 crossover signal but *sizes* it. Each day it estimates annualized
volatility from the trailing `vol_window` returns and levers the position by
`target / realized_vol` (capped at `max_lev`): calm markets get geared up
toward the target, turbulent ones scaled down. Signal and vol estimate are
both `.shift(1)`-ed, so there is no lookahead; the returned `position` is now
continuous rather than 0/1.

The lesson is what targeting does *not* do: it adds no alpha. Leverage scales
return and risk together, so a *constant* leverage leaves Sharpe unchanged —
`test_engine.py` pins exactly that (`metrics(3 * strat)` Sharpe equals
`metrics(strat)`). What it buys is risk *stability*: on invested days the
synthetic asset's ~20% vol is pulled toward the 15% target, so realized risk
stops drifting with the market and drawdowns steady. It is not free — the
position nudges every day, so turnover and cost rise above the on/off signal.

## Long-short positions

`long_short(close, fast, slow, cost)` keeps the crossover but goes **short**
(position −1) in a downtrend instead of flat, so the book is always fully
invested. Same `.shift(1)` guard and the same `_equity_frame` accounting; the
position is `np.sign(ma_fast - ma_slow)` rather than a 0/1 indicator.

The lesson is that "always in the market" is not automatically better.
`test_engine.py` pins two facts on the synthetic series: once the MAs are warm
the short book is exactly the long/flat book mapped {0,1}→{−1,+1}, and its
turnover is higher because every crossover now closes one side and opens the
other (|Δposition| = 2, not 1). On an up-drifting asset the short legs fight
the equity risk premium, so long-short trails buy-and-hold (≈163% vs 304% on
the test series) at a lower Sharpe — capturing the down-moves rarely pays for
being short through the drift.

## Extensions

The three original extensions plus several follow-ups are done, each as its
own commit: transaction-cost modeling (the `cost` parameter); trade-level
statistics (`trade_returns()`); parameter sweep with Sharpe heatmap
(`sweep()`, `heatmap()`); walk-forward validation (`walk_forward()`);
multi-asset basket (`basket()`); ML-based signal (`ml_backtest()`);
volatility targeting (`vol_target()`); and long-short positions
(`long_short()`).

Natural next steps: a survivorship-bias-free universe, or a portfolio layer
that combines the basket into one equity curve with risk-based weights.

## Files

- `backtest.py` — the engine (load, backtest, metrics, report, plot)
- `test_engine.py` — synthetic-data check + lookahead-bias demo
- `requirements.txt` — dependencies
