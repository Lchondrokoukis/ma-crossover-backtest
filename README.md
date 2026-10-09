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

## Market impact and capacity

A flat cost per trade ignores *size*. `trading_costs(close, volume, trades,
aum, half_spread, impact)` adds the empirical **square-root law**: an order of
Q dollars moves the price against itself by about `impact × σ × √(Q / ADV)`,
σ the daily volatility and ADV the average daily dollar volume, both trailing
estimates known before the trade. With `impact=0` it is exactly the flat
`cost` above; the tests pin that 4× the dollars cost 2× as much per dollar and
that a quarter-size trade costs an eighth.

Because impact per dollar grows with the square root of fund size, every
strategy has a **capacity**. `capacity(close, volume, fast, slow)` finds the
size that maximizes dollar profit, in closed form: profit `A·(G − S − K√A)`
peaks at `√A* = (G − S) / 1.5K`, where impact eats exactly two-thirds of the
edge left after spreads — growing past it adds dollars of cost faster than
dollars of return, which is why good funds close to new money. Capacity is
edge² over trading cost², so it scales linearly with the market's liquidity
and as 1/impact², and at a given edge turnover is what kills it: on the test
series, with gross Sharpes of 1.29 and 1.24, the 50/200 crossover (7 trades)
can run $8.2B and the 5/20 (89 trades) only $29M — 286× less.

The model executes each trade within a day. `capacity()` reports the largest
day's trade as a multiple of ADV, and above ~0.1 a real desk would spread the
order over days. At capacity the fast crossover's biggest trade is 0.6× a day's
volume — a few days of careful execution — but the slow one's is 165×, months
of the market's entire volume: its capacity is an extrapolation of the model,
read it as "far more than you will ever run", not as a number.

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
- **Beta / alpha** — how much of the market the strategy carried, and what its
  timing added on top (see Alpha vs beta).

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
- **One-day execution.** `trading_costs()` sizes impact by liquidity and order
  size, but charges each trade within a day; orders far above ~10% of daily
  volume would really be spread over days, so their costs are extrapolated.
- **Survivors-only real data.** The engine handles point-in-time universes
  (see Survivorship bias), but Yahoo serves only tickers that still trade, so
  `main()`'s basket is survivors-only and its numbers are flattered.

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

## Deflated Sharpe ratio

`probabilistic_sharpe(ret, sr_star)` and `deflated_sharpe(close, table, cost)`
put a number on the sweep's central warning. The **Probabilistic Sharpe Ratio**
(Bailey & López de Prado) corrects an observed Sharpe for sample length, skew
and kurtosis and returns the probability the *true* Sharpe beats a benchmark.
The **Deflated Sharpe Ratio** sets that benchmark to the Sharpe you would
*expect* as the maximum of N independent trials — so it asks whether the best
cell of an N-pair sweep is still significant once you admit you went looking.

`test_engine.py` makes the point unmissable: on the drifting series the best
cell deflates only from ≈100% to ≈100% (a real edge survives), but on a
*driftless random walk* the best cell looks plausible naively (PSR ≈ 71%) and
collapses to DSR ≈ 31% once corrected for the ~20 pairs tried. That gap is the
quantified cost of in-sample optimisation — the same lesson the heatmap shows
visually, now as a probability.

## Alpha vs beta

`alpha_beta(strat, bench)` regresses the strategy's daily returns on the
benchmark's (CAPM, plain OLS): **beta** is how much market the strategy
carried, **alpha** the annualized return left once that is taken out, with its
t-stat, R² and information ratio (alpha over residual volatility). `report()`
prints the line under every strategy.

The lesson is that a good Sharpe is not an edge. On the test series the 50/200
crossover's Sharpe (1.29) matches buy-and-hold's (1.28), yet its beta is 0.74 —
close to the 72% of days it is invested, weighted toward the volatile ones —
and its alpha has t = 0.88, nowhere near significant: the same return per unit
of risk, just less risk. Leverage is beta too: a fixed fraction of the market
has that fraction as its beta and zero alpha, whatever its return.

Alpha only appears when there is structure to time. Across 40 random walks the
crossover's alpha t-stat averages −0.14; across 40 markets with persistent
bull/bear regimes it averages +1.15. And even there a single 6-year sample
clears t > 2 in only 20% of runs: a real edge is hard to *prove* from a few
years of daily data, which is the same humility the deflated Sharpe teaches.

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

## Portfolio construction

`portfolio(closes, fast, slow, cost, scheme, vol_window)` is the capstone: it
runs the crossover on every asset in a basket and blends their daily strategy
returns into **one** portfolio equity curve. Two weighting schemes: `"equal"`
(1/N) and `"inverse_vol"` (weight ∝ 1 / the asset's trailing volatility, the
seed of risk parity). The inverse-vol weights are a trailing estimate shifted
one day, so the allocation uses only past data — the same no-lookahead
discipline as everywhere else. The volatility is the *asset's*, not the
strategy's: a crossover sleeve that sits flat has zero realized volatility,
and an earlier version that weighted by 1/0 left the whole book in cash on
half the days of the test. The estimate is taken on each asset's own trading
days and floored at half the day's median across names, so a missed print or
a stale price can neither zero a weight nor hand the book to one name (a test
freezes one price for 100 days: its weight stays near 0.3 instead of 0.8).

The lesson is the one genuinely free lunch in investing: **diversification.**
`test_engine.py` builds six independent drifting assets and shows the
equal-weight blend's Sharpe (≈1.0) comfortably beating the average component
Sharpe (≈0.4) — blending imperfectly correlated strategies keeps the average
return while cancelling part of the idiosyncratic risk. The inverse-vol scheme
then equalizes *risk* rather than capital, handing more weight to the calmer
assets so that weight × volatility is the same for every name (a property the
tests pin directly).

## Survivorship bias

Backtest only the tickers that exist *today* and you have already peeked.
Whether a name survives is an outcome only the future knows, and the names
missing from today's list are mostly the ones that collapsed — drop them and
every average improves, though no investor at the start could have known
which to skip. Survivorship bias is lookahead in the *universe* rather than in
the signal.

`survivors(closes)` builds that biased universe on purpose (the names still
trading on the last date; `grace` widens that window for names on different
calendars), and `survivorship_bias(closes, fast, slow, cost)` runs the same
portfolio on it and on the **point-in-time** universe (every name that existed
at the time, dead ones included), with a Bias row between them. `portfolio()`
is point-in-time now: a name joins the book on its first print and leaves after
its last, so a delisting name is held through its last close, final loss
included, and its capital moves to the survivors the next day. A missing day in
between is a halt: the name keeps its capital at zero return until it trades
again, instead of lending it to the others and then also collecting the whole
gap move. Before, one delisting truncated the whole book — at day 228 of 1500
in the test. `basket()` needed no change, since it already runs each name on its
own dates, but its Average row is only as honest as its rows: on the survivors
it lifts the average buy-and-hold Sharpe from −0.23 to +0.27.

`test_engine.py` simulates 30 zero-edge stocks (expected return exactly 0)
that delist at a 30% haircut once they close below 30; 11 die. Point-in-time,
an always-long equal-weight book (1/N of the listed names, rebalanced daily —
the `EW` columns) returns −5%; the 19 survivors alone show +80% at a Sharpe of
1.24 — profit made purely by who was left out. About 14 points of
that gap are the delisting returns alone: a database that keeps the dead
tickers but drops their final loss still reports +8%. The crossover dodges
part of the bias (it was flat on every delisting day), yet its Sharpe still
rises from 0.46 to 0.92 on the survivors. The test also pins the lookahead
itself: delete the data after a date and the point-in-time book up to it is
unchanged, while the survivors-only book moves, because who "survives" was
decided later.

Only performance delistings are modelled; takeovers also remove names, at a
premium, so the real-market bias is smaller — these numbers size the
mechanism, not the market. Yahoo serves no delisted tickers, so `main()`'s
basket is survivors-only by construction and its measured bias reads zero:
invisible, not absent.

## Extensions

The three original extensions plus several follow-ups are done, each as its
own commit: transaction-cost modeling (the `cost` parameter); trade-level
statistics (`trade_returns()`); parameter sweep with Sharpe heatmap
(`sweep()`, `heatmap()`); walk-forward validation (`walk_forward()`);
multi-asset basket (`basket()`); ML-based signal (`ml_backtest()`);
volatility targeting (`vol_target()`); long-short positions (`long_short()`);
portfolio construction with risk-based weighting (`portfolio()`); the
deflated Sharpe ratio (`deflated_sharpe()`, `probabilistic_sharpe()`);
survivorship-bias control (point-in-time `portfolio()`, `survivors()`,
`survivorship_bias()`); alpha/beta attribution (`alpha_beta()`); and market
impact with capacity (`trading_costs()`, `capacity()`).

What remains is data, not code: measuring the bias on real markets needs a
delisting-aware source (e.g. CRSP) that keeps the dead tickers and their
delisting returns.

## Files

- `backtest.py` — the engine (load, backtest, metrics, report, plot)
- `test_engine.py` — synthetic-data check + lookahead-bias demo
- `requirements.txt` — dependencies
