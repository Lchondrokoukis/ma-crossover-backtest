"""
Moving Average Crossover Backtest
=================================
A long/flat trend-following strategy, benchmarked against buy-and-hold.

Go long when the fast moving average is above the slow one, stay flat otherwise.
The engine (backtest, metrics) is independent of the data source, so it can be
tested on synthetic data without a network connection.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

YEAR = 252  # trading days per year, for annualization


def load_prices(ticker, start, end):
    """Daily split/dividend-adjusted closes from Yahoo Finance, as a Series."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    close = df["Close"]
    return close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close


def _equity_frame(close, position, ret, cost, **extra):
    """Cost, P&L and equity accounting shared by every strategy frame.

    `position` is the (already lookahead-safe) exposure and `ret` the asset's
    daily return; `extra` columns are inserted right after `close` so each
    caller keeps its own column order. Keeping this in one place is the
    "same accounting everywhere" invariant the whole repo leans on.
    """
    # |diff| is the fraction of capital traded on each position change
    trades = position.diff().abs().fillna(0)
    strat = position * ret - cost * trades  # subtract cost on every trade
    return pd.DataFrame({
        "close": close, **extra,
        "position": position, "ret": ret, "trades": trades, "strat": strat,
        "equity": (1 + strat).cumprod(),
        "equity_bh": (1 + ret).cumprod(),
    })


def backtest(close, fast, slow, cost=0.0):
    """Return DataFrame of signals, returns, and equity curves.

    cost: per-trade fraction of traded value (e.g. 0.0005 = 5 bps).
    .shift(1) prevents lookahead: signal built on Close[t] is traded at t+1.
    """
    ma_fast, ma_slow = close.rolling(fast).mean(), close.rolling(slow).mean()
    position = (ma_fast > ma_slow).astype(int).shift(1).fillna(0)
    ret = close.pct_change().fillna(0)
    return _equity_frame(close, position, ret, cost, ma_fast=ma_fast, ma_slow=ma_slow)


def long_short(close, fast, slow, cost=0.0):
    """Crossover that goes SHORT instead of flat: +1 when the fast MA is above
    the slow one, -1 when below.

    Same lookahead guard (.shift(1)) and accounting as backtest(); the only
    change is the position takes -1 in a downtrend rather than 0, so the book
    is always fully invested, long or short.

    The lesson: being always-in is not strictly better. Shorting earns the
    down-moves the long/flat version sits out, but it also fights the equity
    risk premium -- on an asset that drifts up, the short legs carry negative
    expected return and the strategy usually trails buy-and-hold. It also
    doubles turnover: every crossover now closes one side and opens the other,
    so |position change| is 2, not 1, and costs bite twice as hard.
    """
    ma_fast, ma_slow = close.rolling(fast).mean(), close.rolling(slow).mean()
    position = np.sign(ma_fast - ma_slow).shift(1).fillna(0)   # +1 long / -1 short
    ret = close.pct_change().fillna(0)
    return _equity_frame(close, position, ret, cost, ma_fast=ma_fast, ma_slow=ma_slow)


def metrics(ret):
    """Total return, CAGR, Sharpe and max drawdown for any return series.

    CAGR annualizes over the number of return periods (len - 1: the first
    point is the opening mark, not a return). A series with fewer than two
    points spans no return, so the growth metrics are NaN rather than an
    explosive ``x ** YEAR``. An account wiped out along the way (equity at or
    below zero at the end) has a CAGR of -100%, not the NaN a fractional power
    of a negative number gives.
    """
    if len(ret) < 2:
        return {"Total return": np.nan, "CAGR": np.nan,
                "Sharpe": 0.0, "Max drawdown": np.nan}
    equity = (1 + ret).cumprod()
    sd = ret.std()
    return {
        "Total return": equity.iloc[-1] - 1,
        "CAGR": (equity.iloc[-1] ** (YEAR / (len(ret) - 1)) - 1
                 if equity.iloc[-1] > 0 else -1.0),
        "Sharpe": np.sqrt(YEAR) * ret.mean() / sd if sd and np.isfinite(sd) else 0.0,
        "Max drawdown": (equity / equity.cummax() - 1).min(),
    }


def sweep(close, fasts, slows, cost=0.0):
    """Sharpe ratio for every (fast, slow) pair with fast < slow.

    Returns a DataFrame (rows=fast, cols=slow); invalid pairs are NaN.
    """
    table = pd.DataFrame(index=fasts, columns=slows, dtype=float)
    table.index.name, table.columns.name = "fast", "slow"
    for f in fasts:
        for s in slows:
            if f < s:
                table.loc[f, s] = metrics(backtest(close, f, s, cost)["strat"])["Sharpe"]
    return table


def probabilistic_sharpe(ret, sr_star=0.0):
    """Probability that the TRUE (per-period) Sharpe exceeds `sr_star`.

    An observed Sharpe is a noisy estimate; with few observations, skew or fat
    tails it can look good by chance. The Probabilistic Sharpe Ratio (Bailey &
    Lopez de Prado) corrects for sample length, skewness and kurtosis and
    returns a probability. `sr_star` is a per-period benchmark Sharpe -- 0 asks
    simply "is the Sharpe positive at all?".
    """
    from scipy.stats import norm
    r = ret.dropna()
    sr = r.mean() / r.std()                       # per-period Sharpe (ddof=1)
    skew, kurt = r.skew(), r.kurt() + 3           # pandas kurt is excess -> Pearson
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return float(norm.cdf((sr - sr_star) * np.sqrt(len(r) - 1) / denom))


def _expected_max_sharpe(sharpes):
    """Per-period Sharpe one would EXPECT as the best of N independent trials
    whose Sharpes have this cross-sectional spread -- the multiple-testing
    benchmark (more trials -> a higher luckiest Sharpe even with no real edge)."""
    from scipy.stats import norm
    s = pd.Series(sharpes).dropna()
    N, g = len(s), 0.5772156649                   # Euler-Mascheroni constant
    return s.std() * ((1 - g) * norm.ppf(1 - 1.0 / N)
                      + g * norm.ppf(1 - 1.0 / (N * np.e)))


def deflated_sharpe(close, table, cost=0.0):
    """Deflate the sweep's best cell for the number of (fast, slow) trials.

    The single best cell of a sweep looks impressive partly because we tried
    many pairs -- the more configurations searched, the higher the luckiest
    Sharpe climbs even with no real edge. The Deflated Sharpe Ratio is the PSR
    of the best strategy measured not against zero but against the Sharpe you
    would EXPECT as the maximum of that many trials. Returns
    (best_pair, psr_vs_zero, dsr): the gap between the naive significance and
    the deflated one is the price of searching the grid.
    """
    per_period = table.stack().dropna() / np.sqrt(YEAR)   # annualized -> per-period
    f, s = table.stack().idxmax()
    best = backtest(close, int(f), int(s), cost)["strat"]
    return (f, s), probabilistic_sharpe(best, 0.0), \
        probabilistic_sharpe(best, _expected_max_sharpe(per_period))


def alpha_beta(strat, bench):
    """Split a strategy's daily returns into market exposure and skill (CAPM).

    Regresses the strategy's returns on the benchmark's (plain OLS, risk-free
    rate 0): strat_t = alpha + beta * bench_t + e_t. Returns beta, alpha
    (annualized), alpha's t-stat, R^2 and the information ratio (annualized
    alpha over residual volatility, as in Grinold & Kahn).

    The lesson: a return is not an edge until the market is taken out of it.
    Beta is how much benchmark the strategy carried -- for a long/flat rule on
    its own asset, roughly the share of days invested, weighted toward the
    volatile ones -- and alpha is what the timing added on top. A crossover
    that matches buy-and-hold's Sharpe while sitting out a quarter of the days
    can have no alpha at all: the same return per unit of risk, just less risk.
    """
    df = pd.concat({"s": strat, "b": bench}, axis=1).dropna()
    s, b, n = df["s"], df["b"], len(df)
    if n < 3:
        return {k: np.nan for k in ("Beta", "Alpha", "Alpha t-stat", "R2",
                                    "Information ratio")}
    sxx = ((b - b.mean()) ** 2).sum()
    beta = ((s - s.mean()) * (b - b.mean())).sum() / sxx if sxx else 0.0
    a = (s - beta * b).mean()                     # daily alpha, the OLS intercept
    e = s - beta * b - a                          # what the market doesn't explain
    sig = np.sqrt((e ** 2).sum() / (n - 2))       # residual vol, OLS dof
    se = sig * np.sqrt(1 / n + (b.mean() ** 2 / sxx if sxx else 0.0))
    if sig <= 1e-12 * s.std():                    # strat is exactly a + k * bench:
        sure = 0.0 if abs(a) <= 1e-12 * s.std() else np.copysign(np.inf, a)
        t = ir = sure                             # no residual risk, so no doubt
    else:
        t, ir = a / se, a * np.sqrt(YEAR) / sig
    sst = ((s - s.mean()) ** 2).sum()
    return {"Beta": beta, "Alpha": a * YEAR, "Alpha t-stat": t,
            "R2": 1 - (e ** 2).sum() / sst if sst else 0.0, "Information ratio": ir}


def walk_forward(close, fasts, slows, train=4 * YEAR, test=YEAR, cost=0.0):
    """Rolling train/test: pick the best-Sharpe pair in-sample, trade it out-of-sample.

    Returns (oos, folds): stitched out-of-sample returns and a per-fold summary.
    MAs are warmed on all prior prices so the first test bar is never NaN
    (this assumes max(slows) <= train, which holds for the grids used here).
    """
    if len(close) < train + 2:
        raise ValueError(f"need at least {train + 2} rows for train={train}, "
                         f"got {len(close)}")
    oos, rows = [], []
    for start in range(train, len(close), test):
        if len(close) - start < 2:        # skip a degenerate trailing 1-day fold
            break
        table = sweep(close.iloc[start - train:start], fasts, slows, cost)
        f, s = table.stack().idxmax()
        # run on all history up to the test end so the MAs are warm on day one
        seg = backtest(close.iloc[:start + test], f, s, cost)["strat"].iloc[start:]
        oos.append(seg)
        rows.append({"test_start": close.index[start].date(), "fast": f,
                     "slow": s, "train_sharpe": table.loc[f, s],
                     "test_sharpe": metrics(seg)["Sharpe"]})
    return pd.concat(oos), pd.DataFrame(rows)


def ml_features(close):
    """Features known at the close of day t: momentum at three horizons,
    the MA gap the crossover trades on, and recent volatility."""
    ret = close.pct_change()
    return pd.DataFrame({
        "mom_5": close.pct_change(5),
        "mom_20": close.pct_change(20),
        "mom_60": close.pct_change(60),
        "ma_gap": close.rolling(50).mean() / close.rolling(200).mean() - 1,
        "vol_20": ret.rolling(20).std(),
    })


def ml_backtest(close, cost=0.0, train=3 * YEAR, test=YEAR):
    """Replace the crossover with a learned signal, same discipline throughout.

    A logistic regression is refit every `test` days on the prior `train`
    days, predicting whether the NEXT day's return is positive; the strategy
    is long on days the model says up. Lookahead is closed at three places:
    features on day t use only closes up to t; the last training row is
    dropped (its label is the first test day's return); the prediction made
    at the close of day t is acted on at t+1 -- the same .shift(1) as the
    crossover. Returns the same frame as backtest(), minus the MA columns.
    """
    from sklearn.linear_model import LogisticRegression

    if len(close) <= train:
        raise ValueError(f"need more than train={train} rows, got {len(close)}")

    X = ml_features(close)
    ret = close.pct_change().fillna(0)
    y = (ret.shift(-1) > 0).astype(int)             # label: is tomorrow up?

    signal = pd.Series(0.0, index=close.index)
    for start in range(train, len(close), test):
        # train on the prior `train` rows minus the last one: that row's label
        # is the first test day's return, so dropping it closes the leak
        fit = X.iloc[start - train:start].dropna().index[:-1]
        if len(fit) == 0 or y.loc[fit].nunique() < 2:
            continue                                # degenerate window -> stay flat
        model = LogisticRegression(max_iter=1000).fit(X.loc[fit], y.loc[fit])
        pred = X.iloc[start:start + test].dropna().index
        signal.loc[pred] = model.predict(X.loc[pred])

    position = signal.shift(1).fillna(0)            # act the day after the signal
    return _equity_frame(close, position, ret, cost)


def basket(closes, fast, slow, cost=0.0):
    """Run one (fast, slow) pair across a mapping of name -> close Series.

    Returns one row of strategy metrics per asset (buy-and-hold Sharpe
    alongside) plus an "Average" row. An edge worth believing survives on
    average across assets it was never tuned on; any single row may be luck.
    """
    rows = {}
    for name, close in closes.items():
        df = backtest(close.dropna(), fast, slow, cost)
        m = metrics(df["strat"])
        m["B&H Sharpe"] = metrics(df["ret"])["Sharpe"]   # same ret the frame already holds
        rows[name] = m
    table = pd.DataFrame(rows).T
    table.loc["Average"] = table.mean()
    return table


def _listed(frame):
    """True from each column's first to its last valid value -- listed, halts
    included -- and False before a listing or after a delisting."""
    seen = frame.notna()
    return seen.cummax() & seen[::-1].cummax()[::-1]


def portfolio(closes, fast, slow, cost=0.0, scheme="inverse_vol", vol_window=60):
    """Combine per-asset crossover strategies into one portfolio equity curve.

    Each asset is run through backtest() for its daily strategy return; the
    returns are then blended with daily weights:

    - "equal":       1/N in every asset.
    - "inverse_vol": weight proportional to 1 / the asset's recent volatility,
                     so a calmer asset gets more capital (risk-weighting, the
                     seed of risk parity). The vol is a trailing estimate on
                     the asset's own trading days, lagged one print, so the
                     weights use only past data -- no lookahead. It is the
                     asset's vol, not the strategy's: a sleeve that sat flat
                     for `vol_window` days has zero realized vol, and 1/0 would
                     send the whole book to cash. A stale price can still push
                     an asset's vol toward 0, so the estimate is floored at
                     half the day's median across names -- no single name can
                     take over the book.

    Returns a DataFrame with the blended `strat`, its `equity`, and one
    `w_<name>` column per asset showing the daily weights.

    The universe is point-in-time: assets need not share a calendar. A name
    joins the book on its first print and leaves after its last one (a
    delisting, final loss included). In between, a day without a print is a
    halt: the name keeps its capital and earns nothing until it trades again,
    when its return covers the whole gap -- its capital is never lent to the
    other names meanwhile, so nothing is counted twice. Telling a halt from a
    delisting needs to know whether trading resumes: exchanges announce both,
    but a price series only shows the gap, so this is read from the sample
    (at its very edge a halted name looks delisted). It uses listing status,
    never a future return, and parking a halted name's capital at zero is the
    conservative choice.

    The lesson: diversification is the one free lunch in investing. Blending
    imperfectly correlated strategies keeps the average return but cancels part
    of the idiosyncratic risk, so the portfolio Sharpe is *higher* than the
    average of its components. Weighting by inverse volatility (equalizing risk
    contribution rather than capital) is the standard refinement when the
    assets' volatilities differ.
    """
    strat = pd.DataFrame({name: backtest(c.dropna(), fast, slow, cost)["strat"]
                          for name, c in closes.items()})       # union of dates
    listed = _listed(strat)
    strat = strat.fillna(0.0).where(listed)         # a halt earns 0, keeps its weight
    if scheme == "equal":
        w = pd.DataFrame(1.0, index=strat.index, columns=strat.columns)
    elif scheme == "inverse_vol":
        sd = pd.DataFrame({name: c.dropna().pct_change().rolling(vol_window).std().shift(1)
                           for name, c in closes.items()}).reindex(strat.index).ffill()
        sd = sd.clip(lower=sd.median(axis=1) / 2, axis=0)   # a stale price can't take the book
        w = 1.0 / sd.where(sd > 0)                  # asset risk, past prints only
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    w = w.where(listed)                             # point-in-time: listed names only
    w = w.div(w.sum(axis=1), axis=0).fillna(0.0)    # normalize each day to sum 1
    port = (w * strat).sum(axis=1)
    return pd.DataFrame({"strat": port, "equity": (1 + port).cumprod(),
                         **{f"w_{name}": w[name] for name in strat.columns}})


def survivors(closes, grace=0):
    """Keep only the names still trading at the end of the sample.

    This is the universe you get by downloading today's tickers -- and it is
    biased, kept here so the bias can be measured. Survival is an outcome
    nobody knew at the start (the names that collapsed got delisted), so
    filtering on it is lookahead in disguise: the losers vanish, their final
    losses with them, and every average looks better than anything an investor
    could actually have held. The honest test passes every name that existed
    at the time, dead ones included, to basket() / portfolio().

    A name survives if it printed on one of the last `grace` + 1 dates of the
    universe; the default 0 means the final date itself. Raise `grace` when
    names trade on different calendars, so one that merely skipped the last
    day (a local holiday) is not mistaken for a delisting -- at the price of
    counting a death inside that window as survival.
    """
    last = {name: c.last_valid_index() for name, c in closes.items()}
    dates = sorted({d for c in closes.values() for d in c.dropna().index})
    if not dates:
        raise ValueError("survivors() needs at least one name with a price")
    cutoff = dates[max(len(dates) - 1 - grace, 0)]
    return {name: c for name, c in closes.items()
            if last[name] is not None and last[name] >= cutoff}


def survivorship_bias(closes, fast, slow, cost=0.0, scheme="equal", grace=0):
    """The same portfolio on the point-in-time universe and on its survivors.

    Returns rows "Point-in-time" (every name, dead ones included), "Survivors
    only" and "Bias" (survivors minus point-in-time), with the total return
    and Sharpe of the crossover portfolio and of "EW", an always-long
    equal-weight book (1/N of the listed names, rebalanced daily -- the
    strategy portfolio's own construction without the signal).

    The lesson: survivorship bias is lookahead by another name -- choosing the
    sample by who exists today lets the outcome pick the test. The always-long
    book eats the whole bias; the crossover is usually already flat when a name
    dies, so it dodges part of it, but not the selection itself.
    """
    rows = {}
    for label, u in (("Point-in-time", closes), ("Survivors only", survivors(closes, grace))):
        s = metrics(portfolio(u, fast, slow, cost, scheme)["strat"])
        rets = pd.DataFrame({k: c.dropna().pct_change() for k, c in u.items()})
        rets = rets.fillna(0.0).where(_listed(pd.DataFrame(u)))   # halts earn 0
        e = metrics(rets.mean(axis=1).fillna(0.0))
        rows[label] = {"Strategy return": s["Total return"], "Strategy Sharpe": s["Sharpe"],
                       "EW return": e["Total return"], "EW Sharpe": e["Sharpe"]}
    table = pd.DataFrame(rows).T
    table.loc["Bias"] = table.loc["Survivors only"] - table.loc["Point-in-time"]
    return table


def vol_target(close, fast, slow, target=0.15, cost=0.0, vol_window=20, max_lev=3.0):
    """Size the long/flat crossover so realized volatility tracks `target`.

    The crossover gives a 0/1 signal; here we lever it. Each day we estimate
    annualized volatility from the trailing `vol_window` daily returns and set
    leverage = target / realized_vol (capped at `max_lev`), so a calm market is
    geared up toward the target and a turbulent one is scaled down. Both the
    signal and the vol estimate use only past data (.shift(1)), so there is no
    lookahead. The returned `position` is continuous (>= 0), not 0/1.

    The lesson: vol targeting adds no alpha. Leverage scales return and risk
    together, so a *constant* leverage leaves Sharpe unchanged; what targeting
    buys is risk *stability* -- realized vol stays near the target instead of
    drifting with the market, which steadies drawdowns. It is not free: the
    position now nudges every day, so turnover (and cost) rises above the
    on/off crossover.
    """
    base = backtest(close, fast, slow, cost=0.0)        # reuse signal & ret
    realized = base["ret"].rolling(vol_window).std() * np.sqrt(YEAR)
    lev = (target / realized).shift(1).clip(upper=max_lev).fillna(0.0)
    position = base["position"] * lev                   # both already lookahead-safe
    return _equity_frame(close, position, base["ret"], cost)


def trade_returns(df):
    """Compounded return of each round-trip trade, including entry/exit costs.

    An open position at the end of the sample is marked to market.
    """
    prev = df["position"].shift(fill_value=0)           # implicit flat before day 0
    change = df["position"] - prev                      # so a leading 1 counts as an entry
    tid = (change == 1).cumsum()                        # trade number, set at entry
    mask = ((df["position"] == 1) | (change == -1)) & (tid > 0)
    return (1 + df.loc[mask, "strat"]).groupby(tid[mask]).prod() - 1


def report(df):
    """Print strategy vs buy-and-hold metrics, then per-trade statistics."""
    strat, bh = metrics(df["strat"]), metrics(df["ret"])
    print(f"\n{'Metric':<16}{'Strategy':>12}{'Buy & Hold':>12}")
    print("-" * 40)
    for k in strat:
        f = "{:.2f}".format if k == "Sharpe" else "{:.1%}".format
        print(f"{k:<16}{f(strat[k]):>12}{f(bh[k]):>12}")
    ab = alpha_beta(df["strat"], df["ret"])
    print(f"Beta {ab['Beta']:.2f}   Alpha {ab['Alpha']:+.1%}/yr (t = {ab['Alpha t-stat']:.2f})"
          f"   R2 {ab['R2']:.2f}")

    tr = trade_returns(df)
    if tr.empty:
        print("No trades.\n")
        return
    wins = tr > 0
    avg_win = tr[wins].mean() if wins.any() else 0.0
    avg_loss = tr[~wins].mean() if (~wins).any() else 0.0
    print(f"Trades: {len(tr)}   Win rate: {wins.mean():.0%}   "
          f"Avg win: {avg_win:+.1%}   Avg loss: {avg_loss:+.1%}\n")


def plot(df, fast, slow, ticker, outfile="backtest.png"):
    """Price with both MAs and entry/exit markers, plus the equity curves."""
    fig, (top, bot) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    top.plot(df.index, df["close"], color="#888780", lw=1, label="Close")
    top.plot(df.index, df["ma_fast"], color="#378ADD", lw=1.2, label=f"MA {fast}")
    top.plot(df.index, df["ma_slow"], color="#D85A30", lw=1.2, label=f"MA {slow}")
    trade = df["position"].diff()  # +1 marks an entry, -1 marks an exit
    top.scatter(df.index[trade == 1], df["close"][trade == 1],
                marker="^", color="#1D9E75", s=70, zorder=5, label="Entry")
    top.scatter(df.index[trade == -1], df["close"][trade == -1],
                marker="v", color="#E24B4A", s=70, zorder=5, label="Exit")
    top.set(title=f"{ticker} - MA({fast}/{slow}) crossover", ylabel="Price")
    top.legend(fontsize=9)

    bot.plot(df.index, df["equity"], color="#534AB7", lw=1.5, label="Strategy")
    bot.plot(df.index, df["equity_bh"], color="#888780", lw=1.2, ls="--", label="Buy & Hold")
    bot.set(ylabel="Growth of 1", xlabel="Date")
    bot.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(outfile, dpi=130)
    print(f"Saved plot -> {outfile}")


def heatmap(table, outfile="sweep.png"):
    """Save a colour grid of Sharpe ratios across the (fast, slow) parameter space."""
    data = np.ma.masked_invalid(table.values.astype(float))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#e8e6dc")  # invalid pairs (fast >= slow) in neutral grey

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set(xlabel="slow window", ylabel="fast window",
           title="Sharpe ratio by (fast, slow)")
    for i in range(len(table.index)):          # annotate each valid cell
        for j in range(len(table.columns)):
            if not np.isnan(table.iat[i, j]):
                ax.text(j, i, f"{table.iat[i, j]:.2f}",
                        ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Sharpe")
    fig.tight_layout()
    fig.savefig(outfile, dpi=130)
    print(f"Saved heatmap -> {outfile}")


def main():
    ticker, start, end, fast, slow = "SPY", "2015-01-01", "2024-12-31", 50, 200
    close = load_prices(ticker, start, end)

    gross = backtest(close, fast, slow, cost=0.0)
    net = backtest(close, fast, slow, cost=0.0005)  # 5 bps per trade

    print(f"\n{ticker}  MA({fast}/{slow})")
    report(net)

    n = int(net["trades"].sum())
    g, nr = gross["equity"].iloc[-1] - 1, net["equity"].iloc[-1] - 1
    print(f"Cost drag: {g:.1%} gross -> {nr:.1%} net "
          f"over {n} position changes at {0.0005:.2%} each\n")

    # alpha vs beta: how much of that return is just being in the market?
    ab = alpha_beta(net["strat"], net["ret"])
    print(f"Alpha vs beta: beta {ab['Beta']:.2f} with {net['position'].mean():.0%} of days "
          f"invested; alpha {ab['Alpha']:+.1%}/yr at t = {ab['Alpha t-stat']:.2f} -- judge "
          f"the timing by alpha's t-stat (|t| > 2), not by the Sharpe.\n")

    table = sweep(close, [5, 10, 20, 30, 50, 80], [20, 50, 100, 150, 200, 250],
                  cost=0.0005)
    print("Sharpe by (fast, slow):")
    print(table.round(2).to_string(na_rep="-"))
    bf, bs = table.stack().idxmax()
    print(f"Best in-sample: MA({bf}/{bs}), Sharpe {table.loc[bf, bs]:.2f} -- "
          f"partly luck until proven out-of-sample.")
    _, psr0, dsr = deflated_sharpe(close, table, cost=0.0005)
    print(f"Deflated for {table.stack().size} trials: naive P(Sharpe>0) = {psr0:.0%}, "
          f"deflated = {dsr:.0%} -- the gap is the cost of picking the best cell.\n")
    heatmap(table)

    oos, folds = walk_forward(close, [5, 10, 20, 30, 50, 80],
                              [20, 50, 100, 150, 200, 250], cost=0.0005)
    print("Walk-forward folds:")
    print(folds.to_string(index=False))
    m = metrics(oos)
    print(f"Out-of-sample: Sharpe {m['Sharpe']:.2f}, total {m['Total return']:.1%} "
          f"over {len(oos)} days -- judge the strategy on this, not on the "
          f"best in-sample cell above.\n")

    # basket: same fixed pair across assets it was never tuned on
    tickers = ["SPY", "QQQ", "IWM", "EFA", "EEM", "GLD"]
    closes = {t: load_prices(t, start, end) for t in tickers}
    tab = basket(closes, fast, slow, cost=0.0005)
    print(f"MA({fast}/{slow}) across a basket:")
    print(tab.to_string(formatters={
        c: ("{:.2f}".format if "Sharpe" in c else "{:.1%}".format)
        for c in tab.columns}))
    print("Judge the Average row, not the best one -- a single good ticker "
          "proves nothing.\n")

    # ML signal: same walk-forward discipline, learned instead of hand-coded
    ml = ml_backtest(close, cost=0.0005)
    print("ML signal (logistic regression, refit each year on the prior 3):")
    report(ml)

    # volatility targeting: size the same crossover to a 15% annual vol target
    vt = vol_target(close, fast, slow, target=0.15, cost=0.0005)
    vtm, rawm = metrics(vt["strat"]), metrics(net["strat"])
    print("Volatility-targeted 50/200 (15% target) vs the raw crossover:")
    print(f"  Sharpe {vtm['Sharpe']:.2f} vs {rawm['Sharpe']:.2f}   "
          f"max drawdown {vtm['Max drawdown']:.1%} vs {rawm['Max drawdown']:.1%}")
    print("  leverage moves risk, not edge -- Sharpe is roughly unchanged while "
          "the risk profile steadies.\n")

    # long-short: go short in downtrends instead of sitting flat
    ls = long_short(close, fast, slow, cost=0.0005)
    lsm = metrics(ls["strat"])
    print(f"Long-short {fast}/{slow}: total {lsm['Total return']:.1%}, "
          f"Sharpe {lsm['Sharpe']:.2f} vs long/flat {rawm['Sharpe']:.2f} "
          f"(B&H {metrics(net['ret'])['Total return']:.1%}) -- shorting an "
          f"up-drifting market usually trails buy-and-hold and doubles turnover.\n")

    # portfolio: blend the same crossover across the basket into one curve
    port = portfolio(closes, fast, slow, cost=0.0005, scheme="inverse_vol")
    comp = [metrics(backtest(c.dropna(), fast, slow, 0.0005)["strat"])["Sharpe"]
            for c in closes.values()]
    pm = metrics(port["strat"])
    print(f"Inverse-vol portfolio of {len(closes)} assets: Sharpe {pm['Sharpe']:.2f} "
          f"vs avg component {np.mean(comp):.2f} -- diversification lifts the "
          f"risk-adjusted return above any single name on average.\n")

    # survivorship: Yahoo serves only tickers alive today, so this basket is
    # survivors-only by construction -- the bias is invisible here, not absent
    sb = survivorship_bias(closes, fast, slow, cost=0.0005)
    print(f"Survivorship: {len(survivors(closes))} of {len(closes)} tickers trade to "
          f"the end, so the measured always-long bias reads {sb.loc['Bias', 'EW return']:+.1%} "
          f"-- invisible, not absent: free data has no delisted names, so every "
          f"real-data number above is survivors-only. test_engine.py sizes the bias.\n")

    plot(net, fast, slow, ticker)


if __name__ == "__main__":
    main()
