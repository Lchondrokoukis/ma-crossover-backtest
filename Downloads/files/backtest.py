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


def metrics(ret):
    """Total return, CAGR, Sharpe and max drawdown for any return series.

    CAGR annualizes over the number of return periods (len - 1: the first
    point is the opening mark, not a return). A series with fewer than two
    points spans no return, so the growth metrics are NaN rather than an
    explosive ``x ** YEAR``.
    """
    if len(ret) < 2:
        return {"Total return": np.nan, "CAGR": np.nan,
                "Sharpe": 0.0, "Max drawdown": np.nan}
    equity = (1 + ret).cumprod()
    sd = ret.std()
    return {
        "Total return": equity.iloc[-1] - 1,
        "CAGR": equity.iloc[-1] ** (YEAR / (len(ret) - 1)) - 1,
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

    table = sweep(close, [5, 10, 20, 30, 50, 80], [20, 50, 100, 150, 200, 250],
                  cost=0.0005)
    print("Sharpe by (fast, slow):")
    print(table.round(2).to_string(na_rep="-"))
    bf, bs = table.stack().idxmax()
    print(f"Best in-sample: MA({bf}/{bs}), Sharpe {table.loc[bf, bs]:.2f} -- "
          f"partly luck until proven out-of-sample.\n")
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

    plot(net, fast, slow, ticker)


if __name__ == "__main__":
    main()
