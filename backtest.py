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


def backtest(close, fast, slow, cost=0.0):
    """Run the strategy and return everything needed to analyse and plot it.

    cost is the per-trade transaction cost as a fraction of traded value
    (e.g. 0.0005 = 5 basis points), charged on every entry and every exit.

    The .shift(1) is the line that matters most: the signal on day t is built
    from Close[t], which you only know once the day has closed -- so you can
    act on it no earlier than day t+1. Without the shift you trade on
    information you would not have had in real time (lookahead bias) and the
    results look impressive but are fake.
    """
    ma_fast, ma_slow = close.rolling(fast).mean(), close.rolling(slow).mean()
    position = (ma_fast > ma_slow).astype(int).shift(1).fillna(0)
    ret = close.pct_change().fillna(0)

    # trades: 1 on each entry and each exit, 0 while holding. The position
    # moves between 0 and 1, so |change| is the fraction of capital traded.
    trades = position.diff().abs().fillna(0)
    strat = position * ret - cost * trades  # subtract cost on every trade
    return pd.DataFrame({
        "close": close, "ma_fast": ma_fast, "ma_slow": ma_slow,
        "position": position, "ret": ret, "trades": trades, "strat": strat,
        "equity": (1 + strat).cumprod(),
        "equity_bh": (1 + ret).cumprod(),
    })


def metrics(ret):
    """Total return, CAGR, Sharpe and max drawdown for any return series."""
    equity = (1 + ret).cumprod()
    return {
        "Total return": equity.iloc[-1] - 1,
        "CAGR": equity.iloc[-1] ** (YEAR / len(ret)) - 1,
        "Sharpe": np.sqrt(YEAR) * ret.mean() / ret.std() if ret.std() else 0.0,
        "Max drawdown": (equity / equity.cummax() - 1).min(),
    }


def sweep(close, fasts, slows, cost=0.0):
    """Sharpe ratio for every (fast, slow) pair with fast < slow.

    Returns a DataFrame (rows = fast, columns = slow), ready for a heatmap.
    Pairs with fast >= slow make no sense for a crossover and are left NaN.
    One warning matters more than the mechanics: the best cell of this table
    is in-sample optimisation -- some pair always looks great on the data it
    was tuned on, by luck alone. Whether it survives on unseen data is what
    walk-forward validation answers.
    """
    table = pd.DataFrame(index=fasts, columns=slows, dtype=float)
    table.index.name, table.columns.name = "fast", "slow"
    for f in fasts:
        for s in slows:
            if f < s:
                table.loc[f, s] = metrics(backtest(close, f, s, cost)["strat"])["Sharpe"]
    return table


def trade_returns(df):
    """Return of each round-trip trade (entry 0->1 through exit 1->0).

    Entries are numbered with a cumulative sum; each trade's daily strategy
    returns are then compounded from the entry day through the exit day, so
    entry and exit costs are both included. A trade still open at the end of
    the sample is marked to market.
    """
    change = df["position"].diff()
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
    """Plot the sweep grid: one cell per (fast, slow) pair, colour = Sharpe.

    What to look for is a broad plateau of similar colour. A strategy that
    only shines at one isolated cell is fitted to noise, not to structure;
    a wide region of decent Sharpes means the result is robust to the exact
    parameter choice.
    """
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

    # parameter sweep: Sharpe across the (fast, slow) grid, plus its heatmap
    table = sweep(close, [5, 10, 20, 30, 50, 80], [20, 50, 100, 150, 200, 250],
                  cost=0.0005)
    print("Sharpe by (fast, slow):")
    print(table.round(2).to_string(na_rep="-"))
    bf, bs = table.stack().idxmax()
    print(f"Best in-sample: MA({bf}/{bs}), Sharpe {table.loc[bf, bs]:.2f} -- "
          f"partly luck until proven out-of-sample.\n")
    heatmap(table)

    plot(net, fast, slow, ticker)


if __name__ == "__main__":
    main()
