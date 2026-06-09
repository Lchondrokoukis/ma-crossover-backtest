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


def report(df):
    """Print strategy vs buy-and-hold side by side."""
    strat, bh = metrics(df["strat"]), metrics(df["ret"])
    print(f"\n{'Metric':<16}{'Strategy':>12}{'Buy & Hold':>12}")
    print("-" * 40)
    for k in strat:
        f = "{:.2f}".format if k == "Sharpe" else "{:.1%}".format
        print(f"{k:<16}{f(strat[k]):>12}{f(bh[k]):>12}")
    print()


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


def main():
    ticker, start, end, fast, slow = "SPY", "2015-01-01", "2024-12-31", 50, 200
    close = load_prices(ticker, start, end)

    gross = backtest(close, fast, slow, cost=0.0)
    net = backtest(close, fast, slow, cost=0.0005)  # 5 bps per trade

    print(f"\n{ticker}  MA({fast}/{slow})")
    report(net)

    n = int(net["trades"].sum())
    g, nr = gross["equity"].iloc[-1] - 1, net["equity"].iloc[-1] - 1
    print(f"Trades (entries + exits): {n}")
    print(f"Cost drag: {g:.1%} gross -> {nr:.1%} net  "
          f"({g - nr:.2%} lost to {0.0005:.2%}/trade)\n")

    plot(net, fast, slow, ticker)


if __name__ == "__main__":
    main()
