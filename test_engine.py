"""Verify the engine on synthetic data (Yahoo is not reachable in CI):
the lookahead-bias demo and the transaction-cost model."""

import numpy as np
import pandas as pd
from backtest import backtest, metrics, report, plot

np.random.seed(42)
n = 1500
ret = np.random.normal(0.08 / 252, 0.20 / np.sqrt(252), n)
close = pd.Series(100 * np.exp(np.cumsum(ret)),
                  index=pd.bdate_range("2018-01-01", periods=n))

# --- basic sanity ---
df = backtest(close, 50, 200, cost=0.0)
report(df)
assert not df["equity"].isna().any()
assert df["position"].isin([0, 1]).all()

# --- lookahead demo: trading on the SAME day's signal (no shift) cheats ---
ma_fast, ma_slow = close.rolling(50).mean(), close.rolling(200).mean()
cheat_pos = (ma_fast > ma_slow).astype(int)          # missing the .shift(1)
cheat = (1 + cheat_pos * close.pct_change().fillna(0)).cumprod().iloc[-1] - 1
print(f"Correct (with shift):  {df['equity'].iloc[-1] - 1:>7.1%}")
print(f"Cheating (no shift):   {cheat:>7.1%}")

# --- transaction-cost checks ---
COST = 0.0005
net = backtest(close, 50, 200, cost=COST)
n_trades = int(net["trades"].sum())
gross_ret = df["equity"].iloc[-1] - 1
net_ret = net["equity"].iloc[-1] - 1

assert net_ret < gross_ret, "costs must reduce return"
drag = gross_ret - net_ret
print(f"\n50/200 (slow):  {n_trades} trades, drag {drag:.2%}")

# --- the lesson: a FAST crossover trades far more, so costs bite hard ---
fast_gross = backtest(close, 5, 20, cost=0.0)
fast_net = backtest(close, 5, 20, cost=COST)
fn = int(fast_net["trades"].sum())
fast_drag = (fast_gross["equity"].iloc[-1] - 1) - (fast_net["equity"].iloc[-1] - 1)
print(f"5/20  (fast):   {fn} trades, drag {fast_drag:.3%}")
print("=> low-turnover strategies barely feel costs; high-turnover ones suffer.")

plot(net, 50, 200, "SYNTHETIC", outfile="test_plot.png")
print("\nOK: engine + transaction costs verified.")
