"""Verify the engine on synthetic data (Yahoo is not reachable in CI):
the lookahead-bias demo and the transaction-cost model."""

import numpy as np
import pandas as pd
from backtest import (backtest, metrics, report, plot, trade_returns, sweep,
                      heatmap, walk_forward, basket, ml_backtest, vol_target)

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
# the buy-and-hold benchmark and the headline metrics must track the curve
assert np.isclose(df["equity_bh"].iloc[-1], (1 + df["ret"]).cumprod().iloc[-1])
assert np.isclose(metrics(df["strat"])["Total return"], df["equity"].iloc[-1] - 1)
assert metrics(df["strat"])["Max drawdown"] <= 0

# --- lookahead demo: trading on the SAME day's signal (no shift) cheats ---
ma_fast, ma_slow = close.rolling(50).mean(), close.rolling(200).mean()
cheat_pos = (ma_fast > ma_slow).astype(int)          # missing the .shift(1)
cheat = (1 + cheat_pos * close.pct_change().fillna(0)).cumprod().iloc[-1] - 1
print(f"Correct (with shift):  {df['equity'].iloc[-1] - 1:>7.1%}")
print(f"Cheating (no shift):   {cheat:>7.1%}")
# the cardinal rule, pinned: peeking at today's signal must beat the honest,
# shifted strategy -- this fires if .shift(1) is ever dropped from backtest()
assert cheat > df["equity"].iloc[-1] - 1, "no-shift lookahead must inflate returns"

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
assert fn > n_trades                                     # fast crossover trades far more
assert fast_drag > drag, "a higher-turnover strategy must feel costs more"
print("=> low-turnover strategies barely feel costs; high-turnover ones suffer.")

# --- trade-level stats ---
# flat days contribute nothing, so compounding round-trip returns rebuilds final equity
tr = trade_returns(net)
assert len(tr) == int((net["position"].diff() == 1).sum())   # one per entry
assert np.isclose((1 + tr).prod(), net["equity"].iloc[-1])   # trades rebuild equity

day_wr = (net.loc[net["position"] == 1, "strat"] > 0).mean()
wins = tr > 0
print(f"\nDay-level win rate:   {day_wr:.0%}")
print(f"Trade-level:          {wins.mean():.0%} of {len(tr)} round trips  "
      f"(avg win {tr[wins].mean():+.1%}, avg loss {tr[~wins].mean():+.1%})")
print("=> a few large winners carry the curve; and with this few round trips,")
print("   a win rate is far too noisy to be trusted on its own.")

# --- parameter sweep: grid cells must match direct runs ---
table = sweep(close, [10, 20, 50], [20, 100, 200], cost=COST)
assert np.isnan(table.loc[50, 20])                       # fast >= slow skipped
direct = metrics(backtest(close, 50, 200, cost=COST)["strat"])["Sharpe"]
assert np.isclose(table.loc[50, 200], direct)            # cell == direct run
print(f"\nSweep grid (Sharpe):\n{table.round(2).to_string(na_rep='-')}")

# --- heatmap renders on a denser grid ---
dense = sweep(close, [5, 10, 20, 30, 50, 80], [20, 50, 100, 150, 200, 250],
              cost=COST)
heatmap(dense, outfile="test_sweep.png")

# --- walk-forward on pure noise: the honest number is the OOS one ---
oos, folds = walk_forward(close, [10, 20, 50], [50, 100, 200],
                          train=504, test=252, cost=COST)
# the stitched OOS series tiles everything after the first train window
assert len(oos) == len(close) - 504
assert (oos.index == close.index[504:]).all()
# each fold's choice must be reproducible from its train window alone
t0 = sweep(close.iloc[:504], [10, 20, 50], [50, 100, 200], cost=COST)
assert (folds.iloc[0]["fast"], folds.iloc[0]["slow"]) == t0.stack().idxmax()
# and the stitched OOS slice must equal a direct run of that chosen pair --
# this catches a mis-slice or a cold-MA-warmup bug the index checks miss
recon = backtest(close.iloc[:504 + 252], int(folds.iloc[0]["fast"]),
                 int(folds.iloc[0]["slow"]), cost=COST)["strat"].iloc[504:504 + 252]
assert np.allclose(recon.values, oos.iloc[:252].values)

best_in_sample = sweep(close, [10, 20, 50], [50, 100, 200], cost=COST).stack().max()
oos_sharpe = metrics(oos)["Sharpe"]
print(f"\nWalk-forward on noise:  best in-sample Sharpe {best_in_sample:+.2f}  "
      f"vs out-of-sample {oos_sharpe:+.2f}")
print("=> on a random walk the in-sample best is always flattering;")
print("   the out-of-sample number is the one you can believe.")

# --- basket: rows must match direct runs, Average must be the mean ---
rng = np.random.default_rng(7)
noise = {f"A{i}": pd.Series(
    100 * np.exp(np.cumsum(rng.normal(0, 0.20 / np.sqrt(252), n))),
    index=close.index) for i in range(6)}
tab = basket(noise, 50, 200, cost=COST)
direct = metrics(backtest(noise["A0"], 50, 200, cost=COST)["strat"])["Sharpe"]
assert np.isclose(tab.loc["A0", "Sharpe"], direct)
# the B&H Sharpe column must equal a direct buy-and-hold run (was a tautology)
assert np.isclose(tab.loc["A0", "B&H Sharpe"],
                  metrics(noise["A0"].pct_change().fillna(0))["Sharpe"])

# the lesson: six assets from the SAME driftless process still spread widely,
# so the best single row always looks like an edge. The average is the test.
spread = tab["Sharpe"].iloc[:-1].max() - tab["Sharpe"].iloc[:-1].min()
assert spread > 0.5                                  # driftless rows disperse widely
assert abs(tab.loc["Average", "Sharpe"]) < spread    # dispersion dwarfs the average
print(f"\nBasket of driftless noise: per-asset Sharpe spread {spread:.2f}, "
      f"average {tab.loc['Average', 'Sharpe']:+.2f}")
print(tab.round(2).to_string())
print("=> the best row is luck, not edge; only the Average row means anything.")

# --- ML signal: structural no-lookahead invariants, then the humbling ---
ml = ml_backtest(close, cost=COST)
assert ml["position"].isin([0, 1]).all()
assert not ml["equity"].isna().any()
# the first fit happens at day 756 and its prediction is acted on a day
# later, so no position can exist on or before day 756
assert (ml["position"].iloc[:757] == 0).all()

held = ml["position"].iloc[757:]
hit = (held == (ml["ret"].iloc[757:] > 0)).mean()    # direction hit rate OOS
assert ml["trades"].sum() > 0                        # a degenerate model would never trade
print(f"\nML signal on the base series: hit rate {hit:.1%}, "
      f"long {held.mean():.0%} of days, Sharpe "
      f"{metrics(ml['strat'])['Sharpe']:.2f}")

# on driftless noise the model has nothing to learn: hit rate ~ coin flip
walk = pd.Series(100 * np.exp(np.cumsum(
    np.random.default_rng(1).normal(0, 0.20 / np.sqrt(252), n))),
    index=close.index)
mln = ml_backtest(walk, cost=COST)
hitn = (mln["position"].iloc[757:] == (mln["ret"].iloc[757:] > 0)).mean()
print(f"ML signal on driftless noise: hit rate {hitn:.1%}, Sharpe "
      f"{metrics(mln['strat'])['Sharpe']:.2f}")
assert 0.40 < hitn < 0.60, "on noise the hit rate must be near a coin flip"
print("=> the harness is the exercise, not the alpha: daily direction is")
print("   close to a coin flip, and the same accounting exposes that honestly.")

# --- volatility targeting: leverage moves risk, not edge ---
vt = vol_target(close, 50, 200, target=0.15, cost=0.0)
base = backtest(close, 50, 200, cost=0.0)
# leverage cannot create exposure where the crossover is flat
assert (vt.loc[base["position"] == 0, "position"] == 0).all()
# constant leverage scales returns but leaves Sharpe identical -- targeting
# reshapes risk, it cannot manufacture alpha
assert np.isclose(metrics(3 * base["strat"])["Sharpe"], metrics(base["strat"])["Sharpe"])
# on invested days the targeted vol sits closer to the 15% target than the raw
inv = base["position"] == 1
raw_vol = base.loc[inv, "ret"].std() * np.sqrt(252)
vt_vol = vt.loc[inv, "strat"].std() * np.sqrt(252)   # gross: strat == position*ret
assert abs(vt_vol - 0.15) < abs(raw_vol - 0.15)
print(f"\nVol targeting: invested vol {raw_vol:.0%} raw -> {vt_vol:.0%} targeted (target 15%)")

# --- robustness: degenerate inputs are handled cleanly, not cryptically ---
short = close.iloc[:100]
for bad in (lambda: walk_forward(short, [10, 20], [50, 100], train=504, test=252),
            lambda: ml_backtest(short, train=756)):
    try:
        bad()
        assert False, "expected ValueError on a series shorter than the train window"
    except ValueError:
        pass

# a monotonic series makes every next-day return positive (single-class
# labels), so ml_backtest must skip those folds rather than crash fit()
up = pd.Series(100 * np.exp(np.cumsum(np.full(n, 0.001))), index=close.index)
assert ml_backtest(up, train=756)["position"].isin([0, 1]).all()

# trade_returns must count a position already open on day 0 (boundary case)
lead = pd.DataFrame({"position": [1, 1, 1, 0, 1, 1],
                     "strat": [0.10, 0.05, 0.02, 0.0, 0.03, 0.04]})
lt = trade_returns(lead)
assert len(lt) == 2                                  # both round trips, incl. the day-0 entry
assert np.isclose(lt.iloc[0], 1.10 * 1.05 * 1.02 - 1)

# metrics on a flat series has no risk-adjusted return and does not explode
assert metrics(pd.Series([0.0] * 10))["Sharpe"] == 0.0
assert np.isfinite(metrics(pd.Series([0.01]))["CAGR"]) or np.isnan(metrics(pd.Series([0.01]))["CAGR"])
print("\nRobustness: short-series guards, single-class folds, day-0 trade, flat metrics OK.")

plot(net, 50, 200, "SYNTHETIC", outfile="test_plot.png")
print("\nOK: engine + costs + trade stats + sweep + heatmap + walk-forward "
      "+ basket + ML signal + vol targeting verified.")
