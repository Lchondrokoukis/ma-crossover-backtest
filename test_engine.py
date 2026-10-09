"""Verify the engine on synthetic data (Yahoo is not reachable in CI):
the lookahead-bias demo and the transaction-cost model."""

import numpy as np
import pandas as pd
from backtest import (backtest, metrics, report, plot, trade_returns, sweep,
                      heatmap, walk_forward, basket, ml_backtest, vol_target,
                      long_short, portfolio, probabilistic_sharpe, deflated_sharpe,
                      survivors, survivorship_bias, alpha_beta)

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

# --- deflated Sharpe: the best cell, corrected for the number of trials ---
FS, SL = [5, 10, 20, 30, 50, 80], [20, 50, 100, 150, 200, 250]
# PSR is a probability and grows with sample length (more data -> more sure)
pos = pd.Series(np.random.default_rng(1).normal(0.0008, 0.01, 2000))
assert 0.0 <= probabilistic_sharpe(pos) <= 1.0
assert probabilistic_sharpe(pos) > probabilistic_sharpe(pos.iloc[:200])
assert probabilistic_sharpe(pos) > 0.9 and probabilistic_sharpe(-pos) < 0.1
# drift vs driftless noise: deflation barely dents a real edge but guts a lucky one
noise_c = pd.Series(100 * np.exp(np.cumsum(
    np.random.default_rng(7).normal(0, 0.20 / np.sqrt(252), n))), index=close.index)
_, psr_d, dsr_d = deflated_sharpe(close, sweep(close, FS, SL, COST), COST)
_, psr_n, dsr_n = deflated_sharpe(noise_c, sweep(noise_c, FS, SL, COST), COST)
assert dsr_d < psr_d and dsr_n < psr_n               # deflation always reduces
assert dsr_d > 0.9                                   # the genuine edge survives
assert dsr_n < dsr_d                                 # noise is deflated far harder
print(f"Deflated Sharpe of best cell: drift PSR {psr_d:.0%} -> DSR {dsr_d:.0%}; "
      f"noise PSR {psr_n:.0%} -> DSR {dsr_n:.0%}")
print("=> on noise the best cell looks plausible (PSR) until the deflated Sharpe,")
print("   correcting for the many pairs tried, exposes it as luck.")

# --- alpha vs beta: is the return skill, or just market exposure? ---
# leverage is beta, not alpha: any fixed fraction of the benchmark has that
# fraction as its beta, zero alpha and an R2 of 1
fixed = alpha_beta(0.6 * net["ret"], net["ret"])
assert np.isclose(fixed["Beta"], 0.6) and abs(fixed["Alpha"]) < 1e-12
assert np.isclose(fixed["R2"], 1.0) and fixed["Alpha t-stat"] == 0.0
# a riskless extra 1bp a day on top of it is pure alpha: 2.52%/yr, beyond doubt
sure = alpha_beta(0.6 * net["ret"] + 0.0001, net["ret"])
assert np.isclose(sure["Alpha"], 0.0001 * 252) and sure["Alpha t-stat"] == np.inf
# the crossover here matches buy-and-hold's Sharpe, yet its beta is about its
# share of days invested and its alpha is not significant: less risk, same edge
abx = alpha_beta(net["strat"], net["ret"])
assert abs(abx["Beta"] - net["position"].mean()) < 0.05
assert abs(abx["Alpha t-stat"]) < 2
# cross-check against textbook matrix OLS: coef = lstsq([1, bench], strat),
# t = intercept / sqrt(sigma^2 (X'X)^-1 [0, 0])
X = np.column_stack([np.ones(len(net)), net["ret"]])
coef = np.linalg.lstsq(X, net["strat"], rcond=None)[0]
res = net["strat"].to_numpy() - X @ coef
cov = res @ res / (len(net) - 2) * np.linalg.inv(X.T @ X)
assert np.isclose(abx["Beta"], coef[1]) and np.isclose(abx["Alpha"], coef[0] * 252)
assert np.isclose(abx["Alpha t-stat"], coef[0] / np.sqrt(cov[0, 0]))

# alpha needs structure to find: 40 random walks (none) against 40 markets with
# persistent bull/bear regimes. Timing earns alpha only in the second -- and
# even there one 6-year sample clears t > 2 in under half of the runs
def regime_walk(seed, p_stay=0.997, bull=0.25, bear=-0.35, vol=0.15):
    rng = np.random.default_rng(seed)
    state = np.cumsum(rng.random(n) > p_stay) % 2     # flips between 0=bull, 1=bear
    r = rng.normal(np.where(state == 0, bull, bear) / 252, vol / np.sqrt(252))
    return pd.Series(100 * np.exp(np.cumsum(r)), index=close.index)

def iid_walk(seed):
    r = np.random.default_rng(seed).normal(0.08 / 252, 0.20 / np.sqrt(252), n)
    return pd.Series(100 * np.exp(np.cumsum(r)), index=close.index)

def alpha_t(c):
    d = backtest(c, 50, 200, cost=COST)
    return alpha_beta(d["strat"], d["ret"])["Alpha t-stat"]

t_iid = np.array([alpha_t(iid_walk(s)) for s in range(40)])
t_reg = np.array([alpha_t(regime_walk(s)) for s in range(40)])
assert abs(t_iid.mean()) < 1                        # no structure: alpha centred on 0
assert t_reg.mean() > 0.8 and t_reg.mean() - t_iid.mean() > 0.8   # structure: alpha
assert np.mean(t_reg > 2) < 0.5                     # ...that one sample rarely proves
print(f"\nAlpha vs beta: crossover Sharpe {metrics(net['strat'])['Sharpe']:.2f} vs B&H "
      f"{metrics(net['ret'])['Sharpe']:.2f}, but beta {abx['Beta']:.2f} "
      f"({net['position'].mean():.0%} of days invested), alpha t = {abx['Alpha t-stat']:.2f}")
print(f"Mean alpha t over 40 runs: random walks {t_iid.mean():+.2f}, regime markets "
      f"{t_reg.mean():+.2f} (t > 2 in {np.mean(t_reg > 2):.0%} of them)")
print("=> a good Sharpe can be pure beta; alpha needs real structure, and even")
print("   then a few years of data rarely prove it.")

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

# --- long-short: go short instead of flat ---
ls = long_short(close, 50, 200, cost=0.0)
lf = backtest(close, 50, 200, cost=0.0)
assert ls["position"].isin([-1, 0, 1]).all()
# once both MAs are warm, the short book is the long/flat book mapped
# {0,1} -> {-1,+1}, i.e. ls_position == 2*lf_position - 1 wherever invested
warm = ls["position"].iloc[200:] != 0
assert (ls["position"].iloc[200:][warm] == (2 * lf["position"].iloc[200:] - 1)[warm]).all()
# shorting doubles turnover: a flip closes one side and opens the other
assert long_short(close, 50, 200, cost=COST)["trades"].sum() > net["trades"].sum()
# and on an up-drifting series the short legs fight the drift -> it trails B&H
assert metrics(ls["strat"])["Total return"] < metrics(ls["ret"])["Total return"]
print(f"\nLong-short (gross): total {metrics(ls['strat'])['Total return']:+.0%}, "
      f"Sharpe {metrics(ls['strat'])['Sharpe']:.2f}  vs long/flat "
      f"{metrics(lf['strat'])['Total return']:+.0%} / {metrics(lf['strat'])['Sharpe']:.2f}  "
      f"(B&H {metrics(ls['ret'])['Total return']:+.0%})")
print("=> always-in is not free: shorting fights the equity risk premium and "
      "doubles turnover.")

# --- portfolio: blend many strategies into one curve (diversification) ---
pcloses = {f"P{i}": pd.Series(100 * np.exp(np.cumsum(
    np.random.default_rng(100 + i).normal(
        0.10 / 252, (0.18 + 0.03 * i) / np.sqrt(252), n))), index=close.index)
    for i in range(6)}
comp = [metrics(backtest(c, 50, 200, cost=0.0)["strat"])["Sharpe"] for c in pcloses.values()]
pe = portfolio(pcloses, 50, 200, cost=0.0, scheme="equal")
wcols = [c for c in pe.columns if c.startswith("w_")]
# weights normalize to 1 on active days; no exposure during the warm-up
active = pe[wcols].sum(axis=1) > 0
assert np.allclose(pe.loc[active, wcols].sum(axis=1), 1.0)
assert (pe["strat"][~active] == 0).all()
# the free lunch: the diversified blend beats the average component Sharpe
assert metrics(pe["strat"])["Sharpe"] > np.mean(comp)
# inverse-vol risk-weighting: the calmer ASSET (lower trailing vol) gets more
# capital and every name carries the same risk budget (weight x vol is equal)
pv = portfolio(pcloses, 50, 200, cost=0.0, scheme="inverse_vol")
avol = pd.DataFrame({k: c.pct_change() for k, c in pcloses.items()}).rolling(60).std()
prev, day = pv.index[-2], pv.index[-1]            # weight on `day` uses vol on `prev`
hi, lo = avol.loc[prev].idxmax(), avol.loc[prev].idxmin()
assert pv[f"w_{hi}"].loc[day] < pv[f"w_{lo}"].loc[day]
budget = pv.loc[day, wcols].to_numpy(float) * avol.loc[prev].to_numpy(float)
assert np.allclose(budget, budget[0])
# and the book never sits all in cash after the warm-up: weighting by the
# STRATEGY's vol did exactly that whenever a sleeve was flat (1/0 -> inf/inf)
assert np.allclose(pv[wcols].iloc[61:].sum(axis=1), 1.0)

# trading gaps: each name skips different days (local holidays). A halted name
# keeps its capital and earns nothing until it trades again, so under both
# schemes the book stays fully invested and every listed name keeps a weight
holes = {k: c.drop(close.index[300 + 7 * i::40]) for i, (k, c) in enumerate(pcloses.items())}
for scheme in ("equal", "inverse_vol"):
    wh = portfolio(holes, 50, 200, scheme=scheme).filter(like="w_")
    assert np.allclose(wh.iloc[61:].sum(axis=1), 1.0) and (wh.iloc[61:] > 0).all().all()
# ...and a halt is never counted twice: with one name halted for a week, the
# equal-weight book is exactly the 1/N average of the sleeves, the halted one
# earning 0 meanwhile and its whole gap move on the day it reopens
gap = dict(pcloses, P2=pcloses["P2"].drop(close.index[800:805]))
sleeves = pd.DataFrame({k: backtest(c, 50, 200)["strat"] for k, c in gap.items()})
assert np.allclose(portfolio(gap, 50, 200, scheme="equal")["strat"],
                   sleeves.fillna(0.0).mean(axis=1))
# a halt that ends in a -90% delisting print still weighs that final loss
dying = pcloses["P3"].copy()
dying.iloc[1000:1003] = np.nan
dying.iloc[1003] = dying.iloc[999] * 0.1
dying.iloc[1004:] = np.nan
for scheme in ("equal", "inverse_vol"):
    pdie = portfolio(dict(pcloses, P3=dying), 50, 200, scheme=scheme)
    assert pdie.loc[close.index[1003], "w_P3"] > 0
# a stale price (100 identical closes, zero realized vol) neither sends the
# inverse-vol book to cash nor hands it to the stale name
stale = dict(pcloses, P0=pcloses["P0"].copy())
stale["P0"].iloc[700:800] = stale["P0"].iloc[700]
ws = portfolio(stale, 50, 200, scheme="inverse_vol").filter(like="w_")
assert np.allclose(ws.iloc[61:].sum(axis=1), 1.0) and ws["w_P0"].max() < 0.4

try:
    portfolio(pcloses, 50, 200, scheme="bogus")
    assert False, "unknown weighting scheme must raise"
except ValueError:
    pass
print(f"\nPortfolio of 6 strategies: avg component Sharpe {np.mean(comp):.2f} -> "
      f"equal-weight {metrics(pe['strat'])['Sharpe']:.2f}, "
      f"inverse-vol {metrics(pv['strat'])['Sharpe']:.2f}")
print("=> diversification is the free lunch: the blend's Sharpe tops the "
      "average component's.")

# --- survivorship bias: a universe picked by who exists today peeks ahead ---
def dying_universe(seed, names=30, floor=30.0, haircut=-0.30):
    """Fair-game stocks (zero expected daily return) starting at 100, riskier
    the higher the index. A name that closes below `floor` is delisted the
    next day at `haircut` to that close (about the average performance-
    delisting return) and is NaN afterwards."""
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(names):
        vol = (0.25 + 0.01 * i) / np.sqrt(252)
        px = pd.Series(100 * np.exp(np.cumsum(rng.normal(-vol ** 2 / 2, vol, n))),
                       index=close.index)
        below = np.flatnonzero(px.values[:-2] < floor)   # the death fits in the sample
        if below.size:
            px.iloc[below[0] + 1] = px.iloc[below[0]] * (1 + haircut)
            px.iloc[below[0] + 2:] = np.nan
        out[f"S{i:02d}"] = px
    return out

uni = dying_universe(0)
surv = survivors(uni)
last = {k: c.last_valid_index() for k, c in uni.items()}
dead = [k for k in uni if k not in surv]
# survivors() keeps exactly the names still listed on the final date
assert dead and all(last[k] == close.index[-1] for k in surv)
assert all(last[k] < close.index[-1] for k in dead)
# a live name that only skipped the final day (a local holiday) is dead by the
# default exact rule but survives with grace=1; a name with no prices is ignored
alive = dict(surv)
k0 = next(iter(alive))
alive[k0] = alive[k0].iloc[:-1]
assert k0 not in survivors(alive) and k0 in survivors(alive, grace=1)
assert set(survivors(dict(uni, Z=pd.Series(np.nan, index=close.index)))) == set(surv)
for bad in ({}, {"X": pd.Series(dtype=float)}):
    try:
        survivors(bad)
        assert False, "survivors() needs at least one name with a price"
    except ValueError:
        pass

# point-in-time portfolio: a delisting no longer truncates the curve, the book
# stays fully invested, and a dead name is held through its delisting day (its
# haircut counts) and gets zero weight after it
pit = portfolio(uni, 50, 200, cost=COST, scheme="equal")
assert pit.index.equals(close.index)
assert np.allclose(pit.filter(like="w_").sum(axis=1), 1.0)
for k in dead:
    assert pit.loc[last[k], f"w_{k}"] > 0 and (pit.loc[pit.index > last[k], f"w_{k}"] == 0).all()
# entry side: a late listing gets no capital before its first price
late = portfolio(dict(pcloses, P0=pcloses["P0"].iloc[300:]), 50, 200, scheme="equal")
assert late.index.equals(close.index) and (late["w_P0"].iloc[:300] == 0).all()

# survivorship IS lookahead: delete everything after day t and the point-in-time
# book up to t is unchanged; the survivors-only book is not, because who
# "survives" is decided by data that arrives after t
t = max(last[k] for k in dead)
cut = {k: c.loc[:t] for k, c in uni.items()}
assert np.allclose(portfolio(cut, 50, 200, cost=COST, scheme="equal")["strat"],
                   pit["strat"].loc[:t])
assert not np.allclose(portfolio(survivors(cut), 50, 200, cost=COST, scheme="equal")["strat"],
                       portfolio(surv, 50, 200, cost=COST, scheme="equal")["strat"].loc[:t])

# the lesson: dropping the dead inflates the backtest -- even basket()'s
# "judge the Average row" is fooled when the rows were chosen by survival
sb = survivorship_bias(uni, 50, 200, cost=COST)
assert sb.loc["Bias", "EW return"] > 0 and sb.loc["Bias", "EW Sharpe"] > 0
# its EW row is a direct run: an equal-weight book of the names listed each day
panel = pd.DataFrame(uni)
assert np.isclose(sb.loc["Point-in-time", "EW return"],
                  metrics((panel / panel.shift() - 1).mean(axis=1).fillna(0.0))["Total return"])
assert (basket(surv, 50, 200, COST).loc["Average", "B&H Sharpe"]
        > basket(uni, 50, 200, COST).loc["Average", "B&H Sharpe"])
# the crossover is usually flat before a name dies, so it dodges part of the bias
held_dead = np.mean([backtest(uni[k].dropna(), 50, 200)["position"].iloc[-1] for k in dead])
assert held_dead < 0.5 and sb.loc["Bias", "Strategy Sharpe"] < sb.loc["Bias", "EW Sharpe"]
# keeping the dead but dropping their delisting return is a second, quieter bias
nohc = survivorship_bias(dying_universe(0, haircut=0.0), 50, 200, cost=COST)
assert nohc.loc["Point-in-time", "EW return"] > sb.loc["Point-in-time", "EW return"]
print(f"\nSurvivorship: {len(dead)} of {len(uni)} zero-edge names delisted; the "
      f"crossover held {held_dead:.0%} of them on their delisting day")
print(sb.round(2).to_string())
print(f"=> survivors alone turn a {sb.loc['Point-in-time', 'EW return']:+.0%} "
      f"always-long equal-weight book into {sb.loc['Survivors only', 'EW return']:+.0%}; keeping the dead "
      f"but not their delisting returns still shows {nohc.loc['Point-in-time', 'EW return']:+.0%}.")

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
assert np.isnan(metrics(pd.Series([0.01]))["CAGR"])   # one point spans no return
# an account wiped out along the way compounds to -100% a year, not to NaN
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("error")                      # and quietly, no RuntimeWarning
    assert metrics(pd.Series([0.0, 0.5, -1.5, 0.1]))["CAGR"] == -1.0
print("\nRobustness: short-series guards, single-class folds, day-0 trade, flat metrics OK.")

plot(net, 50, 200, "SYNTHETIC", outfile="test_plot.png")
print("\nOK: engine + costs + trade stats + sweep + heatmap + deflated-Sharpe "
      "+ walk-forward + basket + ML signal + vol targeting + long-short + portfolio "
      "+ survivorship + alpha/beta verified.")
