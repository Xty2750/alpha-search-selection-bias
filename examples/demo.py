"""Runs the tools on synthetic data with NO true edge. `python examples/demo.py`

Every factor below is pure noise, independent of the forward returns. Procedure A should
still report a respectable out-of-sample rank IC, because the selection step is doing the
work. That is the demonstration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.nested_selection import naive_select, nested_select      # noqa: E402
from tools.selection_bias import compare_procedures, report_gap     # noqa: E402
from tools.trial_ledger import TrialLedger                          # noqa: E402

rng = np.random.default_rng(3)


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------- 1. selection bias
rule("1. Selection bias - pure-noise factors, one instance then averaged over seeds")

periods = pd.date_range("2024-01-07", periods=130, freq="W")
instruments = [f"X{i:02d}" for i in range(12)]
cut = periods[int(len(periods) * 0.7)]


def noise_panel(seed: int, n_factors: int = 30):
    g = np.random.default_rng(seed)
    fwd = pd.DataFrame(g.normal(0, 0.05, (len(periods), len(instruments))),
                       index=periods, columns=instruments)
    factors = {f"f{i:02d}": pd.DataFrame(g.normal(0, 1, (len(periods), len(instruments))),
                                         index=periods, columns=instruments)
               for i in range(n_factors)}
    return factors, fwd


factors, fwd = noise_panel(3)
a, b = compare_procedures(factors, fwd, cut, t_threshold=1.8)
print(report_gap(a, b))

print("\nOne instance proves nothing, so the same experiment over 15 independent panels:")
rows = []
for s in range(100, 115):
    fa, fw = noise_panel(s)
    ra, rb = compare_procedures(fa, fw, cut, t_threshold=1.8)
    rows.append({"A_ic": ra.oos_ic_mean, "B_ic": rb.oos_ic_mean,
                 "A_sharpe": ra.oos_sharpe, "B_sharpe": rb.oos_sharpe})
avg = pd.DataFrame(rows).mean()
print(f"  mean OOS rank IC   A {avg['A_ic']:+.4f}   B {avg['B_ic']:+.4f}"
      f"   gap {avg['A_ic'] - avg['B_ic']:+.4f}")
print(f"  mean OOS Sharpe    A {avg['A_sharpe']:+.2f}   B {avg['B_sharpe']:+.2f}"
      f"   gap {avg['A_sharpe'] - avg['B_sharpe']:+.2f}")
print("\nEvery factor is independent noise, so the true value of both columns is zero.")
print("Procedure B scatters around zero. Procedure A does not - the gap is the bias.")

# ---------------------------------------------------------------- 2. nested selection
rule("2. Nested vs naive parameter selection - again, no true edge")

n = 900
idx = pd.date_range("2024-01-01", periods=n, freq="D")
data = pd.DataFrame({"x": rng.normal(0, 1, n),
                     "r": rng.normal(0, 0.02, n)}, index=idx)
split = idx[int(n * 0.7)]

space = {"window": [5, 10, 20, 40, 80], "q": [0.05, 0.10, 0.20, 0.30, 0.40]}


def score(df: pd.DataFrame, p: dict) -> float:
    sig = df["x"].rolling(p["window"], min_periods=p["window"]).mean()
    thr = sig.quantile(1 - p["q"])
    r = df.loc[sig >= thr, "r"].to_numpy()
    if len(r) < 8 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r))))   # t-statistic


print("-- naive: best configuration on the whole training window --")
print(naive_select(data, space, score, cut=split))
print("\n-- nested: chosen on inner-select, screened on inner-validate --")
print(nested_select(data, space, score, cut=split, inner_split=0.6))

print("\nA single instance is noise, so the same comparison over 30 independent panels:")
naive_claims, nested_claims, empties = [], [], 0
for s in range(200, 230):
    g = np.random.default_rng(s)
    d = pd.DataFrame({"x": g.normal(0, 1, n), "r": g.normal(0, 0.02, n)}, index=idx)
    naive_claims.append(naive_select(d, space, score, cut=split).inner_select_score)
    nr = nested_select(d, space, score, cut=split, inner_split=0.6)
    if not nr.chosen:
        empties += 1
    else:
        nested_claims.append(nr.inner_validate_score)

print(f"  mean t a naive search would REPORT as its finding : {np.mean(naive_claims):+.2f}")
print(f"  mean t nested selection reports (validation score): "
      f"{np.mean(nested_claims) if nested_claims else float('nan'):+.2f}")
print(f"  panels where nested selection returned NOTHING     : {empties}/30")
print("\nTrue edge is zero in every panel. The naive number is what gets written down and")
print("believed; the nested number is roughly half of it. With a screen this permissive")
print("(validation t > 0) some configuration almost always survives - raise the screen and")
print("nested selection starts returning nothing, which is the correct answer here.")

# ---------------------------------------------------------------- 3. trial ledger
rule("3. Trial ledger - what 385 trials does to the significance bar")

ledger = TrialLedger()
protocols = {"A": 36, "B": 21, "C": 18, "X1": 96, "X2": 36, "X3": 22, "X4": 108, "X5": 48}
for proto, count in protocols.items():
    for i in range(count):
        t = float(rng.normal(0, 1))
        ledger.log(protocol=proto, candidate=f"{proto}-{i:03d}", t_stat=t,
                   n_obs=int(rng.integers(40, 400)),
                   outcome="rejected" if abs(t) < 2 else "candidate")

print(ledger.summary())
print("\nWith this many trials, a t of 2.5 is unremarkable: out of 385 independent noise")
print("draws you expect roughly 19 above |t| = 2 by chance alone.")

print("\nDone.")
