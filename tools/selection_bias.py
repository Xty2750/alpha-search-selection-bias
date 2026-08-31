"""Measure selection bias by running the honest and dishonest procedures side by side.

Both procedures fit no model on test data. Both would be described as "out-of-sample" in a
methods section. The only difference is whether the *selection step* - which factors, which
signs - was allowed to see the test period.

On this project the gap was rank IC +0.190 vs +0.081 and Sharpe 2.40 vs 1.11, on the same
factor family and the same test data.

Run it on a factor panel with no true edge and Procedure A will still report a respectable
out-of-sample IC. That is the whole point: the number is manufactured by the selection step,
not found in the data.

Dependencies: numpy, pandas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ProcedureResult:
    name: str
    selected: list[str]
    signs: dict
    oos_ic_mean: float
    oos_ic_t: float
    oos_mean_return: float
    oos_sharpe: float

    def __str__(self) -> str:
        return (f"{self.name}\n"
                f"  factors selected : {len(self.selected)}  {self.selected}\n"
                f"  OOS rank IC      : {self.oos_ic_mean:+.4f}  (t = {self.oos_ic_t:+.2f})\n"
                f"  OOS mean return  : {self.oos_mean_return:+.4%} per period\n"
                f"  OOS Sharpe       : {self.oos_sharpe:+.2f}")


def _rank_ic(factor: pd.DataFrame, fwd: pd.DataFrame, min_names: int = 3) -> pd.Series:
    """Per-period cross-sectional Spearman IC, vectorised.

    Spearman is Pearson on ranks, so rank each row and take a row-wise correlation. A
    per-period `.corr(method='spearman')` loop is the obvious implementation and is roughly
    two orders of magnitude slower on a panel of any size.
    """
    f, r = factor.align(fwd, join="inner")
    valid = f.notna() & r.notna()
    f, r = f.where(valid), r.where(valid)

    fr, rr = f.rank(axis=1), r.rank(axis=1)
    fr = fr.sub(fr.mean(axis=1), axis=0)
    rr = rr.sub(rr.mean(axis=1), axis=0)

    num = (fr * rr).sum(axis=1, min_count=1)
    den = np.sqrt((fr ** 2).sum(axis=1, min_count=1) * (rr ** 2).sum(axis=1, min_count=1))
    ic = (num / den).replace([np.inf, -np.inf], np.nan)
    return ic.where(valid.sum(axis=1) >= min_names).dropna()


def _score(factors: dict, fwd: pd.DataFrame, selected, signs,
           cost_per_side: float, periods_per_year: float):
    """Equal-weight rank combination of the selected factors, then long/short."""
    if not selected:
        return np.nan, np.nan, np.nan, np.nan
    combo = None
    for name in selected:
        r = factors[name].rank(axis=1, pct=True)   # pct=True matters - integer ranks break
        r = r * signs[name]                        # quantile thresholds silently
        combo = r if combo is None else combo + r
    combo = combo / len(selected)

    ic = _rank_ic(combo, fwd)
    ic_t = float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))) if len(ic) > 1 else np.nan

    # long top tercile, short bottom tercile, equal weight, cost on both legs both sides
    q = combo.rank(axis=1, pct=True)
    w = pd.DataFrame(0.0, index=combo.index, columns=combo.columns)
    w[q >= 2 / 3] = 1.0
    w[q <= 1 / 3] = -1.0
    w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    gross = (w * fwd).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    net = gross - turnover * cost_per_side
    sharpe = (float(net.mean() / net.std(ddof=1) * np.sqrt(periods_per_year))
              if net.std(ddof=1) > 0 else np.nan)
    return float(ic.mean()), ic_t, float(net.mean()), sharpe


def compare_procedures(
    factors: dict[str, pd.DataFrame],
    fwd_returns: pd.DataFrame,
    cut: pd.Timestamp | str,
    *,
    t_threshold: float = 2.0,
    cost_per_side: float = 0.0007,
    periods_per_year: float = 52.0,
) -> tuple[ProcedureResult, ProcedureResult]:
    """Return (procedure_A_full_sample_selection, procedure_B_train_only_selection).

    Parameters
    ----------
    factors : {name: DataFrame indexed by period, columns = instruments}
    fwd_returns : forward returns, same shape.
    cut : in/out-of-sample boundary.
    t_threshold : |t| screen applied to each factor's IC series during selection.
    """
    cut = pd.Timestamp(cut)
    is_mask = fwd_returns.index < cut
    oos_idx = fwd_returns.index[~is_mask]

    def select(window_idx) -> tuple[list[str], dict]:
        keep, signs = [], {}
        for name, f in factors.items():
            ic = _rank_ic(f.loc[window_idx], fwd_returns.loc[window_idx])
            if len(ic) < 3:
                continue
            t = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))
            if abs(t) >= t_threshold:
                keep.append(name)
                signs[name] = 1.0 if ic.mean() > 0 else -1.0
        return keep, signs

    # A: selection sees everything, including the period it will be evaluated on
    sel_a, sign_a = select(fwd_returns.index)
    # B: selection sees the training window only
    sel_b, sign_b = select(fwd_returns.index[is_mask])

    results = []
    for name, sel, sign in (("Procedure A - factors chosen on the FULL sample", sel_a, sign_a),
                            ("Procedure B - factors chosen on TRAINING only", sel_b, sign_b)):
        ic_m, ic_t, ret, sh = _score({k: v.loc[oos_idx] for k, v in factors.items()},
                                     fwd_returns.loc[oos_idx], sel, sign,
                                     cost_per_side, periods_per_year)
        results.append(ProcedureResult(name, sel, sign, ic_m, ic_t, ret, sh))
    return results[0], results[1]


def report_gap(a: ProcedureResult, b: ProcedureResult) -> str:
    lines = [str(a), "", str(b), "", "-" * 70]
    if np.isfinite(a.oos_ic_mean) and np.isfinite(b.oos_ic_mean):
        lines.append(f"selection bias in rank IC : {a.oos_ic_mean - b.oos_ic_mean:+.4f}")
    if np.isfinite(a.oos_sharpe) and np.isfinite(b.oos_sharpe):
        lines.append(f"selection bias in Sharpe  : {a.oos_sharpe - b.oos_sharpe:+.2f}")
    lines.append("")
    lines.append("Both are 'out-of-sample'. Only B is honest: A let the test period decide")
    lines.append("which factors and which signs to use.")
    return "\n".join(lines)
