"""Nested parameter selection.

Choosing parameters "inside the training window" sounds safe and is not. On this project,
three-fold minimum-t selection within the training window picked a near-fully-invested
quantile and degraded to out-of-sample t = 0.50.

Nested selection splits the training window again:

    [ inner-select 60% | inner-validate 40% ] [ test - touched once ]

Parameters are chosen on inner-select, screened on inner-validate, and only the surviving
configuration ever sees the test window. On this project that lifted the final result from
t = 0.50 to t = 1.51 - which was still rejected by the bootstrap, and that is the correct
outcome to report.

Dependencies: numpy, pandas.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Sequence

import numpy as np
import pandas as pd


@dataclass
class NestedResult:
    chosen: dict
    inner_select_score: float
    inner_validate_score: float
    test_score: float
    n_configs: int
    degradation: float

    def __str__(self) -> str:
        return (f"configurations searched : {self.n_configs}\n"
                f"chosen                  : {self.chosen}\n"
                f"inner-select score      : {self.inner_select_score:+.4f}\n"
                f"inner-validate score    : {self.inner_validate_score:+.4f}\n"
                f"test score (single use) : {self.test_score:+.4f}\n"
                f"degradation select->test: {self.degradation:+.4f}")


def _grid(space: dict[str, Sequence]) -> list[dict]:
    keys = list(space)
    return [dict(zip(keys, vals)) for vals in product(*(space[k] for k in keys))]


def nested_select(
    data: pd.DataFrame,
    param_space: dict[str, Sequence],
    score_fn: Callable[[pd.DataFrame, dict], float],
    *,
    cut: pd.Timestamp | str,
    inner_split: float = 0.6,
    min_validate_score: float | None = 0.0,
) -> NestedResult:
    """Select parameters without letting the test window influence the choice.

    Parameters
    ----------
    data : indexed by time.
    param_space : {parameter: candidate values}.
    score_fn : (subset, params) -> score. Higher is better.
    cut : train/test boundary. Everything at or after `cut` is touched exactly once.
    inner_split : fraction of the training window used for selection; the rest validates.
    min_validate_score : configurations failing this on inner-validate are discarded before
        the test pass. Set to None to skip the screen (and lose most of the protection).
    """
    cut = pd.Timestamp(cut)
    train = data[data.index < cut]
    test = data[data.index >= cut]
    if len(train) < 20 or len(test) < 5:
        raise ValueError("not enough data either side of the cut")

    k = int(len(train) * inner_split)
    inner_sel, inner_val = train.iloc[:k], train.iloc[k:]

    configs = _grid(param_space)
    scored = [(c, score_fn(inner_sel, c)) for c in configs]
    scored.sort(key=lambda cs: (-cs[1] if np.isfinite(cs[1]) else np.inf))

    chosen, sel_score, val_score = None, np.nan, np.nan
    for c, s in scored:
        v = score_fn(inner_val, c)
        if min_validate_score is None or (np.isfinite(v) and v > min_validate_score):
            chosen, sel_score, val_score = c, s, v
            break

    if chosen is None:
        # correct outcome, not an error: nothing survived the inner validation
        best_c, best_s = scored[0]
        return NestedResult(chosen={}, inner_select_score=best_s,
                            inner_validate_score=score_fn(inner_val, best_c),
                            test_score=np.nan, n_configs=len(configs),
                            degradation=np.nan)

    test_score = score_fn(test, chosen)
    return NestedResult(chosen, sel_score, val_score, test_score,
                        len(configs), test_score - sel_score)


def naive_select(
    data: pd.DataFrame,
    param_space: dict[str, Sequence],
    score_fn: Callable[[pd.DataFrame, dict], float],
    *,
    cut: pd.Timestamp | str,
) -> NestedResult:
    """The comparison case: pick the best configuration on the whole training window.

    Included so the degradation can be measured rather than asserted.
    """
    cut = pd.Timestamp(cut)
    train, test = data[data.index < cut], data[data.index >= cut]
    configs = _grid(param_space)
    scored = [(c, score_fn(train, c)) for c in configs]
    scored.sort(key=lambda cs: (-cs[1] if np.isfinite(cs[1]) else np.inf))
    best, s = scored[0]
    t = score_fn(test, best)
    return NestedResult(best, s, np.nan, t, len(configs), t - s)
