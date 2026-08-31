# Automated Alpha Search — and Measuring the Selection Bias

**385 pre-enumerated candidate signals across seven search protocols. None reached the
project's acceptance bar. The program was reported as not validated and stopped.**

That is the result. This repository exists because the interesting part is not the null —
it is that the null was *measured*, and that the measurement produced a number you can act
on:

> The same six-factor cross-sectional model gave out-of-sample rank IC **+0.190** and
> post-cost Sharpe **2.40** when its factors were chosen on the full sample, and IC
> **+0.081** with Sharpe **1.11** when factor choice was restricted to the training window.
> Same factor family, same combination method, same test data. The only difference is how
> much data the *selection step* was allowed to see.
>
> **The gap between those two numbers is the selection bias, quantified.**

And the honest version — the +0.081 one — had a bootstrap 95% confidence interval that
spanned zero. So it was recorded as a candidate, not a finding.

---

## The seven protocols

Every candidate was enumerated and frozen **before** its lock-box period was opened.

| Protocol | Candidates | Outcome | Most informative detail |
|---|---:|---|---|
| A — event-driven ML (single asset) | 36 | not validated | Lock-box: 88 trades, +8.08 bps/trade after 7bp/side, Sharpe 1.25 — but bootstrap 95% CI [−13.46, +29.59] bps, and removing the best 3 trades left +0.35 bps |
| B — 8-hour cross-sectional ranking | 21 | no candidate | Weak pre-cost signal existed; turnover consumed all of it |
| C — regime gating / mixture of experts | 18 | no candidate | Gating models underperformed a fixed slow expert |
| X1 — cross-sectional imbalance events | 96 | not validated | +45 to +50 bps/trade in two development periods → −2.79 bps in the frozen period |
| X2 — open-interest / funding events | 36 | no candidate | A few positive point estimates, never with enough events |
| X3 — medium-horizon trend | 22 | no candidate | Different speeds worked in different periods; no stable plateau |
| X4 — long-wick rejection patterns | 108 | no candidate | Systematically negative across both development periods |
| X5 — daily cross-sectional crowding rank | 48 | no candidate | Sign-flipped between periods |

Two protocols came close enough to be worth stating precisely, because "close" is where
self-deception lives:

- **A** produced a positive point estimate on data it had never seen. It failed on two
  independent grounds: the confidence interval crossed zero, and the entire result came
  from three trades.
- **X1** produced +45–50 bps per trade in two consecutive development windows and then
  −2.79 bps over 18 trades in the frozen window. Two out of three is not two out of two.

## The selection-bias experiment

The cleanest thing in this repository, and it is reproducible on any dataset.

**Procedure A — the way it is usually done.** Scan all factors on the full sample. Keep
those with |t| ≥ 2. Fix each factor's sign from its full-sample behaviour. Combine with
equal weights, rank cross-sectionally, and report out-of-sample performance.

**Procedure B — the honest version.** Identical, except factor scanning, the |t| ≥ 2 screen,
and sign determination all happen **using training data only**. Out-of-sample is touched
once, at the end, to report.

| | Procedure A | Procedure B |
|---|---:|---:|
| Out-of-sample rank IC | +0.190 | +0.081 |
| IC t-statistic | — | 2.13 |
| Post-cost Sharpe (7 bps/side) | 2.40 | 1.11 |
| Bootstrap 95% CI on mean period return | — | **contains zero** |

Procedure B survived several robustness checks that Procedure A never needed to face: five
calendar offsets all positive, leave-one-factor-out all positive, 11 of 12 instruments
contributing positively. It still did not clear the bootstrap.

**Both procedures are "out-of-sample" by the usual description.** Neither touches test data
during model fitting. The difference is that Procedure A lets test data influence *which
model gets fitted*, which is the part that rarely appears in a methods section.

Implemented in [`tools/selection_bias.py`](tools/selection_bias.py); run it on synthetic
data with `python examples/demo.py` and watch a factor set with **zero true edge** produce a
respectable out-of-sample IC under Procedure A.

## Nested selection

The same failure recurs one level down, when parameters rather than features are chosen.

A formula-search candidate had its window and quantile parameters chosen by three-fold
minimum-t inside the training window. That selected a near-fully-invested quantile, and
out-of-sample performance degraded immediately to t = 0.50.

Replacing it with **nested selection** — choose parameters on the first 60% of training,
validate on the remaining 40%, touch out-of-sample once — improved the final result to
t = 1.51. It was still rejected, because the bootstrap CI contained zero.

> In a single-instrument two-year price sample, "best on the training window" carries no
> reliable mapping to "good out-of-sample." The endpoint of any automated search must be
> data that participated in no selection step at all.

See [`tools/nested_selection.py`](tools/nested_selection.py).

## Genetic programming: fitness functions overfit too

Formula search at daily, 1-hour and 15-minute resolutions.

| Version | Fitness function | Result |
|---|---|---|
| v1 | training-window t-statistic | Best formula reached training t = 2.96. **0 of 20** top formulas had positive out-of-sample IC |
| v2 | minimum t across three internal walk-forward folds, all three required positive | **4 of 20** passed the initial screen. All four were rejected at the bootstrap stage |

Two things made v1 worse than useless. The fitness function selected noise formulas
directly, and the candidate pool was flooded with algebraically equivalent expressions —
`min(close, close)` is `close` — so the top-20 list was often one formula wearing twenty
costumes. v2 added semantic deduplication alongside the fold-based fitness.

**An overfitting-resistant fitness function is necessary, not sufficient.** It moved the
pass rate from 0/20 to 4/20 and changed no conclusion.

## Why the program was stopped rather than tuned

At 385 explicit candidates plus the internal degrees of freedom of each model, the same
two-year sample had been searched hard. Continuing to adjust thresholds, drop months or
select instruments would keep raising the probability of finding a high-Sharpe
configuration — and every increment of that probability comes from multiple-testing bias,
not from information.

What the data did say, which is worth more than another parameter sweep:

1. **Frequent directional prediction cannot cover costs here.** Protocol B had a real
   pre-cost signal and turnover ate all of it.
2. **Non-linear models showed no incremental value.** Shallow gradient boosting
   underperformed Ridge and simple slow rules in most protocols — evidence that the
   effective sample is far smaller than the row count suggests.
3. **Regime instability is the dominant failure mode**, and training a gating model on the
   same two years does not fix it.
4. **Low-frequency extreme events are the more promising direction**, and they are exactly
   where sample size is smallest.
5. **Candle-pattern narratives did not survive.** 108 long-wick configurations were
   systematically negative.

The correct next step is more *independent information* — order book, trade-level aggressor
side, liquidations, basis, cross-venue flow — not more searches over the same data.

## Reusable tools

| Module | What it does |
|---|---|
| [`tools/selection_bias.py`](tools/selection_bias.py) | Runs Procedure A and Procedure B side by side on the same data and reports the gap. Point it at your own factor panel. |
| [`tools/nested_selection.py`](tools/nested_selection.py) | Nested parameter selection: inner selection window, inner validation window, single-use test. |
| [`tools/trial_ledger.py`](tools/trial_ledger.py) | Records every candidate attempted, so the denominator survives the project. Reports effective trial count and a Bonferroni-adjusted threshold. |

## What is not here

Model configurations, evolved formulas, factor definitions, and raw data. The protocols,
the negative results and the bias measurement are the transferable parts.

## License

MIT — see [LICENSE](LICENSE).
