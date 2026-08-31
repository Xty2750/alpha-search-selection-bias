"""A ledger of every candidate attempted.

The denominator is the first thing lost in a research project and the last thing anyone
asks for. Six months in, the surviving strategy is vivid and the 380 dead ones are a vague
sense that "some things didn't work."

This records every attempt at the moment it is made, so the multiple-testing correction can
be computed from a real count rather than a recollection. On this project the count was 385
explicit candidates across seven protocols - a number that, once written down, made
continuing to search obviously wrong.

Dependencies: numpy, pandas.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Trial:
    protocol: str
    candidate: str
    hypothesis: str = ""
    outcome: str = "pending"     # pending | rejected | candidate | validated
    t_stat: float = float("nan")
    n_obs: int = 0
    note: str = ""
    logged_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class TrialLedger:
    """Append-only ledger. Entries are not edited or removed, only closed out."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.trials: list[Trial] = []
        if self.path and self.path.exists():
            self.trials = [Trial(**r) for r in json.loads(self.path.read_text())]

    def log(self, **kw) -> Trial:
        t = Trial(**kw)
        self.trials.append(t)
        self._flush()
        return t

    def close(self, protocol: str, candidate: str, outcome: str, **kw) -> None:
        for t in self.trials:
            if t.protocol == protocol and t.candidate == candidate:
                t.outcome = outcome
                for k, v in kw.items():
                    setattr(t, k, v)
        self._flush()

    def _flush(self) -> None:
        if self.path:
            self.path.write_text(json.dumps([asdict(t) for t in self.trials], indent=2))

    # ---------------------------------------------------------------- reporting

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(t) for t in self.trials])

    def n_trials(self) -> int:
        return len(self.trials)

    def bonferroni_threshold(self, alpha: float = 0.05) -> float:
        """Per-test alpha implied by the number of trials actually run."""
        n = max(self.n_trials(), 1)
        return alpha / n

    def required_abs_t(self, alpha: float = 0.05) -> float:
        """Two-sided normal critical value at the Bonferroni-adjusted level.

        With 385 trials at alpha = 0.05 this is roughly 3.9 - which is a useful thing to
        know *before* getting excited about a t of 2.5.
        """
        from math import erf, sqrt
        target = 1.0 - self.bonferroni_threshold(alpha) / 2.0
        lo, hi = 0.0, 10.0
        for _ in range(200):
            mid = (lo + hi) / 2
            cdf = 0.5 * (1 + erf(mid / sqrt(2)))
            lo, hi = (mid, hi) if cdf < target else (lo, mid)
        return (lo + hi) / 2

    def summary(self, alpha: float = 0.05) -> str:
        df = self.frame()
        if df.empty:
            return "ledger is empty"
        lines = [f"{len(df)} trials across {df['protocol'].nunique()} protocols"]
        for proto, g in df.groupby("protocol"):
            counts = g["outcome"].value_counts().to_dict()
            lines.append(f"  {proto:<12} {len(g):>4}  {counts}")
        adj = self.required_abs_t(alpha)
        lines.append("")
        lines.append(f"Bonferroni per-test alpha at {alpha}: {self.bonferroni_threshold(alpha):.2e}")
        lines.append(f"implied |t| required: {adj:.2f}")
        best = df["t_stat"].abs().max()
        if np.isfinite(best):
            lines.append(f"best |t| observed  : {best:.2f}"
                         f"  -> {'clears' if best >= adj else 'does NOT clear'} the adjusted bar")
        return "\n".join(lines)
