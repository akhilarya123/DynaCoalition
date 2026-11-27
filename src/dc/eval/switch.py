"""How much do groupings change from step to step?

Two measures over an ``(T, N)`` array of per-step slot assignments:

* :func:`switch_rate` -- the spec metric (EXPERIMENT_PLAN.md Section 5): the
  fraction of agents whose argmax slot changes per step. Caveat from the plan:
  slot labels are arbitrary and can be permuted between steps without the
  grouping changing, and that counts as a (false) switch.
* :func:`pair_flip_rate` -- a relabelling-invariant companion: the fraction of
  agent pairs whose "same coalition?" answer flips between consecutive steps
  (1 - Rand index of consecutive partitions). Permuting slot labels leaves it
  at 0. Reported next to the spec metric so a high switch rate can be
  diagnosed as real regrouping vs label churn.

The "<= 1 switch per agent per 50 steps while static" success criterion is
``switch_rate <= 1/50 = 0.02`` measured over static stretches.
"""

from __future__ import annotations

import numpy as np

__all__ = ["switch_rate", "pair_flip_rate"]


def _check(assignments: np.ndarray) -> np.ndarray:
    a = np.asarray(assignments)
    if a.ndim != 2:
        raise ValueError(f"assignments must be (T, N), got shape {a.shape}")
    return a


def switch_rate(assignments: np.ndarray) -> float:
    """Mean fraction of agents whose slot changed between consecutive steps.

    NaN if there are fewer than 2 steps.
    """
    a = _check(assignments)
    if a.shape[0] < 2:
        return float("nan")
    return float((a[1:] != a[:-1]).mean())


def pair_flip_rate(assignments: np.ndarray) -> float:
    """Mean fraction of agent pairs whose same-coalition relation flipped per step.

    NaN if there are fewer than 2 steps or fewer than 2 agents.
    """
    a = _check(assignments)
    t, n = a.shape
    if t < 2 or n < 2:
        return float("nan")
    same = a[:, :, None] == a[:, None, :]  # (T, N, N)
    flipped = same[1:] != same[:-1]
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)  # each unordered pair once
    return float(flipped[:, upper].mean())
