"""Adjusted Rand index, in plain numpy (no sklearn needed at train time).

Matches ``sklearn.metrics.adjusted_rand_score`` including its conventions:
two partitions that induce the same same-cluster/different-cluster relation
on every pair score 1.0 (this covers "both all one cluster" and "both all
singletons"); one-cluster-vs-split scores 0.0; chance level is ~0; it can go
negative.

Two scoring modes (EXPERIMENT_PLAN.md Section 5), both *per timestep*:

* **contact** (main): only agents in contact with some package, skipping steps
  with fewer than 2 such agents. Background (column 0) is ignored because
  "not touching anything" has no true group to recover.
* **all agents** (secondary): every agent, with "no contact" (0) as its own
  true cluster and the background slot as its own predicted cluster.

Note for reading results: at a step where every contact agent touches the same
package, the true partition is a single cluster, so ARI is 1 if the method put
them all in one slot and 0 otherwise. That is correct, but it makes the metric
coarse early in episodes, so look at the per-step curve, not just the mean.
"""

from __future__ import annotations

import numpy as np

__all__ = ["adjusted_rand_index", "stepwise_ari", "mean_ignoring_nan"]


def _comb2(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float((x * (x - 1.0) / 2.0).sum())


def adjusted_rand_index(labels_true, labels_pred) -> float:
    """ARI between two labelings of the same items (any hashable-int labels)."""
    a = np.asarray(labels_true).ravel()
    b = np.asarray(labels_pred).ravel()
    if a.shape != b.shape:
        raise ValueError(f"label arrays differ in length: {a.shape} vs {b.shape}")
    n = a.size
    if n < 2:
        return 1.0  # nothing to disagree about

    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.int64)
    np.add.at(table, (ai, bi), 1)

    sum_nij = _comb2(table)
    sum_c = _comb2(table.sum(axis=1))
    sum_k = _comb2(table.sum(axis=0))
    total = n * (n - 1) / 2.0

    # pairs together in one labeling but split in the other
    if sum_c == sum_nij and sum_k == sum_nij:
        return 1.0
    expected = sum_c * sum_k / total
    denom = 0.5 * (sum_c + sum_k) - expected
    return float((sum_nij - expected) / denom)


def stepwise_ari(
    gt: np.ndarray,
    assignments: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    min_agents: int = 2,
) -> np.ndarray:
    """ARI at every timestep. Shapes ``(T, N)``; returns ``(T,)`` with NaN where skipped.

    ``mask`` (bool, ``(T, N)``) selects the agents scored at each step; steps
    with fewer than ``min_agents`` selected agents are NaN.
    """
    gt = np.asarray(gt)
    assignments = np.asarray(assignments)
    if gt.shape != assignments.shape or gt.ndim != 2:
        raise ValueError(f"need matching (T, N) arrays, got {gt.shape} and {assignments.shape}")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != gt.shape:
            raise ValueError(f"mask shape {mask.shape} != {gt.shape}")

    out = np.full(gt.shape[0], np.nan)
    for t in range(gt.shape[0]):
        sel = slice(None) if mask is None else mask[t]
        g, p = gt[t][sel], assignments[t][sel]
        if g.size >= min_agents:
            out[t] = adjusted_rand_index(g, p)
    return out


def mean_ignoring_nan(x) -> float:
    """Mean over non-NaN entries; NaN if there are none (no RuntimeWarning)."""
    x = np.asarray(x, dtype=np.float64)
    valid = x[~np.isnan(x)]
    return float(valid.mean()) if valid.size else float("nan")
