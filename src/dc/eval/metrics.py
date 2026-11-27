"""Per-episode and per-evaluation-point metrics from dumped EvalRecords.

Aggregation rules (so the numbers are reproducible and comparable):

* Per-episode ARI is the mean over that episode's scored steps; the
  evaluation-point ARI is the mean over episodes (every episode weighs the
  same). Episodes with no scored step are dropped; no scored episode at all
  gives NaN, written as an empty cell.
* ``return_std`` / ``packages_delivered`` are across episodes (sample std).
* Methods without memberships (QMIX, VDN) dump ``assignments`` filled with
  ``-1``. ARI and switch columns are then ``None`` rather than a misleading 0.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from dc.contracts import METRICS_COLUMNS, ContractError, EvalRecord, blank_metrics_row
from dc.eval.ari import mean_ignoring_nan, stepwise_ari
from dc.eval.switch import pair_flip_rate, switch_rate

__all__ = ["has_memberships", "episode_metrics", "summarize_records", "metrics_row"]


def has_memberships(record: EvalRecord) -> bool:
    """False for methods that dump ``-1`` placeholders (QMIX, VDN)."""
    return int(np.asarray(record.assignments).min()) >= 0


def episode_metrics(record: EvalRecord) -> dict[str, float]:
    """Metrics for one evaluation episode. Membership metrics are NaN if absent."""
    record.validate()
    out: dict[str, float] = {
        "episode_return": float(record.episode_return),
        "packages_delivered": float(record.packages_delivered),
        "ari_contact": math.nan,
        "ari_all": math.nan,
        "switch_rate": math.nan,
        "pair_flip_rate": math.nan,
    }
    if not has_memberships(record):
        return out
    gt = np.asarray(record.gt_coalitions)
    asg = np.asarray(record.assignments)
    contact = np.asarray(record.in_contact, dtype=bool)
    out["ari_contact"] = mean_ignoring_nan(stepwise_ari(gt, asg, contact))
    out["ari_all"] = mean_ignoring_nan(stepwise_ari(gt, asg, None))
    out["switch_rate"] = switch_rate(asg)
    out["pair_flip_rate"] = pair_flip_rate(asg)
    return out


def _none_if_nan(x: float) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def summarize_records(records: Sequence[EvalRecord]) -> dict[str, float | None]:
    """Aggregate the episodes of ONE evaluation point (same run, same env_step)."""
    if not records:
        raise ContractError("no records to summarize")
    keys = {(r.run_id, r.algo, r.n_agents, r.seed, r.env_step) for r in records}
    if len(keys) != 1:
        raise ContractError(f"records come from different evaluation points: {sorted(keys)}")

    per_ep = [episode_metrics(r) for r in records]
    col = lambda name: np.array([m[name] for m in per_ep], dtype=np.float64)  # noqa: E731
    returns = col("episode_return")
    return {
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std(ddof=1)) if len(returns) > 1 else 0.0,
        "packages_delivered": float(col("packages_delivered").mean()),
        "ari_contact": _none_if_nan(mean_ignoring_nan(col("ari_contact"))),
        "ari_all": _none_if_nan(mean_ignoring_nan(col("ari_all"))),
        "switch_rate": _none_if_nan(mean_ignoring_nan(col("switch_rate"))),
        # Not a metrics.csv column (the schema is frozen); available for summary.json.
        "pair_flip_rate": _none_if_nan(mean_ignoring_nan(col("pair_flip_rate"))),
        "n_episodes": float(len(records)),
    }


def metrics_row(
    records: Sequence[EvalRecord],
    *,
    wall_clock_s: float | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One ``metrics.csv`` row for an evaluation point.

    ``diagnostics`` carries trainer-side columns the records cannot know:
    ``grad_gnn_td``, ``mean_max_membership``, ``k_eff``.
    """
    s = summarize_records(records)
    r0 = records[0]
    row = blank_metrics_row(
        run_id=r0.run_id,
        algo=r0.algo,
        n_agents=r0.n_agents,
        seed=r0.seed,
        env_step=r0.env_step,
        wall_clock_s=wall_clock_s,
        return_mean=s["return_mean"],
        return_std=s["return_std"],
        packages_delivered=s["packages_delivered"],
        ari_contact=s["ari_contact"],
        ari_all=s["ari_all"],
        switch_rate=s["switch_rate"],
    )
    if diagnostics:
        unknown = set(diagnostics) - set(METRICS_COLUMNS)
        if unknown:
            raise ContractError(f"not metrics.csv columns: {sorted(unknown)}")
        row.update(diagnostics)
    return row
