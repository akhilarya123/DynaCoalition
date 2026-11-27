"""Offline evaluator: EvalRecord files in, metrics.csv rows out.

The trainer never imports the metrics code; it only dumps raw records with
:func:`write_record` (numpy only, so no torch needed to read them back). Later::

    python -m dc.eval.evaluator experiments/<run_id>

computes ARI and switch rate from those files and merges them into the run's
``metrics.csv``.

On-disk layout (one ``.npz`` per evaluation episode)::

    <run_dir>/eval/step_<env_step, 9 digits>/ep_<episode_idx, 3 digits>.npz

Arrays: ``assignments`` (T, N) int (argmax_k S per step; fill with -1 for
methods that have no memberships), ``gt_coalitions`` (T, N) int,
``in_contact`` (T, N) bool. Scalars: run_id, algo, n_agents, seed, env_step,
episode_idx, episode_return, packages_delivered, and ``meta_json`` (a JSON
object; anything serialisable).

Merge rule for ``metrics.csv``: the trainer owns its own columns, so values it
already wrote are kept. This tool always (re)computes ``ari_contact``,
``ari_all`` and ``switch_rate``, and fills any other column only if it is empty.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from dc.contracts import METRICS_COLUMNS, ContractError, EvalRecord
from dc.eval.metrics import metrics_row

__all__ = [
    "record_path",
    "write_record",
    "read_record",
    "load_run_records",
    "evaluate_run",
    "fill_metrics_csv",
]

_ALWAYS_RECOMPUTED = ("ari_contact", "ari_all", "switch_rate")


# -- record IO --------------------------------------------------------------


def record_path(run_dir: str | Path, record: EvalRecord) -> Path:
    return (
        Path(run_dir)
        / "eval"
        / f"step_{int(record.env_step):09d}"
        / f"ep_{int(record.episode_idx):03d}.npz"
    )


def write_record(record: EvalRecord, run_dir: str | Path) -> Path:
    """Dump one evaluation episode. Call this from the trainer's eval loop."""
    record.validate()
    path = record_path(run_dir, record)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        assignments=np.asarray(record.assignments, dtype=np.int64),
        gt_coalitions=np.asarray(record.gt_coalitions, dtype=np.int64),
        in_contact=np.asarray(record.in_contact, dtype=bool),
        run_id=np.array(record.run_id),
        algo=np.array(record.algo),
        n_agents=np.array(record.n_agents),
        seed=np.array(record.seed),
        env_step=np.array(record.env_step),
        episode_idx=np.array(record.episode_idx),
        episode_return=np.array(float(record.episode_return)),
        packages_delivered=np.array(int(record.packages_delivered)),
        meta_json=np.array(json.dumps(record.meta)),
    )
    return path


def read_record(path: str | Path) -> EvalRecord:
    with np.load(path, allow_pickle=False) as z:
        return EvalRecord(
            run_id=str(z["run_id"]),
            algo=str(z["algo"]),
            n_agents=int(z["n_agents"]),
            seed=int(z["seed"]),
            env_step=int(z["env_step"]),
            episode_idx=int(z["episode_idx"]),
            episode_return=float(z["episode_return"]),
            packages_delivered=int(z["packages_delivered"]),
            assignments=z["assignments"],
            gt_coalitions=z["gt_coalitions"],
            in_contact=z["in_contact"],
            meta=json.loads(str(z["meta_json"])),
        )


def load_run_records(run_dir: str | Path) -> dict[int, list[EvalRecord]]:
    """All records of a run, grouped by ``env_step`` (sorted by episode index)."""
    files = sorted((Path(run_dir) / "eval").glob("step_*/ep_*.npz"))
    groups: dict[int, list[EvalRecord]] = defaultdict(list)
    for f in files:
        rec = read_record(f)
        rec.validate()
        groups[rec.env_step].append(rec)
    return {step: sorted(recs, key=lambda r: r.episode_idx) for step, recs in sorted(groups.items())}


# -- metrics ----------------------------------------------------------------


def evaluate_run(run_dir: str | Path) -> list[dict[str, Any]]:
    """One metrics row per evaluation point found under ``<run_dir>/eval``."""
    return [metrics_row(recs) for recs in load_run_records(run_dir).values()]


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        unknown = set(reader.fieldnames or []) - set(METRICS_COLUMNS)
        if unknown:
            raise ContractError(f"{path} has columns outside the schema: {sorted(unknown)}")
        return [{k: (None if v in ("", None) else v) for k, v in row.items()} for row in reader]


def fill_metrics_csv(run_dir: str | Path, csv_name: str = "metrics.csv") -> Path:
    """Merge computed metrics into ``<run_dir>/<csv_name>``, creating it if absent."""
    run_dir = Path(run_dir)
    path = run_dir / csv_name
    existing = {int(float(r["env_step"])): r for r in _read_csv(path) if r.get("env_step") is not None}

    for row in evaluate_run(run_dir):
        step = int(row["env_step"])
        old = existing.get(step)
        if old is None:
            existing[step] = row
            continue
        for col in METRICS_COLUMNS:
            if col in _ALWAYS_RECOMPUTED or old.get(col) is None:
                old[col] = row[col]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(METRICS_COLUMNS))
        writer.writeheader()
        for step in sorted(existing):
            writer.writerow({c: ("" if existing[step].get(c) is None else existing[step][c])
                             for c in METRICS_COLUMNS})
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Compute ARI/switch rate from dumped EvalRecords.")
    ap.add_argument("run_dir", type=Path, help="e.g. experiments/<run_id>")
    ap.add_argument("--csv", default="metrics.csv", help="file inside run_dir to create or update")
    args = ap.parse_args(argv)
    out = fill_metrics_csv(args.run_dir, args.csv)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
