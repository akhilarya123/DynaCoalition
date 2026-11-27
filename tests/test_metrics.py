"""Evaluator metrics on hand-built cases (numpy only, so CI runs it).

Every expected number below is worked out by hand in the comment next to it.
"""

from __future__ import annotations

import csv
import importlib.util
import math

import numpy as np
import pytest

from dc.contracts import METRICS_COLUMNS, ContractError, EvalRecord
from dc.eval.ari import adjusted_rand_index, mean_ignoring_nan, stepwise_ari
from dc.eval.evaluator import (
    evaluate_run,
    fill_metrics_csv,
    load_run_records,
    read_record,
    write_record,
)
from dc.eval.metrics import episode_metrics, has_memberships, metrics_row, summarize_records
from dc.eval.switch import pair_flip_rate, switch_rate

# -- ARI ---------------------------------------------------------------------


def test_identical_partitions_up_to_relabelling_score_one():
    assert adjusted_rand_index([0, 0, 1, 1, 2], [7, 7, 3, 3, 9]) == pytest.approx(1.0)


def test_orthogonal_split_scores_minus_half():
    assert adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(-0.5)


def test_partial_agreement_scores_four_sevenths():
    # nij pairs = 1, true pairs = 2, pred pairs = 1, total = 6 -> (1-1/3)/(1.5-1/3) = 4/7
    assert adjusted_rand_index([0, 0, 1, 1], [0, 0, 1, 2]) == pytest.approx(4 / 7)


def test_degenerate_partitions_follow_sklearn_conventions():
    assert adjusted_rand_index([0, 0, 0], [1, 1, 1]) == 1.0  # both one cluster
    assert adjusted_rand_index([0, 1, 2], [5, 6, 7]) == 1.0  # both singletons
    assert adjusted_rand_index([0, 0, 0], [0, 1, 2]) == 0.0  # one cluster vs split


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        adjusted_rand_index([0, 1], [0, 1, 2])


@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None, reason="sklearn not installed")
def test_matches_sklearn_on_random_labelings():
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(2, 25))
        a = rng.integers(0, int(rng.integers(1, 6)), n)
        b = rng.integers(0, int(rng.integers(1, 8)), n)
        assert adjusted_rand_index(a, b) == pytest.approx(adjusted_rand_score(a, b), abs=1e-12)


# -- contact vs all-agents ARI ------------------------------------------------

GT = np.array([[1, 1, 2, 2, 0, 0]] * 3)
CONTACT = GT > 0


def test_contact_ari_ignores_background_agents_all_ari_does_not():
    # Contact agents grouped perfectly; the two non-contact agents are dumped into slot 3.
    asg = np.array([[3, 3, 5, 5, 3, 3]] * 3)
    c = stepwise_ari(GT, asg, CONTACT)
    a = stepwise_ari(GT, asg, None)
    assert c == pytest.approx([1.0] * 3)
    # true {1,1,2,2,0,0} vs pred {3,3,5,5,3,3}: nij pairs 3, true 3, pred 7, total 15
    # -> (3 - 1.4) / (5 - 1.4) = 4/9
    assert a == pytest.approx([4 / 9] * 3)


def test_background_in_slot_zero_matches_no_contact_cluster():
    asg = np.array([[3, 3, 5, 5, 0, 0]] * 3)
    assert stepwise_ari(GT, asg, None) == pytest.approx([1.0] * 3)


def test_steps_with_fewer_than_two_contact_agents_are_skipped():
    gt = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]])
    asg = np.array([[1, 2, 3, 4], [1, 1, 3, 4], [1, 2, 3, 4]])
    out = stepwise_ari(gt, asg, gt > 0)
    assert np.isnan(out[0]) and np.isnan(out[2])
    assert out[1] == pytest.approx(1.0)


def test_mean_ignoring_nan_handles_all_nan_without_warning():
    assert mean_ignoring_nan([np.nan, np.nan]) != mean_ignoring_nan([np.nan, np.nan])  # NaN
    assert mean_ignoring_nan([1.0, np.nan, 3.0]) == pytest.approx(2.0)


# -- switch rate --------------------------------------------------------------


def test_switch_rate_counts_changed_agents_per_step():
    asg = np.array([[0, 1], [0, 1], [1, 1], [1, 0]])
    # transitions: none | agent0 | agent1 -> (0 + 1 + 1) changes over 3 steps x 2 agents
    assert switch_rate(asg) == pytest.approx(2 / 6)


def test_static_assignments_never_switch():
    assert switch_rate(np.tile([1, 2, 3], (10, 1))) == 0.0
    assert pair_flip_rate(np.tile([1, 2, 3], (10, 1))) == 0.0


def test_pure_relabelling_is_a_false_switch_for_switch_rate_but_not_pair_flips():
    asg = np.array([[1, 1, 2, 2], [2, 2, 1, 1], [1, 1, 2, 2]])
    assert switch_rate(asg) == 1.0  # the documented caveat
    assert pair_flip_rate(asg) == 0.0  # the grouping never changed


def test_one_agent_moving_groups_flips_its_pairs():
    asg = np.array([[1, 1, 2, 2], [1, 1, 1, 2]])  # agent 2 joins group 1
    # pairs (0,2),(1,2) go apart->together; (2,3) together->apart; 3 of 6 pairs flip
    assert switch_rate(asg) == pytest.approx(1 / 4)
    assert pair_flip_rate(asg) == pytest.approx(3 / 6)


def test_too_short_inputs_give_nan():
    assert math.isnan(switch_rate(np.zeros((1, 4))))
    assert math.isnan(pair_flip_rate(np.zeros((1, 4))))


# -- records -> metrics -------------------------------------------------------


def _record(*, asg, gt=GT, ret=1.0, delivered=2, episode_idx=0, env_step=10_000, algo="full"):
    return EvalRecord(
        run_id="r0", algo=algo, n_agents=gt.shape[1], seed=0, env_step=env_step,
        episode_idx=episode_idx, episode_return=ret, packages_delivered=delivered,
        assignments=np.asarray(asg), gt_coalitions=gt, in_contact=gt > 0,
    )


PERFECT = np.array([[3, 3, 5, 5, 0, 0]] * 3)


def test_episode_metrics_on_a_perfect_grouping():
    m = episode_metrics(_record(asg=PERFECT))
    assert m["ari_contact"] == pytest.approx(1.0)
    assert m["ari_all"] == pytest.approx(1.0)
    assert m["switch_rate"] == 0.0


def test_no_membership_methods_report_none_not_zero():
    rec = _record(asg=np.full((3, 6), -1), algo="qmix")
    assert not has_memberships(rec)
    row = metrics_row([rec])
    assert row["ari_contact"] is None and row["ari_all"] is None and row["switch_rate"] is None
    assert row["return_mean"] == 1.0  # the rest of the row is still filled


def test_summary_averages_episodes_equally_and_uses_sample_std():
    a = _record(asg=PERFECT, ret=1.0, delivered=2, episode_idx=0)
    b = _record(asg=np.array([[3, 3, 5, 5, 3, 3]] * 3), ret=3.0, delivered=4, episode_idx=1)
    s = summarize_records([a, b])
    assert s["return_mean"] == pytest.approx(2.0)
    assert s["return_std"] == pytest.approx(math.sqrt(2.0))  # ddof=1
    assert s["packages_delivered"] == pytest.approx(3.0)
    assert s["ari_contact"] == pytest.approx(1.0)
    assert s["ari_all"] == pytest.approx((1.0 + 4 / 9) / 2)


def test_summary_rejects_records_from_different_evaluation_points():
    a = _record(asg=PERFECT, env_step=10_000)
    b = _record(asg=PERFECT, env_step=20_000, episode_idx=1)
    with pytest.raises(ContractError):
        summarize_records([a, b])


def test_metrics_row_has_every_column_and_validates_diagnostics():
    row = metrics_row([_record(asg=PERFECT)], diagnostics={"k_eff": 3.0, "grad_gnn_td": 0.5})
    assert list(row) == list(METRICS_COLUMNS)
    assert row["k_eff"] == 3.0 and row["grad_gnn_td"] == 0.5
    assert row["ari_contact"] == pytest.approx(1.0)  # diagnostics did not blank the metrics
    with pytest.raises(ContractError):
        metrics_row([_record(asg=PERFECT)], diagnostics={"nonsense": 1.0})


# -- disk round trip and csv merge ----------------------------------------------


def test_record_survives_a_disk_round_trip(tmp_path):
    rec = _record(asg=PERFECT)
    rec.meta = {"note": "x", "deliveries": [[4, 1]]}
    back = read_record(write_record(rec, tmp_path))
    assert back.run_id == "r0" and back.env_step == 10_000 and back.meta == rec.meta
    assert np.array_equal(back.assignments, rec.assignments)
    assert np.array_equal(back.in_contact, rec.in_contact) and back.in_contact.dtype == bool


def test_load_run_records_groups_by_step(tmp_path):
    for step in (20_000, 10_000):
        for ep in (1, 0):
            write_record(_record(asg=PERFECT, env_step=step, episode_idx=ep), tmp_path)
    groups = load_run_records(tmp_path)
    assert list(groups) == [10_000, 20_000]
    assert [r.episode_idx for r in groups[10_000]] == [0, 1]
    assert len(evaluate_run(tmp_path)) == 2


def test_fill_metrics_csv_keeps_trainer_values_and_adds_ari(tmp_path):
    write_record(_record(asg=PERFECT, env_step=10_000, ret=1.5), tmp_path)
    csv_path = tmp_path / "metrics.csv"
    with csv_path.open("w", newline="") as f:  # what the trainer wrote: no ARI, its own return
        w = csv.DictWriter(f, fieldnames=list(METRICS_COLUMNS))
        w.writeheader()
        w.writerow({"run_id": "r0", "algo": "full", "n_agents": 6, "seed": 0,
                    "env_step": 10_000, "return_mean": 99.0, "grad_gnn_td": 0.25})
    fill_metrics_csv(tmp_path)
    (row,) = list(csv.DictReader(csv_path.open()))
    assert float(row["return_mean"]) == 99.0  # trainer's value kept
    assert float(row["grad_gnn_td"]) == 0.25
    assert float(row["ari_contact"]) == pytest.approx(1.0)  # filled in
    assert float(row["packages_delivered"]) == 2.0  # empty cell filled from the records


def test_fill_metrics_csv_creates_the_file_and_is_idempotent(tmp_path):
    write_record(_record(asg=PERFECT, env_step=10_000), tmp_path)
    p = fill_metrics_csv(tmp_path)
    first = p.read_text()
    fill_metrics_csv(tmp_path)
    assert p.read_text() == first
    assert first.splitlines()[0].split(",") == list(METRICS_COLUMNS)


def test_csv_with_foreign_columns_is_rejected(tmp_path):
    (tmp_path / "metrics.csv").write_text("env_step,surprise\n1,2\n")
    write_record(_record(asg=PERFECT), tmp_path)
    with pytest.raises(ContractError):
        fill_metrics_csv(tmp_path)
