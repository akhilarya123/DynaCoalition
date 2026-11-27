"""Contact-label geometry on hand-built cases. numpy only, so CI runs it.

Box used throughout: length 0.4 x width 0.2, so half-extents (0.2, 0.1);
agent radius 0.03. An agent touching the +x edge sits at x = 0.2 + 0.03.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from dc.envs.labels import (
    box_signed_distance,
    contact_gaps,
    contact_labels,
    labels_from_gaps,
    on_goal,
)

HALF = (0.2, 0.1)
R = 0.03
THR = 0.02


def _labels(agents, pkgs, rots=None, thr=THR):
    agents = np.asarray(agents, dtype=np.float64)[None]  # (1, N, 2)
    pkgs = np.asarray(pkgs, dtype=np.float64)[None]  # (1, M, 2)
    rots = np.zeros((1, pkgs.shape[1])) if rots is None else np.asarray(rots, float)[None]
    return contact_labels(
        agents, pkgs, rots, half_extents=HALF, agent_radius=R, threshold=thr
    )[0]


def test_signed_distance_inside_edge_outside():
    pts = np.array([[[0.0, 0.0], [0.2, 0.0], [0.3, 0.0]]])
    sd = box_signed_distance(pts, np.zeros((1, 1, 2)), np.zeros((1, 1)), HALF)[0, :, 0]
    assert sd == pytest.approx([-0.1, 0.0, 0.1])  # centre is 0.1 from the nearest (y) edge


def test_corner_distance_is_euclidean():
    pt = np.array([[[0.23, 0.14]]])  # 0.03 right of, 0.04 above the corner
    sd = box_signed_distance(pt, np.zeros((1, 1, 2)), np.zeros((1, 1)), HALF)
    assert sd[0, 0, 0] == pytest.approx(0.05)


def test_rotation_swaps_the_long_axis():
    rot = np.full((1, 1), np.pi / 2)  # long side now runs along world y
    touching_top = np.array([[[0.0, 0.23]]])  # 0.2 + R along y: touching
    off_side = np.array([[[0.23, 0.0]]])  # box is only 0.1 wide along x now
    c = np.zeros((1, 1, 2))
    gap_top = contact_gaps(touching_top, c, rot, half_extents=HALF, agent_radius=R)
    gap_side = contact_gaps(off_side, c, rot, half_extents=HALF, agent_radius=R)
    assert gap_top[0, 0, 0] == pytest.approx(0.0, abs=1e-9)
    assert gap_side[0, 0, 0] == pytest.approx(0.23 - 0.1 - R)


def test_touching_agent_is_in_contact_and_distant_one_is_not():
    labels = _labels([[0.23, 0.0], [0.30, 0.0]], [[0.0, 0.0]])
    assert labels.tolist() == [1, 0]


def test_threshold_is_inclusive_of_small_gaps_only():
    agent = [[0.23, 0.14]]  # gap 0.05 - 0.03 = 0.02 to the corner
    assert _labels(agent, [[0.0, 0.0]], thr=0.025).tolist() == [1]
    assert _labels(agent, [[0.0, 0.0]], thr=0.015).tolist() == [0]


def test_agent_overlapping_the_box_counts_as_contact():
    assert _labels([[0.0, 0.0]], [[0.0, 0.0]]).tolist() == [1]  # negative gap


def test_labels_use_one_based_package_index():
    labels = _labels([[1.23, 0.0]], [[0.0, 0.0], [1.0, 0.0]])
    assert labels.tolist() == [2]


def test_tie_break_prefers_the_nearer_package():
    # Both boxes within a huge threshold; the agent is nearer to package 2.
    labels = _labels([[0.5, 0.0]], [[-1.0, 0.0], [1.0, 0.0]], thr=10.0)
    assert labels.tolist() == [2]


def test_exact_tie_goes_to_the_lowest_index():
    labels = _labels([[0.0, 0.0]], [[-1.0, 0.0], [1.0, 0.0]], thr=10.0)
    assert labels.tolist() == [1]


def test_dtype_range_and_leading_dims():
    rng = np.random.default_rng(0)
    b, t, n, m = 3, 5, 12, 4
    agents = rng.uniform(-1, 1, (b, t, n, 2))
    pkgs = rng.uniform(-1, 1, (b, t, m, 2))
    rots = rng.uniform(-np.pi, np.pi, (b, t, m))
    out = contact_labels(agents, pkgs, rots, half_extents=HALF, agent_radius=R, threshold=0.3)
    assert out.shape == (b, t, n)
    assert out.dtype == np.int64
    assert out.min() >= 0 and out.max() <= m
    assert (out > 0).any() and (out == 0).any()


def test_labels_from_gaps_matches_contact_labels():
    gaps = np.array([[[0.5, 0.01, 0.2], [0.5, 0.4, 0.3]]])
    assert labels_from_gaps(gaps, 0.02).tolist() == [[2, 0]]


def test_on_goal_is_a_centre_test():
    goal = np.array([[0.0, 0.0]])
    pkgs = np.array([[[0.1, 0.0], [0.2, 0.0]]])
    assert on_goal(pkgs, goal, 0.15).tolist() == [[True, False]]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_torch_and_numpy_agree():
    import torch

    rng = np.random.default_rng(1)
    agents = rng.uniform(-1, 1, (2, 12, 2))
    pkgs = rng.uniform(-1, 1, (2, 4, 2))
    rots = rng.uniform(-np.pi, np.pi, (2, 4))
    want = contact_labels(agents, pkgs, rots, half_extents=HALF, agent_radius=R, threshold=0.3)
    got = contact_labels(
        torch.from_numpy(agents), torch.from_numpy(pkgs), torch.from_numpy(rots),
        half_extents=HALF, agent_radius=R, threshold=0.3,
    )
    assert got.dtype == torch.int64
    assert np.array_equal(got.numpy(), want)
