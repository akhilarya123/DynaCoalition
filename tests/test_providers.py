"""Membership providers: shape, normalisation, background use, and behaviour.

The math is tested on numpy (CI-safe); torch parity and VASTSpatial itself are
gated on torch. "Done when" for Set 4: every provider returns (B, T, N, K+1),
rows sum to 1, background used correctly.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from dc.contracts import ContractError, EnvSpec, MembershipProvider, validate_memberships
from dc.envs.fake import FakeTransportEnv
from dc.eval.ari import adjusted_rand_index
from dc.membership.nearest import NearestPackageProvider
from dc.membership.oracle import OracleProvider
from dc.membership.position import (
    GridProvider,
    PositionClusterProvider,
    grid_assignment,
    grid_shape,
    kmeans,
)
from dc.membership.random_s import RandomProvider
from dc.mixers.vast import subteam_values

N12 = EnvSpec(n_agents=12, n_packages=4, obs_dim=16, state_dim=24, x_dim=7)
N24 = EnvSpec(n_agents=24, n_packages=8, obs_dim=16, state_dim=24, x_dim=7)
B, T = 3, 10


def _inputs(spec=N12, seed=0, backend="numpy"):
    batch = FakeTransportEnv(spec, seed=seed).sample_batch(B, T, backend=backend, ragged=False)
    return dict(
        p=batch["positions"][:, :-1],
        x=batch["agent_features"][:, :-1],
        h=None,
        s=batch["state"][:, :-1],
        gt=batch["gt_coalitions"],
        mask=batch["mask"],
    )


def _all_providers():
    return {
        "oracle": OracleProvider(),
        "kmeans": PositionClusterProvider("kmeans"),
        "soft": PositionClusterProvider("soft"),
        "grid": GridProvider(),
        "random_iid": RandomProvider(mode="iid"),
        "random_ep": RandomProvider(mode="episode"),
        "nearest": NearestPackageProvider(offset_sign=-1),  # fake.py stores agent - package
    }


# -- the shared contract ---------------------------------------------------------


@pytest.mark.parametrize("spec", [N12, N24], ids=["N12", "N24"])
@pytest.mark.parametrize("name", list(_all_providers()))
def test_every_provider_returns_valid_memberships(name, spec):
    prov = _all_providers()[name]
    assert isinstance(prov, MembershipProvider)
    S = prov.forward(k_slots=spec.k_slots, **_inputs(spec))
    validate_memberships(S, spec, batch_size=B, horizon=T)  # shape, float, >= 0, rows sum to 1
    assert S.dtype == np.float32


@pytest.mark.parametrize("name", ["kmeans", "soft", "grid", "random_iid", "nearest"])
def test_providers_ignore_gt(name):
    inp = _inputs()
    prov = _all_providers()[name]
    if name == "random_iid":
        prov = RandomProvider(mode="iid", seed=5)
        other = RandomProvider(mode="iid", seed=5)
    else:
        other = prov
    a = prov.forward(k_slots=N12.k_slots, **inp)
    b = other.forward(k_slots=N12.k_slots, **{**inp, "gt": None})
    assert np.array_equal(a, b)


# -- oracle ----------------------------------------------------------------------


def test_oracle_is_the_one_hot_of_gt_with_background_for_no_contact():
    inp = _inputs()
    S = OracleProvider().forward(k_slots=N12.k_slots, **inp)
    gt = inp["gt"]
    assert np.array_equal(S.argmax(-1), gt)
    assert np.array_equal(S[..., 0] == 1.0, gt == 0)  # background used exactly for non-contact
    assert set(np.unique(S)) <= {0.0, 1.0}
    assert (gt == 0).any() and (gt > 0).any()


def test_oracle_never_uses_slots_beyond_the_packages():
    S = OracleProvider().forward(k_slots=N12.k_slots, **_inputs())
    assert S[..., N12.n_packages + 1 :].sum() == 0  # slots M+1..K stay empty


def test_oracle_rejects_missing_or_misaligned_or_out_of_range_labels():
    inp = _inputs()
    with pytest.raises(ContractError):
        OracleProvider().forward(k_slots=6, **{**inp, "gt": None})
    with pytest.raises(ContractError):
        OracleProvider().forward(k_slots=6, **{**inp, "gt": inp["gt"][:, :-1]})
    with pytest.raises(ContractError):
        OracleProvider().forward(k_slots=3, **inp)  # labels up to M=4 do not fit K=3


# -- position clustering ---------------------------------------------------------


def _blobs(k=3, per=4, spread=0.01, seed=0):
    """k tight blobs far apart; returns p (1, 1, k*per, 2) and the planted labels."""
    rng = np.random.default_rng(seed)
    centres = np.array([[-0.8, -0.8], [0.8, -0.8], [0.0, 0.8], [0.8, 0.8]])[:k]
    p = np.concatenate([c + rng.normal(0, spread, (per, 2)) for c in centres])
    return p[None, None].astype(np.float32), np.repeat(np.arange(k), per)


@pytest.mark.parametrize("mode", ["kmeans", "soft"])
def test_position_clustering_recovers_planted_blobs(mode):
    p, truth = _blobs()
    S = PositionClusterProvider(mode).forward(p=p, k_slots=3)
    assert adjusted_rand_index(truth, S[0, 0].argmax(-1)) == pytest.approx(1.0)


def test_kmeans_is_hard_leaves_background_empty_and_is_deterministic():
    inp = _inputs(N24)
    prov = PositionClusterProvider("kmeans")
    a = prov.forward(k_slots=N24.k_slots, **inp)
    assert np.array_equal(a, prov.forward(k_slots=N24.k_slots, **inp))
    assert set(np.unique(a)) <= {0.0, 1.0}
    assert a[..., 0].sum() == 0


def test_soft_mode_is_soft_but_agrees_with_hard_on_the_argmax():
    inp = _inputs()
    hard = PositionClusterProvider("kmeans").forward(k_slots=6, **inp)
    soft = PositionClusterProvider("soft", tau=0.1).forward(k_slots=6, **inp)
    assert soft.max(-1).mean() < 1.0
    assert np.array_equal(soft.argmax(-1), hard.argmax(-1))
    assert soft[..., 0].sum() == 0


def test_clustering_is_translation_invariant():
    inp = _inputs()
    prov = PositionClusterProvider("kmeans")
    a = prov.forward(k_slots=6, **inp)
    shifted = {**inp, "p": inp["p"] + np.float32(0.37)}
    assert np.array_equal(a, prov.forward(k_slots=6, **shifted))


def test_kmeans_survives_duplicate_points_and_more_clusters_than_distinct_points():
    p = np.zeros((1, 1, 6, 2), dtype=np.float32)
    p[0, 0, 3:] = 1.0  # only two distinct locations, asking for 4 clusters
    centres, d2, oh = kmeans(p, 4, iters=10)
    assert np.isfinite(centres).all() and np.isfinite(d2).all()
    assert (oh.sum(-1) == 1).all()


# -- grid ------------------------------------------------------------------------


def test_grid_shape_is_the_most_square_factorisation():
    assert grid_shape(6) == (3, 2)
    assert grid_shape(12) == (4, 3)
    assert grid_shape(7) == (7, 1)
    assert grid_shape(1) == (1, 1)


def test_grid_assigns_cell_centres_to_distinct_fixed_slots_and_clamps_outliers():
    nx, ny = grid_shape(6)  # 3 columns x 2 rows over [-1, 1]^2
    xs = (np.arange(nx) + 0.5) / nx * 2 - 1
    ys = (np.arange(ny) + 0.5) / ny * 2 - 1
    pts = np.array([[x, y] for x in xs for y in ys], dtype=np.float32)
    slots = grid_assignment(pts, 6)
    assert sorted(slots.tolist()) == [1, 2, 3, 4, 5, 6]  # one agent per cell, slots 1..K
    far = np.array([[5.0, 5.0], [-5.0, -5.0]], dtype=np.float32)
    assert grid_assignment(far, 6).tolist() == [1 + (nx - 1) * ny + (ny - 1), 1]


def test_grid_slots_do_not_depend_on_other_agents():
    inp = _inputs()
    S = GridProvider().forward(k_slots=6, **inp)
    p2 = inp["p"].copy()
    p2[:, :, 1:] = p2[:, :, :-1][:, :, ::-1]  # scramble everyone but agent 0
    S2 = GridProvider().forward(k_slots=6, **{**inp, "p": p2})
    assert np.array_equal(S[:, :, 0], S2[:, :, 0])


# -- random S --------------------------------------------------------------------


def test_random_s_uses_the_background_like_any_other_column():
    S = RandomProvider(alpha=1.0, mode="iid", seed=0).forward(
        p=np.zeros((20, 50, 12, 2), dtype=np.float32), k_slots=6
    )
    assert S[..., 0].mean() == pytest.approx(1 / 7, abs=0.01)
    assert S.max(-1).mean() < 1.0  # genuinely soft


def test_random_modes_differ_in_whether_groups_persist():
    p = np.zeros((2, 8, 12, 2), dtype=np.float32)
    ep = RandomProvider(mode="episode", seed=1).forward(p=p, k_slots=6)
    iid = RandomProvider(mode="iid", seed=1).forward(p=p, k_slots=6)
    assert all(np.array_equal(ep[:, t], ep[:, 0]) for t in range(8))
    assert not np.array_equal(iid[:, 1], iid[:, 0])


def test_random_s_is_reproducible_per_seed_and_differs_across_seeds():
    p = np.zeros((2, 4, 12, 2), dtype=np.float32)
    a = RandomProvider(seed=3).forward(p=p, k_slots=6)
    assert np.array_equal(a, RandomProvider(seed=3).forward(p=p, k_slots=6))
    assert not np.array_equal(a, RandomProvider(seed=4).forward(p=p, k_slots=6))


def test_random_s_rejects_bad_arguments():
    with pytest.raises(ContractError):
        RandomProvider(alpha=0.0)
    with pytest.raises(ContractError):
        RandomProvider(mode="sometimes")


# -- nearest package -------------------------------------------------------------


def _x_for(p, targets, sign=1):
    """Build x with offsets pointing from each agent to its package position."""
    off = sign * (targets - p)
    dist = np.linalg.norm(targets - p, axis=-1, keepdims=True)
    return np.concatenate([p, np.zeros_like(p), off, dist], -1).astype(np.float32)


def test_nearest_groups_agents_that_share_a_package_and_numbers_by_first_member():
    pkgs = np.array([[0.5, 0.5], [-0.5, 0.0]])
    which = np.array([1, 0, 1, 0, 0, 1])  # package index per agent
    p = np.random.default_rng(0).uniform(-1, 1, (6, 2))
    x = _x_for(p, pkgs[which])
    S = NearestPackageProvider().forward(p=p[None, None].astype(np.float32), x=x[None, None], k_slots=3)
    slots = S[0, 0].argmax(-1)
    assert adjusted_rand_index(which, slots) == pytest.approx(1.0)
    assert slots.tolist() == [1, 2, 1, 2, 2, 1]  # numbered by lowest-index member
    assert S[..., 0].sum() == 0


def test_nearest_honours_the_offset_sign_convention():
    pkgs = np.array([[0.5, 0.5], [-0.5, 0.0]])
    which = np.array([0, 1, 0, 1])
    p = np.random.default_rng(1).uniform(-1, 1, (4, 2))
    p4, x_neg = p[None, None].astype(np.float32), _x_for(p, pkgs[which], sign=-1)[None, None]
    good = NearestPackageProvider(offset_sign=-1).forward(p=p4, x=x_neg, k_slots=3).argmax(-1)
    bad = NearestPackageProvider(offset_sign=+1).forward(p=p4, x=x_neg, k_slots=3).argmax(-1)
    assert adjusted_rand_index(which, good[0, 0]) == pytest.approx(1.0)
    assert adjusted_rand_index(which, bad[0, 0]) < 1.0  # wrong sign scatters the targets


def test_nearest_sends_far_agents_to_the_background_when_a_radius_is_set():
    pkgs = np.array([[0.0, 0.0]])
    p = np.array([[0.1, 0.0], [0.2, 0.0], [0.9, 0.0]])
    x = _x_for(p, np.repeat(pkgs, 3, axis=0))[None, None]
    S = NearestPackageProvider(radius=0.5).forward(p=p[None, None].astype(np.float32), x=x, k_slots=3)
    assert S[0, 0].argmax(-1).tolist() == [1, 1, 0]


def test_nearest_caps_at_k_groups():
    p = np.arange(6, dtype=np.float64)[:, None] * np.array([1.0, 0.0])  # 6 distinct packages
    x = _x_for(p, p + 0.1)[None, None]  # each agent its own package
    S = NearestPackageProvider().forward(p=p[None, None].astype(np.float32), x=x, k_slots=3)
    assert S[0, 0].argmax(-1).max() == 3  # overflow groups share the last slot


def test_nearest_matches_ground_truth_on_contact_agents_in_the_fake_env():
    """Fake packages are points, so nearest-by-centre equals the contact label."""
    inp = _inputs(seed=2)
    S = NearestPackageProvider(offset_sign=-1).forward(k_slots=N12.k_slots, **inp)
    contact = inp["gt"] > 0
    scores = [
        adjusted_rand_index(inp["gt"][b, t][contact[b, t]], S[b, t].argmax(-1)[contact[b, t]])
        for b in range(B)
        for t in range(T)
        if contact[b, t].sum() >= 2
    ]
    assert scores and np.mean(scores) == pytest.approx(1.0)


# -- VAST pooling helper -----------------------------------------------------------


def test_subteam_values_sum_members_and_conserve_the_total():
    S = GridProvider().forward(k_slots=6, **_inputs())
    q = np.random.default_rng(0).normal(size=(B, T, 12)).astype(np.float32)
    v = subteam_values(S, q)
    assert v.shape == (B, T, 6)
    assert v.sum(-1) == pytest.approx(q.sum(-1), abs=1e-5)  # hard partition: nothing lost
    slots = S[0, 0].argmax(-1)
    assert v[0, 0, 2] == pytest.approx(q[0, 0][slots == 3].sum(), abs=1e-5)


def test_subteam_values_ignore_background_mass():
    S = OracleProvider().forward(k_slots=6, **_inputs())
    q = np.ones((B, T, 12), dtype=np.float32)
    v = subteam_values(S, q)
    gt = _inputs()["gt"]
    assert v.sum(-1) == pytest.approx((gt > 0).sum(-1).astype(np.float32))  # only in-contact agents


# -- torch (skipped where torch is not installed) ---------------------------------

needs_torch = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")


@needs_torch
@pytest.mark.parametrize("name", ["oracle", "kmeans", "soft", "grid", "nearest"])
def test_deterministic_providers_agree_between_numpy_and_torch(name):
    import torch

    prov = _all_providers()[name]
    a = prov.forward(k_slots=6, **_inputs(backend="numpy"))
    ti = _inputs(backend="torch")
    t = prov.forward(k_slots=6, **ti)
    assert torch.is_tensor(t) and t.dtype == torch.float32
    validate_memberships(t, N12, batch_size=B, horizon=T)
    assert np.allclose(a, t.numpy(), atol=1e-5)


@needs_torch
@pytest.mark.parametrize("mode", ["iid", "episode"])
def test_random_provider_works_on_torch(mode):
    import torch

    ti = _inputs(backend="torch")
    S = RandomProvider(mode=mode).forward(k_slots=6, **ti)
    assert torch.is_tensor(S)
    validate_memberships(S, N12, batch_size=B, horizon=T)


@needs_torch
def test_vast_spatial_pools_mixes_and_passes_gradients():
    import torch
    import torch.nn as nn

    from dc.mixers.vast import VASTSpatial

    class SumMixer(nn.Module):  # stand-in for QMIX: monotone, parameter-free
        def forward(self, values, state):
            return values.sum(-1, keepdim=True)

    ti = _inputs(backend="torch")
    q = torch.randn(B, T, 12, requires_grad=True)
    model = VASTSpatial(N12, SumMixer(), assignment="grid")
    q_tot, S, aux = model(q, None, ti["x"], ti["s"], ti["p"])
    assert q_tot.shape == (B, T, 1)
    validate_memberships(S, N12, batch_size=B, horizon=T)
    assert torch.allclose(q_tot.squeeze(-1), q.sum(-1), atol=1e-5)  # hard partition conserves the sum
    q_tot.sum().backward()
    assert torch.allclose(q.grad, torch.ones_like(q))  # every agent gets credit


@needs_torch
def test_vast_build_fails_loudly_on_a_bad_base_mixer_path():
    from dc.mixers.vast import VASTSpatial

    with pytest.raises(ContractError):  # module exists, attribute does not
        VASTSpatial.build(N12, base="dc.mixers.vast:NoSuchMixer")


# -- configs name real, constructible targets ---------------------------------------


def test_my_algo_configs_resolve_and_build_providers():
    from pathlib import Path

    import yaml

    from dc.contracts import resolve

    cfg_dir = Path(__file__).resolve().parents[1] / "configs" / "algo"
    names = {"oracle", "poscluster", "random_s", "nearest"}
    seen = set()
    for f in sorted(cfg_dir.glob("*.yaml")):
        cfg = yaml.safe_load(f.read_text())
        if cfg["name"] not in names:
            continue
        seen.add(cfg["name"])
        m = cfg["membership"]
        prov = resolve(m["target"])(**m["args"])
        assert isinstance(prov, MembershipProvider)
        S = prov.forward(k_slots=N12.k_slots, **_inputs())
        validate_memberships(S, N12, batch_size=B, horizon=T)
    assert seen == names

    vast = yaml.safe_load((cfg_dir / "vast_spatial.yaml").read_text())
    assert callable(resolve(vast["mixer"]["target"]))
