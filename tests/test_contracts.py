"""Shared contract tests. Both owners run these before every push.

If one of these fails, the two halves of the codebase have drifted apart.
They use numpy only, so CI catches drift without installing torch.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from dc.contracts import (
    METRICS_COLUMNS,
    ContractError,
    EnvSpec,
    EvalRecord,
    MembershipProvider,
    blank_metrics_row,
    k_slots_for,
    resolve,
    validate_batch,
    validate_memberships,
)
from dc.envs.fake import FakeTransportEnv

N12 = EnvSpec(n_agents=12, n_packages=4, obs_dim=16, state_dim=24, x_dim=7)
N24 = EnvSpec(n_agents=24, n_packages=8, obs_dim=16, state_dim=24, x_dim=7)


# -- the K rule ------------------------------------------------------------


@pytest.mark.parametrize("spec,k", [(N12, 6), (N24, 12)])
def test_k_is_half_n_and_never_equals_m(spec: EnvSpec, k: int) -> None:
    assert spec.k_slots == k
    assert spec.n_slots == k + 1  # the background column
    assert spec.k_slots != spec.n_packages, "K = M would leak task knowledge"


def test_k_rule_rejects_odd_teams() -> None:
    with pytest.raises(ContractError):
        k_slots_for(13)


def test_spec_rejects_too_few_slots() -> None:
    with pytest.raises(ContractError):
        EnvSpec(n_agents=12, n_packages=6, obs_dim=8, state_dim=8, x_dim=7)


# -- the batch contract ----------------------------------------------------


def test_fake_env_satisfies_the_batch_contract() -> None:
    env = FakeTransportEnv(N12, seed=0)
    batch = env.sample_batch(batch_size=3, horizon=10, backend="numpy")
    b, t = validate_batch(batch, N12)
    assert (b, t) == (3, 10)


def test_observation_keys_carry_one_extra_step() -> None:
    """T transitions, T+1 states, so Double-Q can bootstrap."""
    env = FakeTransportEnv(N12, seed=1)
    batch = env.sample_batch(batch_size=2, horizon=8, backend="numpy")
    assert batch["obs"].shape[1] == 9
    assert batch["rewards"].shape[1] == 8


def test_batch_is_reproducible_from_the_seed() -> None:
    a = FakeTransportEnv(N12, seed=7).sample_batch(2, 6, backend="numpy")
    b = FakeTransportEnv(N12, seed=7).sample_batch(2, 6, backend="numpy")
    assert np.array_equal(a["positions"], b["positions"])


def test_mask_marks_padded_steps() -> None:
    env = FakeTransportEnv(N12, seed=3)
    batch = env.sample_batch(batch_size=8, horizon=16, backend="numpy")
    mask = batch["mask"]
    assert set(np.unique(mask)) <= {0.0, 1.0}
    # dones fire exactly once per episode, on its last real step
    assert np.array_equal(batch["dones"].sum(axis=1), np.ones((8, 1), dtype=np.float32))


def test_ground_truth_labels_stay_in_range() -> None:
    env = FakeTransportEnv(N12, seed=4)
    gt = env.sample_batch(4, 12, backend="numpy")["gt_coalitions"]
    assert gt.min() >= 0 and gt.max() <= N12.n_packages
    assert (gt == 0).any(), "some agents should be in contact with nothing"


def test_wrong_shape_names_the_offending_key() -> None:
    env = FakeTransportEnv(N12, seed=0)
    batch = env.sample_batch(2, 6, backend="numpy")
    batch["rewards"] = batch["rewards"][:, :-1]
    with pytest.raises(ContractError, match="rewards"):
        validate_batch(batch, N12)


def test_missing_and_extra_keys_are_rejected() -> None:
    env = FakeTransportEnv(N12, seed=0)
    batch = env.sample_batch(2, 6, backend="numpy")
    extra = dict(batch, surprise=np.zeros(1))
    with pytest.raises(ContractError, match="unexpected"):
        validate_batch(extra, N12)
    del batch["state"]
    with pytest.raises(ContractError, match="missing"):
        validate_batch(batch, N12)


# -- the membership contract ----------------------------------------------


class UniformProvider:
    """The simplest conforming provider: everyone equally in every slot."""

    def forward(self, *, p, x, h, s, gt, k_slots, mask):
        b, t, n = p.shape[0], p.shape[1], p.shape[2]
        return np.full((b, t, n, k_slots + 1), 1.0 / (k_slots + 1), dtype=np.float32)


def test_a_conforming_provider_satisfies_the_protocol() -> None:
    assert isinstance(UniformProvider(), MembershipProvider)


def test_provider_output_passes_validation() -> None:
    env = FakeTransportEnv(N12, seed=5)
    batch = env.sample_batch(2, 6, backend="numpy")
    S = UniformProvider().forward(
        p=batch["positions"][:, :-1],
        x=batch["agent_features"][:, :-1],
        h=None,
        s=batch["state"][:, :-1],
        gt=batch["gt_coalitions"],
        k_slots=N12.k_slots,
        mask=batch["mask"],
    )
    validate_memberships(S, N12, batch_size=2, horizon=6)


def test_rows_that_do_not_sum_to_one_are_rejected() -> None:
    S = np.full((2, 6, N12.n_agents, N12.n_slots), 0.5, dtype=np.float32)
    with pytest.raises(ContractError, match="sum to 1"):
        validate_memberships(S, N12, batch_size=2, horizon=6)


def test_negative_memberships_are_rejected() -> None:
    S = np.zeros((2, 6, N12.n_agents, N12.n_slots), dtype=np.float32)
    S[..., 0] = 2.0
    S[..., 1] = -1.0
    with pytest.raises(ContractError, match="negative"):
        validate_memberships(S, N12, batch_size=2, horizon=6)


# -- the results contract --------------------------------------------------


def test_metrics_row_has_every_column() -> None:
    row = blank_metrics_row(run_id="r0", algo="full", n_agents=12, seed=0, env_step=10_000)
    assert list(row) == list(METRICS_COLUMNS)
    assert row["ari_contact"] is None  # not yet computed, still well-formed


def test_metrics_row_rejects_unknown_columns() -> None:
    with pytest.raises(ContractError, match="not metrics.csv columns"):
        blank_metrics_row(retrun_mean=1.0)


def test_eval_record_validates_its_arrays() -> None:
    t, n = 20, N12.n_agents
    record = EvalRecord(
        run_id="r0",
        algo="full",
        n_agents=n,
        seed=0,
        env_step=10_000,
        episode_idx=0,
        episode_return=1.5,
        packages_delivered=3,
        assignments=np.zeros((t, n), dtype=np.int64),
        gt_coalitions=np.zeros((t, n), dtype=np.int64),
        in_contact=np.zeros((t, n), dtype=bool),
    )
    assert record.validate(N12) == (t, n)

    record.in_contact = np.zeros((t, n - 1), dtype=bool)
    with pytest.raises(ContractError, match="disagree on shape"):
        record.validate(N12)


def test_assignments_outside_the_slot_range_are_rejected() -> None:
    t, n = 5, N12.n_agents
    record = EvalRecord(
        run_id="r0",
        algo="full",
        n_agents=n,
        seed=0,
        env_step=0,
        episode_idx=0,
        episode_return=0.0,
        packages_delivered=0,
        assignments=np.full((t, n), N12.k_slots + 5, dtype=np.int64),
        gt_coalitions=np.zeros((t, n), dtype=np.int64),
        in_contact=np.zeros((t, n), dtype=bool),
    )
    with pytest.raises(ContractError, match="slot indices"):
        record.validate(N12)


# -- the config convention -------------------------------------------------


def test_resolve_imports_a_dotted_path() -> None:
    assert resolve("dc.envs.fake:FakeTransportEnv") is FakeTransportEnv


def test_resolve_rejects_a_malformed_target() -> None:
    with pytest.raises(ContractError):
        resolve("dc.envs.fake.FakeTransportEnv")


# -- the torch path (skipped in CI, which installs numpy only) -------------

needs_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch is not installed"
)


@needs_torch
def test_torch_and_numpy_backends_agree() -> None:
    import torch

    numpy_batch = FakeTransportEnv(N12, seed=11).sample_batch(2, 6, backend="numpy")
    torch_batch = FakeTransportEnv(N12, seed=11).sample_batch(2, 6, backend="torch")

    assert torch.is_tensor(torch_batch["obs"])
    assert validate_batch(torch_batch, N12) == (2, 6)
    assert np.allclose(numpy_batch["positions"], torch_batch["positions"].numpy())
    assert torch_batch["actions"].dtype == torch.int64


@needs_torch
def test_memberships_validate_on_tensors() -> None:
    import torch

    S = torch.softmax(torch.randn(2, 6, N12.n_agents, N12.n_slots), dim=-1)
    validate_memberships(S, N12, batch_size=2, horizon=6)
    with pytest.raises(ContractError, match="sum to 1"):
        validate_memberships(S * 2.0, N12, batch_size=2, horizon=6)


@needs_torch
def test_device_helpers_move_a_whole_batch() -> None:
    import torch

    from dc.device import get_device, to_device

    device = get_device("auto")
    batch = to_device(FakeTransportEnv(N12, seed=12).sample_batch(2, 6, backend="numpy"), device)
    assert {str(v.device) for v in batch.values()} == {str(device)}
    assert all(torch.is_tensor(v) for v in batch.values())
