"""The real VMAS transport env satisfies contracts.py (Set 1 "done when").

Skipped automatically where torch or vmas is missing (CI installs numpy only);
run it on the Mac before every push to ``feat/env-*``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vmas")

from dc.contracts import EnvSpec, validate_batch  # noqa: E402
from dc.envs.transport import TransportEnv, rollout_to_batch  # noqa: E402

E = 8
SPEC12 = EnvSpec(n_agents=12, n_packages=4, obs_dim=16, state_dim=66, x_dim=7, episode_limit=30)
SPEC24 = EnvSpec(n_agents=24, n_packages=8, obs_dim=16, state_dim=130, x_dim=7, episode_limit=30)


@pytest.fixture()
def env12():
    return TransportEnv(SPEC12, num_envs=E, device="cpu", seed=0)


def _stay(env):
    return torch.zeros(env.num_envs, env.spec.n_agents, dtype=torch.long)


def test_constructs_and_steps_at_n12(env12):
    data = env12.reset()
    assert data["obs"].shape == (E, 12, 16)
    assert data["state"].shape == (E, 66)
    assert data["positions"].shape == (E, 12, 2)
    assert data["agent_features"].shape == (E, 12, 7)
    assert data["gt_coalitions"].shape == (E, 12)
    data, reward, done, info = env12.step(_stay(env12))
    assert reward.shape == (E, 1) and done.shape == (E, 1)
    assert info["delivered_total"].shape == (E,)


def test_constructs_and_steps_at_n24():
    env = TransportEnv(SPEC24, num_envs=4, device="cpu", seed=0)
    env.reset()
    data, *_ = env.step(_stay(env))
    assert data["obs"].shape == (4, 24, 16)


def test_real_rollout_passes_validate_batch(env12):
    batch = rollout_to_batch(env12, horizon=10)
    b, t = validate_batch(batch, SPEC12)
    assert (b, t) == (E, 10)


def test_no_nans_after_many_random_steps(env12):
    batch = rollout_to_batch(env12, horizon=30)
    for key in ("obs", "state", "positions", "agent_features", "rewards"):
        assert torch.isfinite(batch[key]).all(), key


def test_contact_label_fires_when_an_agent_touches_a_package(env12):
    env12.reset()
    sc = env12.scenario
    pkg0 = sc.pkg_pos()[:, 0]  # (E, 2); rot is 0 right after reset
    touch = pkg0 + torch.tensor([sc.half_extents[0] + sc.agent_radius, 0.0])
    sc.world.agents[0].set_pos(touch, batch_index=None)
    labels = sc.gt_coalitions()
    assert (labels[:, 0] == 1).all()


def test_delivery_counts_and_package_respawns(env12):
    env12.reset()
    sc = env12.scenario
    sc.packages[0].set_pos(sc.goal_pos().clone(), batch_index=None)
    _, reward, _, info = env12.step(_stay(env12))
    assert (info["delivered_now"] >= 1).all()
    assert (info["delivered_total"] >= 1).all()
    assert (reward > 0).all()
    dist = (sc.pkg_pos()[:, 0] - sc.goal_pos()).norm(dim=-1)
    assert (dist > sc.goal_radius).all(), "delivered package must respawn off the goal"
    assert sc.pkg_pos().shape == (E, 4, 2), "package count is conserved"


@pytest.mark.parametrize(
    "action,axis,sign", [(1, 1, +1), (2, 1, -1), (3, 0, -1), (4, 0, +1)], ids=["up", "down", "left", "right"]
)
def test_action_directions_match_the_contract(env12, action, axis, sign):
    env12.reset()
    before = env12.scenario.agent_pos().clone()
    act = torch.full((E, 12), action, dtype=torch.long)
    for _ in range(3):
        data, *_ = env12.step(act)
    delta = (data["positions"] - before).mean(dim=(0, 1))
    assert sign * float(delta[axis]) > 0.0
    assert abs(float(delta[1 - axis])) < 0.5 * abs(float(delta[axis]))


def test_time_limit_is_the_only_done(env12):
    env12.reset()
    for t in range(SPEC12.episode_limit):
        _, _, done, _ = env12.step(_stay(env12))
        assert float(done.max()) == (1.0 if t == SPEC12.episode_limit - 1 else 0.0)
    with pytest.raises(RuntimeError):
        env12.step(_stay(env12))


def test_same_seed_gives_the_same_reset():
    a = TransportEnv(SPEC12, num_envs=E, seed=3).reset()["positions"]
    b = TransportEnv(SPEC12, num_envs=E, seed=3).reset()["positions"]
    assert torch.equal(a, b)
