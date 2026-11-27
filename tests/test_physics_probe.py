"""Sanity checks on the physics probe itself (gated on torch + vmas)."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("torch")
pytest.importorskip("vmas")

from dc.tools.physics_probe import steps_to_travel, sweep  # noqa: E402


def test_steps_to_travel_handles_zero_speed():
    assert math.isinf(steps_to_travel(0.0, 1.0))
    assert steps_to_travel(0.01, 1.0) == pytest.approx(100.0)


def test_pushers_move_the_package_forward_and_more_pushers_move_it_faster():
    rows = sweep([10.0], [0, 1, 3], steps=20, travel=1.0, n_agents=12, n_packages=4, u_mult=0.6)
    disp = {int(r["pushers"]): r["disp"] for r in rows}
    assert abs(disp[0]) < 1e-3, "an unpushed package must stay put"
    assert disp[1] > disp[0] + 1e-4, "one pusher should move it at least a little"
    assert disp[3] > disp[1], "three pushers should move it further than one"


def test_a_full_speed_agent_does_not_tunnel_through_a_package():
    """Chosen physics (u=2.0, mass=3, friction=0.93): a free agent moves 0.08/step."""
    import torch

    from dc.contracts import EnvSpec
    from dc.envs.transport import TransportEnv, state_dim_for
    from dc.tools.physics_probe import _set

    e = 2
    spec = EnvSpec(n_agents=12, n_packages=4, obs_dim=16, state_dim=state_dim_for(12, 4),
                   x_dim=7, episode_limit=40)
    env = TransportEnv(spec, num_envs=e, seed=0, package_mass=3.0,
                       agent_u_multiplier=2.0, package_friction=0.93)
    env.reset()
    sc = env.scenario
    _set(sc.packages[0], torch, (0.0, 0.0), e)
    for j, p in enumerate(sc.packages[1:]):
        _set(p, torch, (-0.9 + 0.25 * j, -0.9), e)
    _set(sc.goal, torch, (0.9, 0.5), e)
    for i, agent in enumerate(sc.world.agents):
        _set(agent, torch, (-0.5, 0.0) if i == 0 else (-0.55 + 0.1 * ((i - 1) % 12), 0.9), e)

    act = torch.zeros(e, 12, dtype=torch.long)
    act[:, 0] = 4  # agent 0 runs right, straight into the package
    for _ in range(40):
        env.step(act)
        assert (sc.world.agents[0].state.pos[:, 0] < sc.pkg_pos()[:, 0, 0]).all(), "agent tunnelled"
    assert (sc.pkg_pos()[:, 0, 0] > 0.0).all(), "package should have been pushed"
