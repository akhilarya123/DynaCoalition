"""Multi-package VMAS transport (fork of ``vmas/scenarios/transport.py``).

What changed from stock VMAS (EXPERIMENT_PLAN.md Section 1):

* ``M = N/3`` packages, each in its own box, all delivered to one goal circle.
* A delivered package **respawns** at a random valid spot, so coalitions must
  re-form inside an episode. ``done`` is therefore the time limit only.
* Exactly 5 discrete actions: stay, up, down, left, right.
* Per-agent observation is partial: own pos/vel plus the ``obs_k_packages``
  nearest packages (so ``obs_dim = 4 + 6 * k`` and it does not grow with M).
* Ground-truth contact labels (``gt_coalitions``) every step, from
  :mod:`dc.envs.labels`.
* Shared team reward = progress shaping + delivery bonus - collision penalty.

All game logic (progress, delivery, respawn) runs in ``post_step`` so that the
observation VMAS builds right afterwards already reflects respawned packages.

Public API (PROPOSED: to be agreed with the training-loop owner before it is
frozen)::

    env = TransportEnv(spec, num_envs=128, device="cpu", seed=0)
    data = env.reset()
    data, reward, done, info = env.step(actions)   # actions: (E, N) ints

``data`` holds the *observation-like* batch keys for the current state plus
``gt_coalitions``; ``reward`` and ``done`` are ``(E, 1)``. Packing T steps into
the ``(B, T+1, ...)`` replay batch is the replay buffer's job
(:func:`rollout_to_batch` shows the convention and is used by the tests).
"""

from __future__ import annotations

from typing import Any

from dc.contracts import ContractError, EnvSpec
from dc.envs.labels import contact_labels, on_goal

__all__ = ["TransportEnv", "TransportScenario", "state_dim_for", "obs_dim_for", "rollout_to_batch"]

# Our action index -> VMAS flat discrete index. VMAS discrete is 3 actions per
# axis (0 = stay, 1 = negative, 2 = positive), flattened x-major: flat = 3*ix + iy.
#   stay -> (0,0)=0   up (+y) -> (0,2)=2   down (-y) -> (0,1)=1
#   left (-x) -> (1,0)=3   right (+x) -> (2,0)=6
ACTION_NAMES = ("stay", "up", "down", "left", "right")
_VMAS_FLAT_ACTION = (0, 2, 1, 3, 6)


def obs_dim_for(k_packages: int) -> int:
    return 4 + 6 * k_packages


def state_dim_for(n_agents: int, n_packages: int) -> int:
    """agent pos+vel (4N) + package pos+vel (4M) + goal pos (2)."""
    return 4 * n_agents + 4 * n_packages + 2


def _make_scenario_class():
    """Defined lazily so importing this module never needs torch or vmas."""
    import torch
    from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
    from vmas.simulator.scenario import BaseScenario
    from vmas.simulator.utils import Color, ScenarioUtils

    class TransportScenario(BaseScenario):
        # ---------------------------------------------------------- world
        def make_world(self, batch_dim: int, device: torch.device, **kwargs):
            self.n_agents = kwargs.pop("n_agents", 12)
            self.n_packages = kwargs.pop("n_packages", 4)
            self.package_width = kwargs.pop("package_width", 0.15)
            self.package_length = kwargs.pop("package_length", 0.15)
            self.package_mass = kwargs.pop("package_mass", 50.0)
            self.agent_u_multiplier = kwargs.pop("agent_u_multiplier", 0.6)
            self.agent_radius = kwargs.pop("agent_radius", 0.03)
            self.goal_radius = kwargs.pop("goal_radius", 0.15)
            self.shaping_factor = kwargs.pop("shaping_factor", 10.0)
            self.delivery_bonus = kwargs.pop("delivery_bonus", 5.0)
            self.collision_penalty = kwargs.pop("collision_penalty", 0.01)
            self.contact_threshold = kwargs.pop("contact_threshold", 0.02)
            self.obs_k = kwargs.pop("obs_k_packages", 2)
            ScenarioUtils.check_kwargs_consumed(kwargs)
            if self.n_packages < self.obs_k:
                raise ContractError(f"need n_packages >= obs_k_packages ({self.obs_k})")

            self.world_semidim = 1.0
            self.half_extents = (self.package_length / 2, self.package_width / 2)
            self.pkg_circum = (self.half_extents[0] ** 2 + self.half_extents[1] ** 2) ** 0.5
            pad = 2 * self.agent_radius + max(self.package_length, self.package_width)

            world = World(
                batch_dim,
                device,
                x_semidim=self.world_semidim + pad,
                y_semidim=self.world_semidim + pad,
            )
            for i in range(self.n_agents):
                world.add_agent(
                    Agent(
                        name=f"agent_{i}",
                        shape=Sphere(self.agent_radius),
                        u_multiplier=self.agent_u_multiplier,
                    )
                )
            self.goal = Landmark(
                name="goal",
                collide=False,
                shape=Sphere(radius=self.goal_radius),
                color=Color.LIGHT_GREEN,
            )
            world.add_landmark(self.goal)
            self.packages = []
            for k in range(self.n_packages):
                pkg = Landmark(
                    name=f"package_{k}",
                    collide=True,
                    movable=True,
                    mass=self.package_mass,
                    shape=Box(length=self.package_length, width=self.package_width),
                    color=Color.RED,
                )
                self.packages.append(pkg)
                world.add_landmark(pkg)

            e = batch_dim
            self.shaping = torch.zeros(e, self.n_packages, device=device)
            self.delivered_total = torch.zeros(e, device=device)
            self.delivered_now = torch.zeros(e, device=device)
            self._rew = torch.zeros(e, device=device)
            self._obs_all = torch.zeros(e, self.n_agents, obs_dim_for(self.obs_k), device=device)
            return world

        # ------------------------------------------------- state accessors
        def agent_pos(self):
            return torch.stack([a.state.pos for a in self.world.agents], dim=1)  # (E,N,2)

        def agent_vel(self):
            return torch.stack([a.state.vel for a in self.world.agents], dim=1)

        def pkg_pos(self):
            return torch.stack([p.state.pos for p in self.packages], dim=1)  # (E,M,2)

        def pkg_vel(self):
            return torch.stack([p.state.vel for p in self.packages], dim=1)

        def pkg_rot(self):
            return torch.stack([p.state.rot.squeeze(-1) for p in self.packages], dim=1)  # (E,M)

        def goal_pos(self):
            return self.goal.state.pos  # (E,2)

        # --------------------------------------------------------- reset
        def reset_world_at(self, env_index: int | None = None):
            ScenarioUtils.spawn_entities_randomly(
                self.world.agents,
                self.world,
                env_index,
                min_dist_between_entities=self.agent_radius * 2,
                x_bounds=(-self.world_semidim, self.world_semidim),
                y_bounds=(-self.world_semidim, self.world_semidim),
            )
            occupied = self.agent_pos()
            if env_index is not None:
                occupied = occupied[env_index].unsqueeze(0)
            ScenarioUtils.spawn_entities_randomly(
                [self.goal] + self.packages,
                self.world,
                env_index,
                min_dist_between_entities=self.pkg_circum + self.goal_radius + 0.01,
                x_bounds=(-self.world_semidim, self.world_semidim),
                y_bounds=(-self.world_semidim, self.world_semidim),
                occupied_positions=occupied,
            )
            dist = (self.pkg_pos() - self.goal_pos()[:, None, :]).norm(dim=-1)
            sel = slice(None) if env_index is None else env_index
            self.shaping[sel] = (dist * self.shaping_factor)[sel]
            self.delivered_total[sel] = 0.0
            self.delivered_now[sel] = 0.0
            self._rew[sel] = 0.0
            self._refresh_obs()

        # ------------------------------------------------------ game logic
        def post_step(self):
            pkg, goal = self.pkg_pos(), self.goal_pos()
            dist = (pkg - goal[:, None, :]).norm(dim=-1)  # (E,M)
            shaping = dist * self.shaping_factor
            progress = (self.shaping - shaping).sum(-1)
            delivered = on_goal(pkg, goal, self.goal_radius)  # (E,M) bool
            n_del = delivered.sum(-1).float()

            ap = self.agent_pos()
            d = (ap[:, :, None, :] - ap[:, None, :, :]).norm(dim=-1)  # (E,N,N)
            eye = torch.eye(self.n_agents, dtype=torch.bool, device=d.device)
            n_coll = ((d < 2 * self.agent_radius) & ~eye).sum((1, 2)).float() / 2

            self._rew = progress + self.delivery_bonus * n_del - self.collision_penalty * n_coll
            self.shaping = shaping
            self.delivered_now = n_del
            self.delivered_total = self.delivered_total + n_del

            if bool(delivered.any()):
                self._respawn(delivered)
            self._refresh_obs()

        def _respawn(self, delivered, tries: int = 16):
            """Move delivered packages to random valid spots; zero their motion."""
            e, m = delivered.shape
            dev = delivered.device
            lim = self.world_semidim
            cand = (torch.rand(e, m, tries, 2, device=dev) * 2 - 1) * lim
            pkg, goal = self.pkg_pos(), self.goal_pos()
            d_goal = (cand - goal[:, None, None, :]).norm(dim=-1)  # (E,M,R)
            d_oth = (cand[:, :, :, None, :] - pkg[:, None, None, :, :]).norm(dim=-1)  # (E,M,R,M)
            self_mask = torch.eye(m, dtype=torch.bool, device=dev)[None, :, None, :]
            ok_oth = ((d_oth >= 2 * self.pkg_circum + 0.02) | self_mask).all(-1)
            ok = (d_goal >= self.goal_radius + self.pkg_circum + 0.05) & ok_oth
            first = ok.int().argmax(-1)  # first valid candidate (0 if none: rare)
            new = torch.gather(cand, 2, first[..., None, None].expand(e, m, 1, 2)).squeeze(2)
            for k, p in enumerate(self.packages):
                hit = delivered[:, k : k + 1]  # (E,1)
                p.set_pos(torch.where(hit, new[:, k], p.state.pos), batch_index=None)
                p.set_vel(torch.where(hit, torch.zeros_like(p.state.vel), p.state.vel), batch_index=None)
                p.set_rot(torch.where(hit, torch.zeros_like(p.state.rot), p.state.rot), batch_index=None)
                p.set_ang_vel(
                    torch.where(hit, torch.zeros_like(p.state.ang_vel), p.state.ang_vel),
                    batch_index=None,
                )
            newdist = (self.pkg_pos() - goal[:, None, :]).norm(dim=-1) * self.shaping_factor
            self.shaping = torch.where(delivered, newdist, self.shaping)

        # ----------------------------------------------- obs / reward / info
        def _nearest_k(self, ap):
            pkg = self.pkg_pos()
            d = (ap[:, :, None, :] - pkg[:, None, :, :]).norm(dim=-1)  # (E,N,M)
            idx = d.topk(self.obs_k, dim=-1, largest=False).indices  # (E,N,k)
            return pkg, idx

        def _refresh_obs(self):
            ap, av = self.agent_pos(), self.agent_vel()
            pkg, idx = self._nearest_k(ap)
            pv, goal = self.pkg_vel(), self.goal_pos()
            e, n, k = idx.shape
            take = lambda t: torch.gather(  # noqa: E731
                t[:, None].expand(e, n, t.shape[1], 2), 2, idx[..., None].expand(e, n, k, 2)
            )
            pp, vv = take(pkg), take(pv)  # (E,N,k,2)
            rel_agent = pp - ap[:, :, None, :]
            rel_goal = pp - goal[:, None, None, :]
            per_pkg = torch.cat([rel_agent, rel_goal, vv], dim=-1).reshape(e, n, -1)
            self._obs_all = torch.cat([ap, av, per_pkg], dim=-1)

        def observation(self, agent):
            return self._obs_all[:, self.world.agents.index(agent)]

        def reward(self, agent):
            return self._rew

        def done(self):
            return torch.zeros(self.world.batch_dim, dtype=torch.bool, device=self.world.device)

        # ------------------------------------------------- batch-key helpers
        def global_state(self):
            e = self.world.batch_dim
            return torch.cat(
                [
                    self.agent_pos().reshape(e, -1),
                    self.agent_vel().reshape(e, -1),
                    self.pkg_pos().reshape(e, -1),
                    self.pkg_vel().reshape(e, -1),
                    self.goal_pos(),
                ],
                dim=-1,
            )

        def agent_features(self):
            """x: position(2), velocity(2), offset to nearest package(2), distance(1)."""
            ap, av = self.agent_pos(), self.agent_vel()
            pkg, idx = self._nearest_k(ap)
            nearest = idx[..., 0]  # (E,N)
            e, n = nearest.shape
            tgt = torch.gather(pkg[:, None].expand(e, n, pkg.shape[1], 2), 2,
                               nearest[..., None, None].expand(e, n, 1, 2)).squeeze(2)
            off = tgt - ap
            return torch.cat([ap, av, off, off.norm(dim=-1, keepdim=True)], dim=-1)

        def gt_coalitions(self):
            return contact_labels(
                self.agent_pos(),
                self.pkg_pos(),
                self.pkg_rot(),
                half_extents=self.half_extents,
                agent_radius=self.agent_radius,
                threshold=self.contact_threshold,
            )

    return TransportScenario


class TransportEnv:
    """Thin wrapper turning the VMAS env into contract-shaped step dicts."""

    def __init__(
        self,
        spec: EnvSpec,
        *,
        num_envs: int = 128,
        device: str = "cpu",
        seed: int = 0,
        **physics: Any,
    ) -> None:
        import torch
        import vmas

        self.spec = spec
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self._torch = torch
        k = physics.get("obs_k_packages", 2)
        if spec.obs_dim != obs_dim_for(k):
            raise ContractError(f"obs_dim must be {obs_dim_for(k)} for k={k}, got {spec.obs_dim}")
        if spec.state_dim != state_dim_for(spec.n_agents, spec.n_packages):
            raise ContractError(
                f"state_dim must be {state_dim_for(spec.n_agents, spec.n_packages)}, got {spec.state_dim}"
            )
        if spec.x_dim != 7:
            raise ContractError(f"transport emits x_dim=7, got {spec.x_dim}")
        if spec.n_actions != 5:
            raise ContractError(f"transport has 5 actions, got {spec.n_actions}")

        self.scenario: Any = _make_scenario_class()()
        self._env = vmas.make_env(
            scenario=self.scenario,
            num_envs=self.num_envs,
            device=str(self.device),
            continuous_actions=False,
            max_steps=None,  # the wrapper owns the time limit
            seed=seed,
            n_agents=spec.n_agents,
            n_packages=spec.n_packages,
            **physics,
        )
        self._action_table = torch.tensor(_VMAS_FLAT_ACTION, dtype=torch.long, device=self.device)
        self._t = 0

    # ------------------------------------------------------------------
    def _data(self) -> dict[str, Any]:
        sc, torch = self.scenario, self._torch
        e = self.num_envs
        return {
            "obs": sc._obs_all.clone(),
            "state": sc.global_state(),
            "positions": sc.agent_pos(),
            "agent_features": sc.agent_features(),
            "avail_actions": torch.ones(e, self.spec.n_agents, self.spec.n_actions,
                                        device=self.device),
            "gt_coalitions": sc.gt_coalitions(),
        }

    def reset(self) -> dict[str, Any]:
        self._env.reset()
        self._t = 0
        return self._data()

    def step(self, actions: Any):
        """``actions``: ``(E, N)`` ints in ``[0, 4]``. Returns ``(data, reward, done, info)``."""
        torch = self._torch
        if self._t >= self.spec.episode_limit:
            raise RuntimeError("episode is over; call reset()")
        a = torch.as_tensor(actions, device=self.device).long()
        if a.shape != (self.num_envs, self.spec.n_agents):
            raise ValueError(f"actions must be {(self.num_envs, self.spec.n_agents)}, got {tuple(a.shape)}")
        if int(a.min()) < 0 or int(a.max()) >= self.spec.n_actions:
            raise ValueError("actions out of range [0, 4]")
        flat = self._action_table[a]  # (E, N)
        self._env.step([flat[:, i : i + 1] for i in range(self.spec.n_agents)])
        self._t += 1

        sc = self.scenario
        done_flag = self._t >= self.spec.episode_limit
        done = torch.full((self.num_envs, 1), float(done_flag), device=self.device)
        info = {
            "delivered_now": sc.delivered_now.clone(),
            "delivered_total": sc.delivered_total.clone(),
            "truncated": done.clone(),  # time limit only: bootstrap-vs-terminal is the loop's call
        }
        return self._data(), sc._rew.clone().unsqueeze(-1), done, info


def rollout_to_batch(env: TransportEnv, horizon: int, action_fn=None) -> dict[str, Any]:
    """Run ``horizon`` steps and pack them in the contract's T / T+1 layout.

    Every env contributes one episode (B = num_envs), all fully unmasked.
    Used by the tests and as the reference for the replay buffer's packing.
    """
    import torch

    spec = env.spec
    if action_fn is None:
        action_fn = lambda d: torch.randint(  # noqa: E731
            0, spec.n_actions, (env.num_envs, spec.n_agents), device=env.device
        )
    data = env.reset()
    obs_like = {k: [v] for k, v in data.items() if k != "gt_coalitions"}
    gt, acts, rews, dones = [data["gt_coalitions"]], [], [], []
    for _ in range(horizon):
        a = action_fn(data)
        data, r, d, _ = env.step(a)
        acts.append(a[..., None])
        rews.append(r)
        dones.append(d)
        for k in obs_like:
            obs_like[k].append(data[k])
        gt.append(data["gt_coalitions"])
    batch = {k: torch.stack(v, dim=1) for k, v in obs_like.items()}  # (B, T+1, ...)
    batch["actions"] = torch.stack(acts, dim=1)
    batch["gt_coalitions"] = torch.stack(gt[:-1], dim=1)  # label of the state each action was taken in
    batch["rewards"] = torch.stack(rews, dim=1)
    batch["dones"] = torch.stack(dones, dim=1)
    batch["mask"] = torch.ones_like(batch["rewards"])
    return batch


def __getattr__(name: str):
    # ``TransportScenario`` is built on first access so imports stay torch-free.
    if name == "TransportScenario":
        return _make_scenario_class()
    raise AttributeError(name)
