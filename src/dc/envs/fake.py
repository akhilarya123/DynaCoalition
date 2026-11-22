"""A contract-shaped fake environment.

Its only job is to let both halves of the codebase run before either half is
finished: the training loop, mixer, losses and unit tests develop against this,
and the real VMAS environment is a one-line config swap later.

The geometry is deliberately plausible rather than random noise — agents drift,
packages sit still until delivered, and ``gt_coalitions`` comes from actual
contact distance — so the oracle, position-clustering and nearest-package
controls can be tested meaningfully against it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dc.contracts import EnvSpec, validate_batch

__all__ = ["FakeTransportEnv"]


def _to_backend(arrays: dict[str, np.ndarray], backend: str) -> dict[str, Any]:
    if backend == "numpy":
        return dict(arrays)
    try:
        import torch
    except ImportError:
        if backend == "torch":
            raise
        return dict(arrays)
    return {k: torch.from_numpy(v) for k, v in arrays.items()}


class FakeTransportEnv:
    """Generates batches that satisfy :func:`dc.contracts.validate_batch`.

    Parameters
    ----------
    spec:
        The shapes everything agrees on.
    seed:
        Fixes the stream, so two calls with the same seed give the same batch.
    world_size:
        Positions live in ``[-world_size, world_size]^2``.
    contact_radius:
        An agent within this distance of a package is "in contact" with it,
        which is what ``gt_coalitions`` records (0 = no contact, k = package k).
    """

    def __init__(
        self,
        spec: EnvSpec,
        *,
        seed: int = 0,
        world_size: float = 2.0,
        contact_radius: float = 0.35,
    ) -> None:
        self.spec = spec
        self.world_size = float(world_size)
        self.contact_radius = float(contact_radius)
        self._rng = np.random.default_rng(seed)

    # -- geometry ---------------------------------------------------------

    def _rollout_positions(self, b: int, t1: int) -> tuple[np.ndarray, np.ndarray]:
        """Agent positions ``(b, t1, N, 2)`` and package positions ``(b, t1, M, 2)``."""
        spec = self.spec
        rng = self._rng
        packages = rng.uniform(-self.world_size, self.world_size, size=(b, spec.n_packages, 2))

        # Each agent is pulled toward one package, so clusters actually form.
        targets = rng.integers(0, spec.n_packages, size=(b, spec.n_agents))
        start = rng.uniform(-self.world_size, self.world_size, size=(b, spec.n_agents, 2))

        pos = np.empty((b, t1, spec.n_agents, 2), dtype=np.float32)
        pos[:, 0] = start
        goal = np.take_along_axis(packages, targets[..., None], axis=1)  # (b, N, 2)
        for step in range(1, t1):
            drift = (goal - pos[:, step - 1]) * 0.08
            jitter = rng.normal(0.0, 0.03, size=(b, spec.n_agents, 2))
            nxt = pos[:, step - 1] + drift + jitter
            pos[:, step] = np.clip(nxt, -self.world_size, self.world_size)

        pkg = np.repeat(packages[:, None], t1, axis=1).astype(np.float32)
        return pos, pkg

    def _contact_labels(self, pos: np.ndarray, pkg: np.ndarray) -> np.ndarray:
        """``gt_coalitions``: 0 if no contact, else the nearer package's index."""
        delta = pos[:, :, :, None, :] - pkg[:, :, None, :, :]  # (B,T1,N,M,2)
        dist = np.linalg.norm(delta, axis=-1)  # (B,T1,N,M)
        nearest = dist.argmin(axis=-1)  # tie-break: the nearer package
        nearest_dist = np.take_along_axis(dist, nearest[..., None], axis=-1).squeeze(-1)
        labels = np.where(nearest_dist <= self.contact_radius, nearest + 1, 0)
        return labels.astype(np.int64)

    def _agent_features(self, pos: np.ndarray, pkg: np.ndarray) -> np.ndarray:
        """``x``: position, velocity, nearest-package offset and distance.

        Never actions or utilities (``Architecture.md`` Section 3).
        Padded or truncated to ``spec.x_dim`` so the spec stays authoritative.
        """
        vel = np.diff(pos, axis=1, prepend=pos[:, :1])
        delta = pos[:, :, :, None, :] - pkg[:, :, None, :, :]
        dist = np.linalg.norm(delta, axis=-1)
        nearest = dist.argmin(axis=-1)
        offset = np.take_along_axis(delta, nearest[..., None, None], axis=3).squeeze(3)
        nearest_dist = np.take_along_axis(dist, nearest[..., None], axis=-1)
        feats = np.concatenate([pos, vel, offset, nearest_dist], axis=-1).astype(np.float32)

        want = self.spec.x_dim
        have = feats.shape[-1]
        if have == want:
            return feats
        if have > want:
            return feats[..., :want]
        pad = self._rng.normal(0.0, 0.1, size=feats.shape[:-1] + (want - have,))
        return np.concatenate([feats, pad.astype(np.float32)], axis=-1)

    # -- public API -------------------------------------------------------

    def sample_batch(
        self,
        batch_size: int = 4,
        horizon: int | None = None,
        *,
        backend: str = "auto",
        ragged: bool = True,
    ) -> dict[str, Any]:
        """One contract-shaped batch of ``batch_size`` episodes.

        ``ragged`` truncates some episodes early and zeroes their mask, so
        padding bugs surface in development rather than during the grid.
        ``backend`` is ``"auto"`` (torch if importable, else numpy), ``"torch"``
        or ``"numpy"``.
        """
        spec = self.spec
        rng = self._rng
        t = int(horizon if horizon is not None else min(spec.episode_limit, 32))
        b = int(batch_size)
        t1 = t + 1

        pos, pkg = self._rollout_positions(b, t1)
        gt = self._contact_labels(pos, pkg)[:, :t]
        x = self._agent_features(pos, pkg)

        if ragged and t > 2:
            lengths = rng.integers(max(2, t // 2), t + 1, size=b)
        else:
            lengths = np.full(b, t)
        steps = np.arange(t)[None, :]
        mask = (steps < lengths[:, None]).astype(np.float32)[..., None]
        dones = (steps == (lengths[:, None] - 1)).astype(np.float32)[..., None]

        arrays: dict[str, np.ndarray] = {
            "obs": rng.normal(0.0, 1.0, size=(b, t1, spec.n_agents, spec.obs_dim)).astype(np.float32),
            "state": rng.normal(0.0, 1.0, size=(b, t1, spec.state_dim)).astype(np.float32),
            "positions": pos,
            "agent_features": x,
            "avail_actions": np.ones((b, t1, spec.n_agents, spec.n_actions), dtype=np.float32),
            "actions": rng.integers(0, spec.n_actions, size=(b, t, spec.n_agents, 1)).astype(np.int64),
            "gt_coalitions": gt,
            "rewards": (rng.normal(0.0, 0.5, size=(b, t, 1)) * mask).astype(np.float32),
            "dones": dones,
            "mask": mask,
        }
        batch = _to_backend(arrays, backend)
        validate_batch(batch, spec)
        return batch
