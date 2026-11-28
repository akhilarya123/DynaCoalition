"""Spatial controls: position clustering (k-means / soft RBF) and a fixed grid.

* :class:`PositionClusterProvider` (EXPERIMENT_PLAN.md "Position clustering"):
  clusters agents by position *at every step* into K groups. ``mode="kmeans"``
  gives hard one-hot rows; ``mode="soft"`` softens them with an RBF over the
  distances to the k-means centres. Same mixer as the full method, so it
  isolates "does the learned/reward-shaped grouping beat plain geometry".
* :class:`GridProvider`: hard assignment to a fixed grid of K map regions.
  Slot ``c`` always means the same region, which is what VAST's QMIX-over-K
  (per-slot hypernetwork weights) needs; per-step k-means labels would be an
  arbitrary permutation of inputs from one step to the next.

Background: neither control has a notion of "in no coalition", so column 0 is
always 0 and all N agents are spread over the K slots.

k-means is deterministic: farthest-point initialisation starting from agent 0,
then ``iters`` Lloyd steps, empty clusters keep their previous centre. No RNG,
so paired runs and tests are reproducible. Cost is O(B*T*N*K*iters), fine at
N <= 48.
"""

from __future__ import annotations

from typing import Any

from dc.contracts import ContractError
from dc.membership._xp import arange, cat, f32, i64, lib, onehot

__all__ = ["kmeans", "grid_shape", "grid_assignment", "PositionClusterProvider", "GridProvider"]


def _sq_dist(a: Any, b: Any) -> Any:
    """Pairwise squared distances: a ``(..., N, 2)``, b ``(..., K, 2)`` -> ``(..., N, K)``."""
    d = a[..., :, None, :] - b[..., None, :, :]
    return (d**2).sum(-1)


def _farthest_point_init(p: Any, k: int) -> Any:
    L = lib(p)
    idx = arange(p.shape[-2], like=p)
    c = p[..., 0, :]
    centres = [c]
    mind = ((p - c[..., None, :]) ** 2).sum(-1)  # (..., N)
    for _ in range(1, k):
        far = mind.argmax(-1)
        pick = f32(far[..., None] == idx)  # (..., N) one-hot of the farthest agent
        c = (p * pick[..., None]).sum(-2)
        centres.append(c)
        mind = L.minimum(mind, ((p - c[..., None, :]) ** 2).sum(-1))
    return L.stack(centres, -2)  # (..., K, 2)


def kmeans(p: Any, k: int, iters: int = 10) -> tuple[Any, Any, Any]:
    """Deterministic k-means over the agent axis.

    ``p`` is ``(..., N, 2)``. Returns ``(centres (..., K, 2), d2 (..., N, K),
    one_hot (..., N, K))`` for the final assignment.
    """
    if k < 1:
        raise ContractError(f"k must be >= 1, got {k}")
    L = lib(p)
    centres = _farthest_point_init(p, k)
    for _ in range(iters):
        oh = onehot(_sq_dist(p, centres).argmin(-1), k)  # (..., N, K)
        cnt = oh.sum(-2)  # (..., K)
        mean = (oh[..., None] * p[..., :, None, :]).sum(-3) / (cnt + (cnt == 0))[..., None]
        centres = L.where(cnt[..., None] > 0, mean, centres)
    d2 = _sq_dist(p, centres)
    return centres, d2, onehot(d2.argmin(-1), k)


class PositionClusterProvider:
    """K-means (hard) or RBF-softened (soft) position clusters, every step."""

    def __init__(self, mode: str = "kmeans", iters: int = 10, tau: float = 0.1) -> None:
        if mode not in ("kmeans", "soft"):
            raise ContractError(f"mode must be 'kmeans' or 'soft', got {mode!r}")
        if tau <= 0:
            raise ContractError(f"tau must be positive, got {tau}")
        self.mode, self.iters, self.tau = mode, int(iters), float(tau)

    def forward(self, *, p: Any, x: Any = None, h: Any = None, s: Any = None,
                gt: Any = None, k_slots: int, mask: Any = None) -> Any:
        _, d2, oh = kmeans(p, k_slots, self.iters)
        if self.mode == "kmeans":
            core = oh
        else:
            dmin = (d2 * oh).sum(-1)  # distance to the nearest centre
            e = lib(p).exp(-(d2 - dmin[..., None]) / self.tau)  # shifted: no underflow
            core = e / e.sum(-1)[..., None]
        return cat([core[..., :1] * 0.0, core], -1)  # column 0 (background) stays empty


def grid_shape(k: int) -> tuple[int, int]:
    """Most-square factorisation ``(nx, ny)`` of K with ``nx >= ny`` (6 -> 3x2, 12 -> 4x3)."""
    if k < 1:
        raise ContractError(f"k must be >= 1, got {k}")
    best = (k, 1)
    for ny in range(1, int(k**0.5) + 1):
        if k % ny == 0:
            best = (k // ny, ny)
    return best


def grid_assignment(p: Any, k: int, bounds: float = 1.0) -> Any:
    """Slot index in ``1..K`` of the grid cell containing each agent: ``(..., N)`` int64.

    The arena is ``[-bounds, bounds]^2`` (the transport world is +-1); agents
    outside it are clamped into the border cells.
    """
    L = lib(p)
    nx, ny = grid_shape(k)
    fx = (p[..., 0] + bounds) / (2.0 * bounds)
    fy = (p[..., 1] + bounds) / (2.0 * bounds)
    ix = i64(L.clip(L.floor(fx * nx), 0, nx - 1))
    iy = i64(L.clip(L.floor(fy * ny), 0, ny - 1))
    return 1 + ix * ny + iy


class GridProvider:
    """Hard, fixed map regions. Slot ``c`` is the same region at every step."""

    def __init__(self, bounds: float = 1.0) -> None:
        if bounds <= 0:
            raise ContractError(f"bounds must be positive, got {bounds}")
        self.bounds = float(bounds)

    def forward(self, *, p: Any, x: Any = None, h: Any = None, s: Any = None,
                gt: Any = None, k_slots: int, mask: Any = None) -> Any:
        return onehot(grid_assignment(p, k_slots, self.bounds), k_slots + 1)
