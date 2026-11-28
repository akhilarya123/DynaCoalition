"""Nearest-package control: group agents by which package is nearest to them.

A task-aware heuristic (drop first if compute is tight). Providers receive
``x`` but not package identities, so identity is recovered from geometry:
``x`` carries each agent's offset to its nearest package, hence

    target_i = p_i + offset_sign * offset_i

is that package's position, and two agents share a group exactly when their
targets coincide (within ``tol``). Groups are numbered by their lowest-index
member, so slots are stable while membership is.

Layout assumption (``x``, 7 columns): position(0:2), velocity(2:4),
offset(4:6), distance(6). **Sign convention differs between sources**:
``dc.envs.transport`` stores ``package - agent`` (``offset_sign=+1``, the
default); ``dc.envs.fake`` stores ``agent - package`` (``offset_sign=-1``).

``radius``: if set, agents farther than this from their nearest package go to
the background column 0 instead of that package's group. ``None`` (default)
assigns every agent to a package.
"""

from __future__ import annotations

from typing import Any

from dc.contracts import ContractError
from dc.membership._xp import arange, f32, i64, lib, onehot

__all__ = ["NearestPackageProvider"]


class NearestPackageProvider:
    def __init__(self, offset_sign: int = 1, radius: float | None = None, tol: float = 1e-4,
                 offset_cols: tuple[int, int] = (4, 6), dist_col: int = 6) -> None:
        if offset_sign not in (1, -1):
            raise ContractError(f"offset_sign must be +1 or -1, got {offset_sign}")
        self.offset_sign, self.radius, self.tol = offset_sign, radius, float(tol)
        self.offset_cols, self.dist_col = offset_cols, dist_col

    def forward(self, *, p: Any, x: Any, h: Any = None, s: Any = None,
                gt: Any = None, k_slots: int, mask: Any = None) -> Any:
        if x is None:
            raise ContractError("NearestPackageProvider needs agent features x")
        L = lib(p)
        n = int(p.shape[2])
        a, b = self.offset_cols
        target = p + self.offset_sign * x[..., a:b]  # (B, T, N, 2): nearest package position

        d = target[..., :, None, :] - target[..., None, :, :]
        same = ((d**2).sum(-1) ** 0.5) < self.tol  # (B, T, N, N); includes the diagonal
        first = f32(same).argmax(-1)  # lowest-index member of each agent's group
        idx = arange(n, like=first)
        rank = L.cumsum(f32(first == idx), -1) - 1.0  # group number, by lowest-index member
        slot = (f32(first[..., None] == idx) * rank[..., None, :]).sum(-1) + 1.0
        slot = slot * f32(slot <= k_slots) + float(k_slots) * f32(slot > k_slots)  # cap at K

        if self.radius is not None:
            slot = slot * (1.0 - f32(x[..., self.dist_col] > self.radius))  # far -> background
        return onehot(i64(slot), k_slots + 1)
