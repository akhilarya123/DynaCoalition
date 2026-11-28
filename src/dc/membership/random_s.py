"""Random-S control: Dirichlet memberships, no relation to the task.

Answers "does any grouping structure matter?". Rows are drawn from a symmetric
Dirichlet(alpha) over all K+1 columns, so the background column gets about
1/(K+1) of the mass on average, like every other slot.

``alpha`` sets sharpness: 1.0 is uniform over the simplex; smaller is peakier.
The default stays at 1.0 because float32 Dirichlet samples with very small
alpha can underflow.

``mode``:
  * ``"iid"``     a fresh draw for every (episode, step, agent): pure noise,
                  flickers every step.
  * ``"episode"`` one draw per (episode, agent), repeated over time: random but
                  *static* groups (closest to VAST's "random" sub-teams).

Reproducibility: the numpy path uses the provider's own seeded generator; the
torch path uses torch's global RNG, so seed it with ``dc.seeding.set_seed``.
"""

from __future__ import annotations

from typing import Any

from dc.contracts import ContractError
from dc.membership._xp import is_torch, ones

__all__ = ["RandomProvider"]


class RandomProvider:
    def __init__(self, alpha: float = 1.0, mode: str = "iid", seed: int = 0) -> None:
        if alpha <= 0:
            raise ContractError(f"alpha must be positive, got {alpha}")
        if mode not in ("iid", "episode"):
            raise ContractError(f"mode must be 'iid' or 'episode', got {mode!r}")
        self.alpha, self.mode, self.seed = float(alpha), mode, int(seed)
        self._rng = None  # numpy path only, created on first use

    def forward(self, *, p: Any, x: Any = None, h: Any = None, s: Any = None,
                gt: Any = None, k_slots: int, mask: Any = None) -> Any:
        b, t, n = (int(d) for d in p.shape[:3])
        k1 = k_slots + 1
        t_draw = 1 if self.mode == "episode" else t
        if is_torch(p):
            import torch

            conc = torch.full((k1,), self.alpha, device=p.device)
            S = torch.distributions.Dirichlet(conc).sample((b, t_draw, n))
        else:
            import numpy as np

            if self._rng is None:
                self._rng = np.random.default_rng(self.seed)
            S = self._rng.dirichlet(np.full(k1, self.alpha), size=(b, t_draw, n)).astype("float32")
        S = S / S.sum(-1)[..., None]  # exact normalisation after the float32 cast
        if self.mode == "episode":
            S = S * ones((1, t, 1, 1), like=S)
        return S
