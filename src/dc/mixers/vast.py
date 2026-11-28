"""VAST-spatial baseline: hard spatial sub-teams, QMIX over the sub-team values.

Prior-work grouping baseline (VAST, NeurIPS 2021) on VMAS. Per step:

1. Each agent is assigned to exactly one of K sub-teams by *map region*
   (``assignment="grid"``, default) or by k-means (``"kmeans"``).
2. A sub-team's value is the plain **sum** of its members' chosen utilities
   (no per-agent weights ``w_i``, unlike DynaCoalition).
3. A QMIX mixer over those K values, conditioned on the global state, gives
   ``Q_tot``.

How this differs from our CoalitionMix (EXPERIMENT_PLAN.md Section 2): hard
sums instead of soft weighted pooling, QMIX's per-slot hypernetwork instead of
one shared ``g`` + background head, and no memberships trained by TD. It is the
"VAST's operator" side of the comparison; ``PositionClusterProvider`` +
CoalitionMix is the "our operator, same geometry" side.

Why ``grid`` is the default: QMIX's hypernetwork weights are *per input
position*, so the meaning of slot ``c`` must be stable over time. A fixed grid
cell is; a k-means cluster index is an arbitrary permutation that can change
every step. (EXPERIMENT_PLAN.md Section 10 lists "grid vs k-means" as open.)

IGM still holds: sub-team values are non-negative-weighted (all weights 1) sums
of utilities and QMIX is monotone in each input, so Q_tot is monotone in every
agent's utility.

``mixer`` is injected (any ``nn.Module`` mapping sub-team values ``(B, T, K)``
and state ``(B, T, state_dim)`` to ``(B, T, 1)``), so this file does not edit
or depend on the exact shape of ``dc.mixers.qmix``. :meth:`VASTSpatial.build`
constructs it from a dotted path; the constructor keyword names it passes
(``n_agents``, ``state_dim``) are an ASSUMPTION about ``qmix.py`` to confirm.

``forward`` mirrors ``coalition_mix`` (Architecture.md Section 8) so the
training loop can call either: ``(q_sel, h, x, s, p, *, detach, temperature)``
returning ``(Q_tot, S, aux)``. ``h``, ``x``, ``detach`` and ``temperature`` are
accepted and ignored. ``S`` is ``(B, T, N, K+1)`` one-hot with column 0 empty,
so the evaluator can read assignments from it like from any other method.
"""

from __future__ import annotations

from typing import Any

from dc.contracts import ContractError, EnvSpec, resolve
from dc.membership.position import GridProvider, PositionClusterProvider

try:  # keep the pure helpers importable (and testable) without torch
    import torch.nn as _nn

    _Base = _nn.Module
except ImportError:  # pragma: no cover - exercised only where torch is absent
    _nn = None
    _Base = object

__all__ = ["subteam_values", "VASTSpatial", "build_vast_spatial"]


def subteam_values(S: Any, q_sel: Any) -> Any:
    """Sum member utilities per sub-team.

    ``S`` is ``(B, T, N, K+1)`` with column 0 unused; ``q_sel`` is ``(B, T, N)``.
    Returns ``(B, T, K)``: ``Q_k = sum_i S[i, k] * q_i`` for k = 1..K.
    """
    return (S[..., 1:] * q_sel[..., None]).sum(2)


class VASTSpatial(_Base):  # type: ignore[misc]
    def __init__(self, spec: EnvSpec, mixer: Any, *, assignment: str = "grid",
                 bounds: float = 1.0, iters: int = 10) -> None:
        if _nn is None:
            raise ImportError("VASTSpatial needs torch to be installed")
        super().__init__()
        if assignment == "grid":
            self.provider: Any = GridProvider(bounds=bounds)
        elif assignment == "kmeans":
            self.provider = PositionClusterProvider(mode="kmeans", iters=iters)
        else:
            raise ContractError(f"assignment must be 'grid' or 'kmeans', got {assignment!r}")
        self.spec = spec
        self.mixer = mixer

    @classmethod
    def build(cls, spec: EnvSpec, base: str = "dc.mixers.qmix:QMixer",
              base_kwargs: dict[str, Any] | None = None, **kwargs: Any) -> "VASTSpatial":
        """Resolve ``base`` (a dotted path) and build QMIX over K inputs.

        ASSUMES the base mixer's constructor accepts ``n_agents`` and
        ``state_dim`` keywords; confirm against the real ``qmix.py``.
        """
        mixer = resolve(base)(n_agents=spec.k_slots, state_dim=spec.state_dim, **(base_kwargs or {}))
        return cls(spec, mixer, **kwargs)

    def forward(self, q_sel: Any, h: Any, x: Any, s: Any, p: Any, *,
                detach: bool = False, temperature: float = 1.0) -> tuple[Any, Any, dict[str, Any]]:
        S = self.provider.forward(p=p, k_slots=self.spec.k_slots)
        values = subteam_values(S, q_sel)  # (B, T, K)
        q_tot = self.mixer(values, s)  # (B, T, 1)
        return q_tot, S, {"W": None, "m": S[..., 1:].sum(2), "z": None}


def build_vast_spatial(spec: EnvSpec, **kwargs: Any) -> VASTSpatial:
    """Factory for configs: ``target: dc.mixers.vast:build_vast_spatial`` (see ``VASTSpatial.build``)."""
    return VASTSpatial.build(spec, **kwargs)
