"""The frozen contract between the two halves of the codebase.

This is the ONLY module both owners depend on (see work_division.md, Set 0).
Changing it is a conversation, not a commit.

It deliberately imports nothing outside the standard library. Validation is
duck-typed over anything with ``.shape`` and ``.dtype``, so it works on torch
tensors and numpy arrays alike, and the contract tests run in CI without
installing torch.

Time convention
---------------
``Architecture.md`` Section 7 lists every batch key with a leading ``(B, T)``,
but the Double-Q snippet in the same section slices ``q_online[:, 1:]``, which
needs one extra step to bootstrap from. We resolve that here:

* **Observation-like keys** (``obs``, ``state``, ``positions``,
  ``agent_features``, ``avail_actions``) have length **T + 1**.
* **Transition keys** (``actions``, ``rewards``, ``dones``, ``mask``,
  ``gt_coalitions``) have length **T**.

So a batch holds T transitions and the T+1 states they run between.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

__all__ = [
    "ContractError",
    "EnvSpec",
    "k_slots_for",
    "OBS_KEYS",
    "TRANSITION_KEYS",
    "BATCH_KEYS",
    "validate_batch",
    "validate_memberships",
    "MembershipProvider",
    "METRICS_COLUMNS",
    "blank_metrics_row",
    "EvalRecord",
    "resolve",
]


class ContractError(ValueError):
    """Raised when data does not match the agreed contract."""


# --------------------------------------------------------------------------
# Environment spec
# --------------------------------------------------------------------------


def k_slots_for(n_agents: int) -> int:
    """K = N/2, the agreed rule (work_division.md Section 8).

    K is an independent hyperparameter and must never equal M = N/3, which
    would leak task knowledge: at N=12, K=6 vs M=4; at N=24, K=12 vs M=8.
    """
    if n_agents <= 0:
        raise ContractError(f"n_agents must be positive, got {n_agents}")
    if n_agents % 2:
        raise ContractError(f"K = N/2 needs an even N, got {n_agents}")
    return n_agents // 2


@dataclass(frozen=True)
class EnvSpec:
    """Shapes every module agrees on. Build one, pass it everywhere."""

    n_agents: int
    n_packages: int
    obs_dim: int
    state_dim: int
    x_dim: int
    n_actions: int = 5  # stay, up, down, left, right
    episode_limit: int = 200
    k_slots_override: int | None = None

    def __post_init__(self) -> None:
        for name in ("n_packages", "obs_dim", "state_dim", "x_dim", "n_actions", "episode_limit"):
            if getattr(self, name) <= 0:
                raise ContractError(f"{name} must be positive, got {getattr(self, name)}")
        k = self.k_slots
        if k <= self.n_packages:
            raise ContractError(
                f"K must exceed M so slots are not forced to share packages: "
                f"K={k}, M={self.n_packages}"
            )

    @property
    def k_slots(self) -> int:
        """Number of coalition slots, excluding background."""
        if self.k_slots_override is not None:
            return self.k_slots_override
        return k_slots_for(self.n_agents)

    @property
    def n_slots(self) -> int:
        """Columns of S: K coalition slots plus the background column 0."""
        return self.k_slots + 1


# --------------------------------------------------------------------------
# Batch contract
# --------------------------------------------------------------------------

# key -> (symbolic shape, allowed dtype kinds)
_FIELDS: dict[str, tuple[tuple[Any, ...], tuple[str, ...]]] = {
    # observation-like: length T + 1
    "obs": (("B", "T1", "N", "obs_dim"), ("float",)),
    "state": (("B", "T1", "state_dim"), ("float",)),
    "positions": (("B", "T1", "N", 2), ("float",)),
    "agent_features": (("B", "T1", "N", "x_dim"), ("float",)),
    "avail_actions": (("B", "T1", "N", "n_actions"), ("float", "int", "bool")),
    # transition-like: length T
    "actions": (("B", "T", "N", 1), ("int",)),
    "gt_coalitions": (("B", "T", "N"), ("int",)),
    "rewards": (("B", "T", 1), ("float",)),
    "dones": (("B", "T", 1), ("float", "bool")),
    "mask": (("B", "T", 1), ("float", "bool")),
}

OBS_KEYS = tuple(k for k, (s, _) in _FIELDS.items() if s[1] == "T1")
TRANSITION_KEYS = tuple(k for k, (s, _) in _FIELDS.items() if s[1] == "T")
BATCH_KEYS = tuple(_FIELDS)


def _shape(arr: Any, key: str) -> tuple[int, ...]:
    try:
        return tuple(int(d) for d in arr.shape)
    except AttributeError as exc:  # not an array at all
        raise ContractError(f"batch['{key}'] has no .shape ({type(arr).__name__})") from exc


def _dtype_kind(arr: Any) -> str:
    text = str(getattr(arr, "dtype", "")).lower()
    if "bool" in text:
        return "bool"
    if "float" in text or "double" in text or "half" in text or "bfloat" in text:
        return "float"
    if "int" in text or "long" in text or "short" in text or "byte" in text:
        return "int"
    return "other"


def validate_batch(
    batch: Mapping[str, Any],
    spec: EnvSpec,
    *,
    allow_extra_keys: bool = False,
) -> tuple[int, int]:
    """Check a sampled batch against the contract.

    Returns ``(batch_size, horizon)``. Raises :class:`ContractError` with a
    message naming the offending key on any mismatch.
    """
    missing = [k for k in BATCH_KEYS if k not in batch]
    if missing:
        raise ContractError(f"batch is missing keys: {sorted(missing)}")
    if not allow_extra_keys:
        extra = [k for k in batch if k not in _FIELDS]
        if extra:
            raise ContractError(f"batch has unexpected keys: {sorted(extra)}")

    b, t = _shape(batch["mask"], "mask")[:2]
    dims = {
        "B": b,
        "T": t,
        "T1": t + 1,
        "N": spec.n_agents,
        "obs_dim": spec.obs_dim,
        "state_dim": spec.state_dim,
        "x_dim": spec.x_dim,
        "n_actions": spec.n_actions,
    }

    for key, (symbolic, kinds) in _FIELDS.items():
        arr = batch[key]
        want = tuple(dims[d] if isinstance(d, str) else d for d in symbolic)
        got = _shape(arr, key)
        if got != want:
            raise ContractError(
                f"batch['{key}'] has shape {got}, expected {want} "
                f"({'×'.join(str(d) for d in symbolic)})"
            )
        kind = _dtype_kind(arr)
        if kind not in kinds:
            raise ContractError(
                f"batch['{key}'] has dtype kind '{kind}', expected one of {list(kinds)}"
            )

    gt = batch["gt_coalitions"]
    lo, hi = int(gt.min()), int(gt.max())
    if lo < 0 or hi > spec.n_packages:
        raise ContractError(
            f"gt_coalitions must lie in [0, M={spec.n_packages}] "
            f"(0 = no contact), got [{lo}, {hi}]"
        )

    acts = batch["actions"]
    lo, hi = int(acts.min()), int(acts.max())
    if lo < 0 or hi >= spec.n_actions:
        raise ContractError(f"actions must lie in [0, {spec.n_actions - 1}], got [{lo}, {hi}]")

    return b, t


def validate_memberships(
    S: Any,
    spec: EnvSpec,
    *,
    batch_size: int,
    horizon: int,
    tol: float = 1e-4,
) -> None:
    """Check an S returned by any :class:`MembershipProvider`.

    Rows must be non-negative and sum to 1; column 0 is background.
    """
    want = (batch_size, horizon, spec.n_agents, spec.n_slots)
    got = _shape(S, "S")
    if got != want:
        raise ContractError(f"S has shape {got}, expected {want} = (B, T, N, K+1)")
    if _dtype_kind(S) != "float":
        raise ContractError(f"S must be floating point, got dtype {getattr(S, 'dtype', '?')}")
    if float(S.min()) < -tol:
        raise ContractError(f"S has negative entries (min {float(S.min()):.3e})")
    drift = float(abs(S.sum(-1) - 1.0).max())
    if drift > tol:
        raise ContractError(f"S rows must sum to 1; worst row is off by {drift:.3e}")


@runtime_checkable
class MembershipProvider(Protocol):
    """Everything that produces coalition memberships, learned or not.

    Returns S of shape ``(B, T, N, K+1)``; rows sum to 1; column 0 is the
    background slot. Stateless controls ignore the arguments they do not
    need. ``gt`` is passed to everyone but may only be read by the oracle.
    """

    def forward(self, *, p, x, h, s, gt, k_slots, mask): ...


# --------------------------------------------------------------------------
# Results contract
# --------------------------------------------------------------------------

METRICS_COLUMNS: tuple[str, ...] = (
    "run_id",
    "algo",
    "n_agents",
    "seed",
    "env_step",
    "wall_clock_s",
    "return_mean",
    "return_std",
    "packages_delivered",
    "ari_contact",
    "ari_all",
    "switch_rate",
    "grad_gnn_td",
    "mean_max_membership",
    "k_eff",
)


def blank_metrics_row(**values: Any) -> dict[str, Any]:
    """A metrics.csv row with every agreed column present.

    Unset columns are ``None`` so a method that cannot report a diagnostic
    (QMIX has no memberships) still writes a well-formed row.
    """
    unknown = set(values) - set(METRICS_COLUMNS)
    if unknown:
        raise ContractError(f"not metrics.csv columns: {sorted(unknown)}")
    row: dict[str, Any] = {c: None for c in METRICS_COLUMNS}
    row.update(values)
    return row


@dataclass
class EvalRecord:
    """One evaluation episode, dumped to disk for offline metrics.

    The evaluator reads these files; it never touches the trainer's objects.
    ``assignments``, ``gt_coalitions`` and ``in_contact`` are ``(T, N)``.
    """

    run_id: str
    algo: str
    n_agents: int
    seed: int
    env_step: int
    episode_idx: int
    episode_return: float
    packages_delivered: int
    assignments: Any
    gt_coalitions: Any
    in_contact: Any
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self, spec: EnvSpec | None = None) -> tuple[int, int]:
        """Check the three arrays agree on ``(T, N)``. Returns that pair."""
        shapes = {
            name: _shape(getattr(self, name), name)
            for name in ("assignments", "gt_coalitions", "in_contact")
        }
        distinct = set(shapes.values())
        if len(distinct) != 1:
            raise ContractError(f"EvalRecord arrays disagree on shape: {shapes}")
        t, n = distinct.pop()
        if n != self.n_agents:
            raise ContractError(f"EvalRecord arrays have N={n}, but n_agents={self.n_agents}")
        if spec is not None:
            if n != spec.n_agents:
                raise ContractError(f"EvalRecord N={n} does not match spec N={spec.n_agents}")
            hi = int(self.assignments.max())
            if hi > spec.k_slots:
                raise ContractError(
                    f"assignments must be slot indices in [0, K={spec.k_slots}], got max {hi}"
                )
        return t, n


# --------------------------------------------------------------------------
# Config convention
# --------------------------------------------------------------------------


def resolve(target: str) -> Any:
    """Import ``"package.module:Attribute"`` and return the attribute.

    This is what keeps a central ``--algo`` dispatch out of the codebase:
    configs name a dotted path, so adding a control is a new file plus a new
    yaml, with no edit to anything the other owner owns.
    """
    if ":" not in target:
        raise ContractError(f"target must look like 'pkg.module:Attr', got {target!r}")
    module_path, _, attr = target.partition(":")
    from importlib import import_module

    module = import_module(module_path)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ContractError(f"{module_path} has no attribute {attr!r}") from exc
