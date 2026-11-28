"""Oracle (contact) control: memberships straight from ground-truth labels.

``S`` is the one-hot of ``gt_coalitions``: label ``k`` (1..M) goes to slot ``k``,
and label 0 ("in contact with nothing") goes to the background column 0. Since
K = N/2 > M = N/3 every package has its own slot (enforced by ``EnvSpec``).

This is the reference ceiling in the comparison ladder: "does *correct*
grouping help at all?". It is the only provider allowed to read ``gt``.

Alignment: ``gt`` must cover the same steps as ``p``. The batch contract gives
``gt_coalitions`` T steps but observation-like keys T+1, so the *target-side*
call (states 1..T) has no label for the last step. See akhil.md, "Open
question 8"; until the contract carries T+1 labels the caller has to decide
what to do there.
"""

from __future__ import annotations

from typing import Any

from dc.contracts import ContractError
from dc.membership._xp import onehot

__all__ = ["OracleProvider"]


class OracleProvider:
    """One-hot memberships from ground-truth contact labels (parameter-free)."""

    def __init__(self, check: bool = True) -> None:
        # ``check`` costs a device sync per call; turn it off in the hot loop.
        self.check = check

    def forward(self, *, p: Any, x: Any = None, h: Any = None, s: Any = None,
                gt: Any, k_slots: int, mask: Any = None) -> Any:
        if gt is None:
            raise ContractError("OracleProvider needs gt_coalitions; got None")
        if tuple(gt.shape) != tuple(p.shape[:3]):
            raise ContractError(f"gt shape {tuple(gt.shape)} must match p[:3] {tuple(p.shape[:3])}")
        if self.check:
            lo, hi = int(gt.min()), int(gt.max())
            if lo < 0 or hi > k_slots:
                raise ContractError(f"gt labels must lie in [0, K={k_slots}], got [{lo}, {hi}]")
        return onehot(gt, k_slots + 1)
