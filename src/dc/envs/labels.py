"""Ground-truth contact labels and delivery geometry (pure functions).

Everything here is stateless and VMAS-free, so the definition of
``gt_coalitions`` (the thing ARI is scored against) can be unit-tested on
hand-built cases. Functions accept numpy arrays or torch tensors and return
the same kind; only ``cos``/``sin`` need the backend, and it is detected from
the input. Leading batch dimensions are arbitrary (``...`` below), so the same
code serves ``(E, N, 2)`` during rollout and ``(B, T, N, 2)`` offline.

Definition of contact (EXPERIMENT_PLAN.md Section 1)
----------------------------------------------------
Agent *i* is in contact with package *k* when the gap between the agent's
circle and the package's (rotated) box is at most ``threshold``::

    gap_ik = signed_distance(agent_i centre, box_k) - agent_radius

``gap`` is negative when the agent overlaps the box (soft collisions let
agents sink in slightly). ``gt_coalitions[i] = k + 1`` for the package with the
smallest gap if that gap is within the threshold, else ``0``. Ties go to the
lowest package index. ``k + 1`` because 0 is "no contact".
"""

from __future__ import annotations

from typing import Any

__all__ = ["box_signed_distance", "contact_gaps", "labels_from_gaps", "contact_labels", "on_goal"]


def _backend(x: Any):
    """numpy or torch, whichever ``x`` belongs to."""
    if type(x).__module__.split(".")[0] == "torch":
        import torch

        return torch
    import numpy

    return numpy


def _relu(a: Any) -> Any:
    return (a + abs(a)) / 2


def box_signed_distance(points: Any, centers: Any, rots: Any, half_extents: tuple[float, float]) -> Any:
    """Signed distance from points to the boundary of rotated boxes.

    Parameters
    ----------
    points:       ``(..., N, 2)`` world positions.
    centers:      ``(..., M, 2)`` box centres.
    rots:         ``(..., M)`` box rotation in radians (VMAS ``state.rot``).
    half_extents: ``(hx, hy)`` = (length / 2, width / 2) in the box frame.

    Returns ``(..., N, M)``: positive outside, negative inside, 0 on the edge.
    """
    lib = _backend(points)
    hx, hy = half_extents
    delta = points[..., :, None, :] - centers[..., None, :, :]  # (..., N, M, 2)
    c = lib.cos(rots)[..., None, :]  # (..., 1, M)
    s = lib.sin(rots)[..., None, :]
    dx, dy = delta[..., 0], delta[..., 1]
    local_x = c * dx + s * dy  # rotate by -rot into the box frame
    local_y = -s * dx + c * dy
    qx = abs(local_x) - hx
    qy = abs(local_y) - hy
    outside = (_relu(qx) ** 2 + _relu(qy) ** 2) ** 0.5
    m = (qx + qy + abs(qx - qy)) / 2  # max(qx, qy)
    inside = (m - abs(m)) / 2  # min(max(qx, qy), 0)
    return outside + inside


def contact_gaps(
    agent_pos: Any,
    pkg_pos: Any,
    pkg_rot: Any,
    *,
    half_extents: tuple[float, float],
    agent_radius: float,
) -> Any:
    """``(..., N, M)`` gap between each agent circle and each package box."""
    return box_signed_distance(agent_pos, pkg_pos, pkg_rot, half_extents) - agent_radius


def labels_from_gaps(gaps: Any, threshold: float) -> Any:
    """Turn ``(..., N, M)`` gaps into ``(..., N)`` int64 labels in ``[0, M]``."""
    near = (gaps <= threshold).sum(-1) > 0  # any package within the threshold
    nearest = gaps.argmin(-1)  # first minimal index on ties
    return (nearest + 1) * near  # int64; 0 where nothing is in contact


def contact_labels(
    agent_pos: Any,
    pkg_pos: Any,
    pkg_rot: Any,
    *,
    half_extents: tuple[float, float],
    agent_radius: float,
    threshold: float,
) -> Any:
    """``gt_coalitions`` for one step (or any leading batch shape)."""
    gaps = contact_gaps(
        agent_pos, pkg_pos, pkg_rot, half_extents=half_extents, agent_radius=agent_radius
    )
    return labels_from_gaps(gaps, threshold)


def on_goal(pkg_pos: Any, goal_pos: Any, goal_radius: float) -> Any:
    """``(..., M)`` bool: package *centre* inside the goal circle.

    A centre test (not VMAS's box-vs-circle overlap) keeps "delivered"
    deterministic and testable; it is slightly stricter than overlap.
    """
    d = pkg_pos - goal_pos[..., None, :]
    return ((d**2).sum(-1) ** 0.5) <= goal_radius
