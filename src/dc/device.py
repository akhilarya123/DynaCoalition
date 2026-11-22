"""Device selection and batch movement, in one place.

The two machines differ (CUDA on the GPU laptop, CPU or MPS on the M1), so
every entry point asks here instead of hard-coding a device. One rule from
Architecture.md Section 11 is enforced by construction: move the whole batch
once, then keep everything on that device.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = ["get_device", "to_device"]


def get_device(preference: str = "auto") -> Any:
    """Return a torch device.

    ``"auto"`` picks CUDA, then MPS, then CPU. MPS is opt-in via ``"mps"``
    because torch's MPS coverage is uneven; Akhil's N=12 runs are CPU by
    default (work_division.md Section 4).
    """
    import torch

    pref = preference.lower()
    if pref not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"unknown device preference {preference!r}")

    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if pref == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device: 'cuda' requested but torch.cuda.is_available() is False")
    if pref == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device: 'mps' requested but it is not available")
    return torch.device(pref)


def to_device(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    """Move every tensor in a batch to ``device``, converting numpy on the way."""
    import torch

    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        elif hasattr(value, "__array__"):
            out[key] = torch.as_tensor(value).to(device)
        else:
            out[key] = value
    return out
