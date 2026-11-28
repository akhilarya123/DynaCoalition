"""Tiny numpy/torch shim so providers run on either without a torch import.

Providers are training-time objects (torch tensors on the training device), but
their *math* is tested in CI on numpy. This module hides the handful of calls
whose spelling differs; everything else in the providers uses operators and
methods that both libraries share. Private to ``dc.membership``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["lib", "is_torch", "arange", "f32", "i64", "zeros", "ones", "cat", "onehot"]


def is_torch(x: Any) -> bool:
    return type(x).__module__.split(".")[0] == "torch"


def lib(x: Any):
    """The array library ``x`` belongs to (numpy or torch)."""
    if is_torch(x):
        import torch

        return torch
    import numpy

    return numpy


def arange(n: int, like: Any):
    return lib(like).arange(n, device=like.device) if is_torch(like) else lib(like).arange(n)


def f32(x: Any):
    return x.to(lib(x).float32) if is_torch(x) else x.astype("float32")


def i64(x: Any):
    return x.long() if is_torch(x) else x.astype("int64")


def zeros(shape: tuple[int, ...], like: Any):
    """float32 zeros on the same device as ``like``."""
    if is_torch(like):
        return lib(like).zeros(shape, dtype=lib(like).float32, device=like.device)
    return lib(like).zeros(shape, dtype="float32")


def ones(shape: tuple[int, ...], like: Any):
    return zeros(shape, like) + 1.0


def cat(xs: list[Any], axis: int):
    L = lib(xs[0])
    return L.cat(xs, axis) if is_torch(xs[0]) else L.concatenate(xs, axis)


def onehot(idx: Any, n: int):
    """``(..., N)`` int labels -> ``(..., N, n)`` float32 one-hot."""
    return f32(idx[..., None] == arange(n, like=idx))
