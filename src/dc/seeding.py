"""One place for reproducible seeding.

One seed = one full training run from scratch (EXPERIMENT_PLAN.md Section 4).
Call :func:`set_seed` once at the top of every entry point.
"""

from __future__ import annotations

import os
import random

__all__ = ["set_seed"]


def set_seed(seed: int, *, deterministic_torch: bool = False) -> int:
    """Seed python, numpy and torch if they are installed. Returns the seed."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        # Slower, and only worth it when chasing a reproducibility bug.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
