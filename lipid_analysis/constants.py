"""Constants and global configuration for the lipid analysis pipeline."""

from __future__ import annotations

import logging
from typing import Final

import numpy as np

# -----------------------------------------------------------------------------
# Runtime flags (these are defaults; your CLI or entry-point can override them)
# -----------------------------------------------------------------------------
VERBOSE: bool = True
PEAKFIT_DEBUG: bool = True  # set True to display per-droplet fit plots

# Derived log level aligned to VERBOSE (entry-point should still configure handlers)
LOG_LEVEL: Final[int] = logging.DEBUG if VERBOSE else logging.WARNING

# Suppress excessive logs from nd2reader regardless of VERBOSE
logging.getLogger("nd2reader").setLevel(logging.ERROR)

# -----------------------------------------------------------------------------
# Imaging / filtering constants
# -----------------------------------------------------------------------------
# Imaging channel constant (CARS channel index in ND2 files)
CARS_CH: Final[int] = 2

# 3×3 kernel for the "east-shadows" directional correlation filter
EAST_SHADOWS_KERNEL: Final[np.ndarray] = np.array(
    [
        [-1, 0, 1],
        [-2, 1, 2],
        [-1, 0, 1],
    ],
    dtype=np.float32,
)
