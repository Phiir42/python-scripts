# lipid_analysis/constants.py
"""
Global constants and default runtime flags.

Notes:
- VERBOSE and PEAKFIT_DEBUG are defaults; they may be overridden by CLI
  or other runtime configuration. To avoid stale values, prefer:

    import lipid_analysis.constants as const
    if const.VERBOSE:
        ...

instead of `from lipid_analysis.constants import VERBOSE`.
"""

import logging

import numpy as np

# Suppress excessive logs from nd2reader
logging.getLogger("nd2reader").setLevel(logging.ERROR)

# Kernel for EAST-shadows filtering
EAST_SHADOWS_KERNEL = np.array(
    [
        [-1, 0, 1],
        [-2, 1, 2],
        [-1, 0, 1],
    ],
    dtype=np.float32,
)

# Imaging channel constants
CARS_CH = 2

# Runtime flags (overridden by CLI)
VERBOSE = True
PEAKFIT_DEBUG = True  # set True to display per-droplet fit plots
