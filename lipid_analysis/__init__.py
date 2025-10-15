# lipid_analysis/__init__.py
"""
lipid_analysis
==============

End-to-end workflow for analyzing lipid inclusions in microscopy images from:
1) Fluorescence ND2 files
2) CARS ND2 files
3) Optional hyperspectral series folders

Typical pipeline steps
----------------------
1. Reference image generation (from a CARS reference ND2).
2. File pairing (fluorescence ↔ CARS) using filename tokens and optional offsets.
3. Image processing & segmentation (cells, lipid droplets, myelin).
4. Hyperspectral processing (corrected series, per-droplet spectra, optional peak fits).
5. Outputs (Excel workbooks, debug figures, overlays).

Usage
-----
    python -m lipid_analysis --config path/to/config.py

Dependencies
------------
- numpy, scipy, pandas, scikit-image, tifffile, matplotlib
- nd2reader, opencv-python, pillow
- lmfit (optional, required for peak fitting)
"""

from .analysis import analyze_3way_intracellular_objects, process_nd2_pair
from .config_utils import load_config, resolve_marker_name
from .constants import CARS_CH, PEAKFIT_DEBUG, VERBOSE
from .filepairing import find_nd2_files, get_file_key, match_fluoro_and_cars, parse_nd2_filename
from .hyperspec import process_hyperspectral_series, visualize_hyperspectral_mask_overlay
from .io_utils import ensure_subdirectory, save_composite_images, save_results_to_excel
from .reference import generate_reference_image

__all__ = [
    # constants (mutability via `import lipid_analysis.constants as const` is recommended)
    "VERBOSE",
    "PEAKFIT_DEBUG",
    "CARS_CH",
    # config / utilities
    "load_config",
    "resolve_marker_name",
    # reference image
    "generate_reference_image",
    # file pairing
    "find_nd2_files",
    "parse_nd2_filename",
    "get_file_key",
    "match_fluoro_and_cars",
    # main analysis
    "process_nd2_pair",
    "analyze_3way_intracellular_objects",
    # hyperspectral
    "process_hyperspectral_series",
    "visualize_hyperspectral_mask_overlay",
    # I/O helpers
    "save_results_to_excel",
    "ensure_subdirectory",
    "save_composite_images",
]
