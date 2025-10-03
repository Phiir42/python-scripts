# lipid_analysis/__init__.py

"""
lipid_analysis

This package provides a workflow for analyzing lipid inclusions in microscopy images
using:
1) Fluorescence .nd2 files
2) CARS (Coherent Anti-Stokes Raman Scattering) .nd2 files
3) Optional hyperspectral series folders

Main steps:
-----------
1. Reference image generation:
   - A reference .nd2 file is opened and used to create a normalized reference .tif.

2. File pairing and offset logic:
   - The code identifies fluorescence vs. CARS .nd2 files (based on config-defined
     keywords).
   - Each file is given a "StacksX" key, with optional marker-based offsets for
     fluorescence, ensuring correct pairing with corresponding CARS images.

3. Image processing:
   - Fluorescence images are processed to generate a binary cell mask.
   - CARS images are processed to identify lipid droplets (foci).
   - The pipeline performs measurements of lipid inclusions (size, intensity) within
     each cell.

4. Hyperspectral analysis (if applicable):
   - Folders containing hyperspectral data are detected, and each .nd2 file in the folder
     is processed to build a series of corrected images and measure lipid intensities
     across different wavenumbers.

5. Results output:
   - The code saves a final Excel file containing detailed measurements for each
     cell (lipid objects, intensities) and a summary table.

Usage:
------
Run as a module:
    python -m lipid_analysis --config path/to/config.py

Dependencies:
-------------
- nd2reader
- scipy
- numpy
- pandas
- scikit-image (skimage)
- tifffile
- matplotlib
- opencv-python (cv2)
- pillow (PIL)
- lmfit   (for peak fitting; optional but recommended)

Note:
-----
The pipeline assumes .nd2 files follow a naming convention containing "StacksX" and a
magnification keyword like "100X". The config file governs marker offsets and how
hyperspectral data are processed. Adjust the config for other systems or filename patterns.
"""


from .analysis import analyze_3way_intracellular_objects, process_nd2_pair
from .config_utils import load_config, resolve_marker_name
from .constants import CARS_CH, PEAKFIT_DEBUG, VERBOSE
from .filepairing import (
    find_nd2_files,
    get_file_key,
    match_fluoro_and_cars,
    parse_nd2_filename,
)
from .hyperspec import (
    process_hyperspectral_series,
    visualize_hyperspectral_mask_overlay,
)
from .io_utils import ensure_subdirectory, save_composite_images, save_results_to_excel
from .reference import generate_reference_image

__all__ = [
    # constants (see note above about importing from .constants if mutating)
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
