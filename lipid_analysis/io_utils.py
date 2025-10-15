"""I/O utilities for saving analysis outputs (tables and composite images)."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Mapping, MutableMapping, Tuple

import cv2
import numpy as np
import pandas as pd

from .constants import LOG_LEVEL
from .imaging import (
    blend_fluorescence_cars,
    composite_fluorescence,
    grayscale_autoscale,
)

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

Results = List[MutableMapping[str, Any]]
Summary = List[MutableMapping[str, Any]]


def save_results_to_excel(
    results: Results,
    summary: Summary,
    output_file: str,
) -> None:
    """
    Save detailed per-object results and per-cell summaries to an Excel workbook.

    Two sheets are created:
      - "Detailed Results": one row per detected object/region
      - "Summary":          one row per cell, with aggregate metrics

    Rows are sorted by (file_name, cell_marker, z_stack) when those columns exist.

    Parameters
    ----------
    results
        List of dictionaries describing per-object features.
    summary
        List of dictionaries describing per-cell aggregates.
    output_file
        Path to the output .xlsx file. Parent directories are created if needed.
    """
    results_df = pd.DataFrame(results)
    summary_df = pd.DataFrame(summary)

    # Stable sort for readability if keys exist
    key_cols = ["file_name", "cell_marker", "z_stack"]
    if not results_df.empty and set(key_cols).issubset(results_df.columns):
        results_df = results_df.sort_values(by=key_cols)
    if not summary_df.empty and set(key_cols).issubset(summary_df.columns):
        summary_df = summary_df.sort_values(by=key_cols)

    out_dir = os.path.dirname(os.path.abspath(output_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Let pandas choose a suitable engine (e.g., openpyxl); users can install it if needed.
    with pd.ExcelWriter(output_file) as writer:  # type: ignore[call-arg]
        results_df.to_excel(writer, sheet_name="Detailed Results", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    logger.info("Saved Excel results to %s", output_file)


def ensure_subdirectory(main_dir: str, sub_name: str = "Images") -> str:
    """
    Ensure that a subdirectory exists within `main_dir` and return its path.

    Parameters
    ----------
    main_dir
        Parent directory.
    sub_name
        Subdirectory name to create/ensure (default: "Images").

    Returns
    -------
    str
        The full path to the ensured subdirectory.
    """
    out_dir = os.path.join(main_dir, sub_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_composite_images(
    fluor_images: Dict[str, "np.ndarray"],
    cars_image: "np.ndarray",
    config_dict: Mapping[str, Any],
    main_dir: str,
    file_stub: str,
    alpha: float = 0.5,
) -> Tuple[str, str, str]:
    """
    Save RGB fluorescence composite, 8-bit CARS, and blended overlays as PNG files.

    Files written:
      - <file_stub>_fluor.png       (RGB fluorescence composite)
      - <file_stub>_cars.png        (8-bit autoscaled CARS grayscale)
      - <file_stub>_fluor_cars.png  (alpha-blended overlay)

    Parameters
    ----------
    fluor_images
        Mapping {marker_name: 2D array} used to build the composite.
    cars_image
        2D numeric array for the CARS image (will be autoscaled to uint8).
    config_dict
        Configuration dict containing `colormaps` for composite coloring.
    main_dir
        Base directory where the "Images" subfolder will be created.
    file_stub
        Filename stem used when writing the PNGs (without extension).
    alpha
        Blending factor for fluorescence in the overlay (default 0.5).

    Returns
    -------
    (fluor_path, cars_path, blend_path) : Tuple[str, str, str]
        The full paths of the three written PNG files.

    Raises
    ------
    ValueError
        If writing any of the images fails.
    """
    out_dir = ensure_subdirectory(main_dir, "Images")

    # Build images
    composite_fluor = composite_fluorescence(fluor_images, config_dict)
    cars_gray_8bit = grayscale_autoscale(cars_image)
    fluor_cars_blended = blend_fluorescence_cars(composite_fluor, cars_gray_8bit, alpha=alpha)

    # OpenCV expects BGR for color images
    fluor_bgr = cv2.cvtColor(composite_fluor, cv2.COLOR_RGB2BGR)
    blend_bgr = cv2.cvtColor(fluor_cars_blended, cv2.COLOR_RGB2BGR)

    fluor_path = os.path.join(out_dir, f"{file_stub}_fluor.png")
    cars_path = os.path.join(out_dir, f"{file_stub}_cars.png")
    blend_path = os.path.join(out_dir, f"{file_stub}_fluor_cars.png")

    ok1 = cv2.imwrite(fluor_path, fluor_bgr)
    ok2 = cv2.imwrite(cars_path, cars_gray_8bit)
    ok3 = cv2.imwrite(blend_path, blend_bgr)

    if not (ok1 and ok2 and ok3):
        failed = [p for ok, p in [(ok1, fluor_path), (ok2, cars_path), (ok3, blend_path)] if not ok]
        raise ValueError(f"Failed to write image(s): {', '.join(failed)}")

    logger.info("Saved composites to %s", out_dir)
    return fluor_path, cars_path, blend_path
