# lipid_analysis/filters.py
from typing import Mapping, Sequence, Tuple

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage as ndi

from .constants import EAST_SHADOWS_KERNEL

RGBTuple = Tuple[int, int, int]


def apply_east_shadows_filter(image: np.ndarray) -> np.ndarray:
    """Apply the 3×3 'East shadows' correlation to a 2D image."""
    if image.ndim != 2:
        raise ValueError(
            f"apply_east_shadows_filter expects 2D, got shape {image.shape}"
        )
    img_float = image.astype(np.float32, copy=False)
    return ndi.correlate(img_float, EAST_SHADOWS_KERNEL, mode="reflect")


def create_custom_colormap(
    start_color: RGBTuple, end_color: RGBTuple
) -> LinearSegmentedColormap:
    """Return LinearSegmentedColormap from start_color → end_color (RGB 0–255)."""
    colors = [
        tuple(c / 255.0 for c in start_color),
        tuple(c / 255.0 for c in end_color),
    ]
    return LinearSegmentedColormap.from_list("custom_colormap", colors)


def get_marker_color(
    marker_name: str, config_colors: Mapping[str, Sequence[int]]
) -> LinearSegmentedColormap:
    """Return a black→marker colormap; falls back to DEFAULT or white if absent."""
    end_color = config_colors.get(
        marker_name, config_colors.get("DEFAULT", (255, 255, 255))
    )
    return create_custom_colormap((0, 0, 0), tuple(end_color))  # type: ignore[arg-type]
