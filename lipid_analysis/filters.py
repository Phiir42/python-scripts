"""Filtering and colormap utilities for lipid analysis."""

from __future__ import annotations

import logging
from typing import Mapping, Sequence, Tuple

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage as ndi

from .constants import EAST_SHADOWS_KERNEL, LOG_LEVEL

# Module logger aligned to VERBOSE via LOG_LEVEL
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

RGBTuple = Tuple[int, int, int]


def apply_east_shadows_filter(image: np.ndarray) -> np.ndarray:
    """
    Apply a 3×3 directional "east-shadows" correlation to a 2D image.

    Emphasizes edges/structures with a directional response using reflect padding.

    Parameters
    ----------
    image
        2D input image (H×W). Processed as float32.

    Returns
    -------
    np.ndarray
        2D float32 array, same shape as input.
    """
    if image.ndim != 2:
        raise ValueError(
            f"apply_east_shadows_filter expects a 2D array, got shape {image.shape}"
        )
    img_float = image.astype(np.float32, copy=False)
    logger.debug("Applying east-shadows correlation to image of shape %s", image.shape)
    return ndi.correlate(img_float, EAST_SHADOWS_KERNEL, mode="reflect")


def _validate_rgb(rgb: Sequence[int]) -> RGBTuple:
    """Validate and coerce a sequence to an (R, G, B) tuple with values in [0, 255]."""
    if len(rgb) != 3:
        raise ValueError(f"Expected RGB of length 3, got {rgb!r}")
    r, g, b = (int(x) for x in rgb)
    r = min(255, max(0, r))
    g = min(255, max(0, g))
    b = min(255, max(0, b))
    return r, g, b


def create_custom_colormap(
    start_color: RGBTuple, end_color: RGBTuple
) -> LinearSegmentedColormap:
    """
    Create a 2-stop LinearSegmentedColormap from RGB endpoints.

    Parameters
    ----------
    start_color, end_color
        (R, G, B) tuples with 0–255 components.

    Returns
    -------
    LinearSegmentedColormap
        A colormap that linearly interpolates start → end.
    """
    colors = [tuple(c / 255.0 for c in start_color), tuple(c / 255.0 for c in end_color)]
    return LinearSegmentedColormap.from_list("custom_colormap", colors)


def get_marker_color(
    marker_name: str, config_colors: Mapping[str, Sequence[int]]
) -> LinearSegmentedColormap:
    """
    Produce a black→marker colormap for a named marker.

    Parameters
    ----------
    marker_name
        Name to look up in `config_colors`.
    config_colors
        Mapping like {"MARKER": [R, G, B], "DEFAULT": [R, G, B], ...} with 0–255 values.

    Returns
    -------
    LinearSegmentedColormap
        Colormap from black to the marker color. Falls back to "DEFAULT" if present,
        otherwise white.
    """
    end_seq = config_colors.get(marker_name, config_colors.get("DEFAULT", (255, 255, 255)))
    end_rgb = _validate_rgb(end_seq)
    logger.debug("Creating marker colormap for %s with RGB %s", marker_name, end_rgb)
    return create_custom_colormap((0, 0, 0), end_rgb)
