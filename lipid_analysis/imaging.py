"""Imaging utilities: grayscale scaling, overlay blending, and RGB composites."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from skimage.exposure import rescale_intensity
from skimage.filters import gaussian
from .filters import apply_east_shadows_filter

from .constants import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


def grayscale_autoscale(image_2d: np.ndarray) -> np.ndarray:
    """
    Rescale a 2D image to the full 0..255 range and return uint8.

    The input is scaled using skimage's 'image' range semantics, which uses the
    min/max of the input image to linearly map intensities.

    Parameters
    ----------
    image_2d
        2D array (H, W).

    Returns
    -------
    np.ndarray
        2D uint8 array (H, W) scaled to [0, 255].

    Raises
    ------
    ValueError
        If the input is not 2D.
    """
    if image_2d.ndim != 2:
        raise ValueError(f"grayscale_autoscale expects 2D, got {image_2d.shape}")
    scaled = rescale_intensity(image_2d, in_range="image", out_range=(0, 255))
    return scaled.astype(np.uint8, copy=False)


def blend_fluorescence_cars(
    fluor_rgb: np.ndarray,
    cars_gray: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Alpha-blend a color fluorescence image with a grayscale CARS image.

    Both images must have identical height and width. The grayscale CARS image
    is broadcast to RGB before blending.

    Parameters
    ----------
    fluor_rgb
        3-channel color image (H, W, 3), uint8 or float-like.
    cars_gray
        2D grayscale image (H, W), uint8 or float-like.
    alpha
        Blending factor for fluorescence (0..1). CARS gets (1 - alpha).

    Returns
    -------
    np.ndarray
        3-channel uint8 blended image (H, W, 3).

    Raises
    ------
    ValueError
        If shapes are incompatible or dimensions are not as expected.
    """
    if fluor_rgb.ndim != 3 or fluor_rgb.shape[2] != 3:
        raise ValueError(f"fluor_rgb must be H×W×3, got {fluor_rgb.shape}")
    if cars_gray.ndim != 2:
        raise ValueError(f"cars_gray must be 2D, got {cars_gray.shape}")
    if fluor_rgb.shape[:2] != cars_gray.shape:
        raise ValueError("fluor_rgb and cars_gray must have the same H×W")

    cars_rgb = np.repeat(cars_gray[..., None], 3, axis=2)

    fluor_f = fluor_rgb.astype(np.float32, copy=False)
    cars_f = cars_rgb.astype(np.float32, copy=False)

    blend = alpha * fluor_f + (1.0 - alpha) * cars_f
    return np.clip(blend, 0, 255).astype(np.uint8)


def _rgb01(
    rgb: Sequence[float] | NDArray[np.floating] | Sequence[int] | NDArray[np.integer]
) -> np.ndarray:
    """
    Convert an RGB triple to float32 in [0, 1] with validation and clamping.

    Accepts values given either in [0, 255] or already in [0, 1]. If any value
    exceeds 1.0, the function assumes the input is 0..255 and rescales.
    """
    arr = np.asarray(rgb, dtype=np.float32).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"RGB must have length 3, got shape {arr.shape}")
    if np.any(arr > 1.0):
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def colorize_channel(
    image_2d: np.ndarray,
    rgb_color: Sequence[float] | NDArray[np.floating] | Sequence[int] | NDArray[np.integer],
) -> np.ndarray:
    """
    Apply a color to a 2D image and return a 3-channel float image in [0, 1].

    The input is rescaled to [0, 1] using its own min/max (skimage 'image' range),
    then multiplied by the provided RGB color.

    Parameters
    ----------
    image_2d
        2D array (H, W).
    rgb_color
        RGB triple. Values may be in 0..1 or 0..255. Length must be 3.

    Returns
    -------
    np.ndarray
        3-channel float32 image (H, W, 3) with values in [0, 1].

    Raises
    ------
    ValueError
        If the input is not 2D or the RGB triple is invalid.
    """
    if image_2d.ndim != 2:
        raise ValueError(f"colorize_channel expects 2D, got {image_2d.shape}")

    col01 = _rgb01(rgb_color)
    scaled = rescale_intensity(image_2d, in_range="image", out_range=(0.0, 1.0)).astype(
        np.float32, copy=False
    )
    return np.stack((scaled * col01[0], scaled * col01[1], scaled * col01[2]), axis=-1)


def composite_fluorescence(
    fluor_images: Dict[str, np.ndarray],
    config_dict: Mapping[str, Any],
) -> np.ndarray:
    """
    Build an RGB composite by colorizing each marker channel and summing (clipped).

    The color for each marker is taken from `config_dict['colormaps'][marker]`,
    which should be an RGB triple either in 0..255 or 0..1. If a marker color is
    not provided, this falls back to `config_dict['colormaps']['DEFAULT']` if present,
    otherwise white.

    All channels must have identical shapes.

    Parameters
    ----------
    fluor_images
        Mapping of {marker_name: 2D image}.
    config_dict
        Configuration dictionary containing an optional
        `colormaps: Dict[str, Sequence[float]]`.

    Returns
    -------
    np.ndarray
        3-channel uint8 composite image (H, W, 3).

    Raises
    ------
    ValueError
        If `fluor_images` is empty, contains non-2D arrays, or shapes differ.
    """
    if not fluor_images:
        raise ValueError("composite_fluorescence received an empty fluor_images dict")

    # Validate shapes and infer H, W
    iterator = iter(fluor_images.items())
    first_marker, first_img = next(iterator)
    if first_img.ndim != 2:
        raise ValueError(f"Channel for marker '{first_marker}' must be 2D, got {first_img.shape}")
    H, W = first_img.shape

    for marker, img in iterator:
        if img.ndim != 2:
            raise ValueError(f"Channel for marker '{marker}' must be 2D, got {img.shape}")
        if img.shape != (H, W):
            raise ValueError(
                f"All fluorescence channels must have the same shape; "
                f"'{marker}' has {img.shape} vs expected {(H, W)}"
            )

    comp = np.zeros((H, W, 3), dtype=np.float32)

    colormaps = config_dict.get("colormaps", {}) or {}
    default_rgb = colormaps.get("DEFAULT", (255, 255, 255))

    for marker, img in fluor_images.items():
        rgb = colormaps.get(marker, default_rgb)
        col_img = colorize_channel(img, rgb).astype(np.float32, copy=False)
        comp += col_img

    comp = np.clip(comp, 0.0, 1.0)
    return (comp * 255.0).astype(np.uint8)


def get_corrected_cars_stack(
    nd2obj,
    c_index: int,
    position: int,
    ref_image: np.ndarray,
    fparams: Mapping[str, object],
) -> np.ndarray:
    """Return corrected CARS stack with shape (Z, H, W)."""
    total_z = nd2obj.sizes.get("z", 1)
    blur_sigma = float(fparams.get("sigma", 0.0) or 0.0)
    den = np.clip(ref_image.astype(np.float32, copy=False), 1e-6, None)

    slices = []
    for z_slice in range(total_z):
        raw = np.nan_to_num(nd2obj.get_frame_2D(v=position, c=c_index, z=z_slice))
        correlated = apply_east_shadows_filter(raw)
        div = correlated / den
        if blur_sigma > 0:
            div = gaussian(div, sigma=blur_sigma, preserve_range=True)
        slices.append(div.astype(np.float32, copy=False))
    return np.stack(slices, axis=0)


def get_fluorescence_stack(
    nd2obj,
    ch_index: int,
    position: int,
    fluoro_params: Mapping[str, object],
) -> np.ndarray:
    """Return processed fluorescence stack (Z, H, W) with optional per-slice Gaussian smoothing."""
    total_z = nd2obj.sizes.get("z", 1)
    gaussian_sigma = float(fluoro_params.get("gaussian_sigma", 0.0) or 0.0)

    slices = []
    for z_slice in range(total_z):
        raw = np.nan_to_num(nd2obj.get_frame_2D(v=position, c=ch_index, z=z_slice))
        if gaussian_sigma > 0:
            raw = gaussian(raw, sigma=gaussian_sigma, preserve_range=True)
        slices.append(raw.astype(np.float32, copy=False))
    return np.stack(slices, axis=0)