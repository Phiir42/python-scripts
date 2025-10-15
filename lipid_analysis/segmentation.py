"""Segmentation utilities for fluorescence and CARS images."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import ndimage as ndi
from skimage import feature, measure, segmentation
from skimage.filters import (
    gaussian,
    threshold_li,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
)
from skimage.morphology import closing, disk, opening, remove_small_objects

from .constants import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


def process_fluorescence_channel(
    image_slice: np.ndarray,
    cell_size: int,
    min_size: int,
    closing_radius: int,
    gaussian_sigma: float,
    fill_holes: bool,
    threshold_method: str,
    offset: float,
    exclude_dark_regions: bool = True,
    dark_threshold: float = 50,
    min_hole_size: int = 20_000,
    debug: bool = False,
) -> np.ndarray:
    """
    Threshold a fluorescence image and apply morphology to obtain a cell mask.

    Steps
    -----
    1) Optional Gaussian smoothing.
    2) Optionally exclude large dark regions from thresholding.
    3) Global threshold (Otsu/Li/Triangle/Yen) applied to valid pixels, then
       multiplied by `offset`.
    4) Remove small objects, morphological closing, optional hole filling.
    5) Final size filter to keep sufficiently large cell masks.

    Parameters
    ----------
    image_slice
        2D fluorescence image.
    cell_size
        Minimum area (in pixels) for the final cell mask objects.
    min_size
        Minimum area (in pixels) after initial thresholding (pre-closing).
    closing_radius
        Disk radius for the closing operation (post-cleaning).
    gaussian_sigma
        Smoothing sigma before thresholding; 0 disables.
    fill_holes
        If True, fill interior holes after closing.
    threshold_method
        One of {"otsu", "li", "triangle", "yen"} (case-insensitive).
    offset
        Multiplier applied to the base threshold.
    exclude_dark_regions
        If True, exclude large dark regions from determining the threshold.
    dark_threshold
        Pixel-value threshold used to seed dark regions.
    min_hole_size
        Minimum area for a dark region to be excluded from thresholding.
    debug
        If True, show a diagnostic figure via matplotlib.

    Returns
    -------
    np.ndarray
        Boolean mask of detected cells.
    """
    if image_slice.ndim != 2:
        raise ValueError(f"Expected a 2D array, but got shape {image_slice.shape}")

    image_slice = np.nan_to_num(image_slice)
    if gaussian_sigma and gaussian_sigma > 0:
        image_slice = gaussian(image_slice, sigma=float(gaussian_sigma), preserve_range=True)

    if exclude_dark_regions:
        preliminary_dark_mask = image_slice < dark_threshold
        labeled_dark = measure.label(preliminary_dark_mask)
        exclude_mask = np.zeros_like(labeled_dark, dtype=bool)
        for region in measure.regionprops(labeled_dark):
            if region.area >= int(min_hole_size):
                exclude_mask[tuple(region.coords.T)] = True
        valid_pixels = image_slice[~exclude_mask].ravel()
    else:
        exclude_mask = np.zeros_like(image_slice, dtype=bool)
        valid_pixels = image_slice.ravel()

    thr_m = (threshold_method or "otsu").lower()
    if valid_pixels.size > 0:
        if thr_m == "otsu":
            base_threshold = float(threshold_otsu(valid_pixels))
        elif thr_m == "li":
            base_threshold = float(threshold_li(valid_pixels))
        elif thr_m == "triangle":
            base_threshold = float(threshold_triangle(valid_pixels))
        elif thr_m == "yen":
            base_threshold = float(threshold_yen(valid_pixels))
        else:
            logger.warning("Unknown threshold_method '%s'; falling back to 'otsu'.", threshold_method)
            base_threshold = float(threshold_otsu(valid_pixels))
    else:
        base_threshold = float("inf")

    final_threshold = base_threshold * float(offset)
    binary_mask = image_slice > final_threshold
    binary_mask[exclude_mask] = False

    cleaned_mask = remove_small_objects(binary_mask, min_size=int(min_size))
    binary_closed = closing(cleaned_mask, disk(int(closing_radius)))
    if fill_holes:
        binary_closed = ndi.binary_fill_holes(binary_closed)

    cell_mask = remove_small_objects(binary_closed, min_size=int(cell_size))

    if debug:
        import matplotlib.pyplot as plt  # local import to avoid hard dependency at import time

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        axes[0].imshow(image_slice, cmap="gray")
        axes[0].set_title("Raw Fluorescence")
        axes[1].imshow(binary_mask, cmap="gray")
        axes[1].set_title(f"Thresholded (> {final_threshold:.2f})")
        axes[2].imshow(cleaned_mask, cmap="gray")
        axes[2].set_title("After First Cleaning")
        axes[3].imshow(binary_closed, cmap="gray")
        axes[3].set_title("After Closing + Fill")
        axes[4].imshow(cell_mask, cmap="gray")
        axes[4].set_title("Final Cleaned Mask")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    return cell_mask


def robust_mad(a: np.ndarray) -> float:
    """Median absolute deviation (MAD) of a 1D array."""
    med = float(np.median(a))
    return float(np.median(np.abs(a - med)))


def find_foci(
    image_slice: np.ndarray,
    sigma: float,
    min_distance: int,
    min_size: int,
    std_dev_multiplier: float,
    remove_saturated: bool,
    saturation_threshold: float,
    saturated_min_size: int,
    separate_objects: bool = True,
    morph_op: str = "opening",
    morph_radius: int = 3,
    min_snr: Optional[float] = None,
    debug: bool = False,
) -> np.ndarray:
    """
    Detect foci using robust global thresholding and optional watershed splitting.

    Steps
    -----
    1) Optional smoothing with Gaussian(sigma).
    2) Exclude saturated objects from threshold estimation.
    3) Robust global threshold: median + std_dev_multiplier * (1.4826 * MAD).
    4) Optional early exit if approximate SNR < min_snr.
    5) Morphological regularization (opening/closing).
    6) Split touching objects with watershed if `separate_objects=True`.
    7) Area filter by `min_size`.
    8) Reinstate saturated objects into the final mask.

    Parameters
    ----------
    image_slice
        2D image.
    sigma
        Gaussian sigma for pre-smoothing (0 disables).
    min_distance
        Minimum peak separation (pixels) for seed detection.
    min_size
        Minimum area (pixels) for final objects.
    std_dev_multiplier
        Multiplier on approx. std (1.4826*MAD) for thresholding.
    remove_saturated
        If True, exclude saturated objects from threshold estimation.
    saturation_threshold
        Pixel value regarded as saturated for exclusion.
    saturated_min_size
        Minimum area for an object to be considered saturated.
    separate_objects
        If True, split touching objects via watershed.
    morph_op
        "opening" (default), "closing", or "none".
    morph_radius
        Disk radius for morphological operation.
    min_snr
        If provided, return an empty mask when approx SNR < min_snr.
    debug
        If True, emit debug logs.

    Returns
    -------
    np.ndarray
        Boolean mask of detected foci.
    """
    if image_slice.ndim != 2:
        raise ValueError(f"find_foci expects a 2D array, got shape {image_slice.shape}")

    image_slice = np.nan_to_num(image_slice)

    # Exclude saturated pixels for thresholding only
    exclude_mask = np.zeros_like(image_slice, dtype=bool)
    if remove_saturated:
        labeled_sat = measure.label(image_slice > saturation_threshold)
        for region in measure.regionprops(labeled_sat):
            if region.area >= int(saturated_min_size):
                exclude_mask[tuple(region.coords.T)] = True

    smoothed = (
        gaussian(image_slice, sigma=float(sigma), preserve_range=True)
        if sigma > 0
        else image_slice.copy()
    )

    valid_pixels = smoothed[~exclude_mask].ravel()
    if valid_pixels.size > 0:
        median_val = float(np.median(valid_pixels))
        mad_val = float(robust_mad(valid_pixels))
        approx_std = 1.4826 * mad_val
        threshold_val = median_val + (float(std_dev_multiplier) * approx_std)

        if debug:
            snr_dbg = approx_std / max(median_val, 1e-9)
            logger.debug(
                "[FOCI] median=%.4f, std≈%.4f, SNR=%.4f, threshold=%.4f",
                median_val,
                approx_std,
                snr_dbg,
                threshold_val,
            )

        # Optional early low-contrast guard
        if min_snr is not None:
            snr = approx_std / max(median_val, 1e-9)
            if snr < float(min_snr):
                if debug:
                    logger.debug("[FOCI] Low-contrast frame: SNR=%.4f < %.4f → empty", snr, min_snr)
                return np.zeros_like(image_slice, dtype=bool)
    else:
        threshold_val = float("inf")

    mask_std = smoothed > threshold_val

    # Morphological regularization
    work = mask_std.copy()
    if morph_op == "opening" and morph_radius > 0:
        work = opening(work, disk(int(morph_radius)))
    elif morph_op == "closing" and morph_radius > 0:
        work = closing(work, disk(int(morph_radius)))
    # else "none": leave as-is

    if separate_objects:
        # Blob mode: split touching objects with watershed
        distance = ndi.distance_transform_edt(work)
        local_maxi_coords = feature.peak_local_max(
            smoothed, min_distance=int(min_distance), labels=work
        )
        local_maxi = np.zeros_like(work, dtype=bool)
        if local_maxi_coords.size:
            local_maxi[tuple(local_maxi_coords.T)] = True
        markers = ndi.label(local_maxi)[0]
        labels_ws = segmentation.watershed(-distance, markers, mask=work)
    else:
        # Filament mode: connected components (avoid over-segmentation)
        labels_ws = measure.label(work)

    final_mask = np.zeros_like(labels_ws, dtype=bool)
    for region in measure.regionprops(labels_ws):
        if region.area >= int(min_size):
            final_mask[tuple(region.coords.T)] = True

    # Reinstate saturated objects so they are not lost
    final_mask[exclude_mask] = True

    if debug:
        logger.debug(
            "[FOCI] thr=%.3f | morph_op=%s r=%d | separate=%s | kept_px=%d",
            threshold_val,
            morph_op,
            int(morph_radius),
            separate_objects,
            int(final_mask.sum()),
        )

    return final_mask
