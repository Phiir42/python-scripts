"""Reference image generation from ND2 CARS channel."""

from __future__ import annotations

import logging
import os

import numpy as np
from nd2reader import ND2Reader
from skimage.filters import gaussian
from tifffile import imwrite

from .constants import CARS_CH, LOG_LEVEL
from .filters import apply_east_shadows_filter

# Module logger aligned to VERBOSE via LOG_LEVEL
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


def _get_pixel_size_microns(nd2: ND2Reader) -> float:
    """Extract a scalar pixel size in microns from ND2 metadata with validation."""
    meta = getattr(nd2, "metadata", None)
    if not isinstance(meta, dict) or "pixel_microns" not in meta:
        raise ValueError("ND2 metadata missing 'pixel_microns'.")
    pixel_size = meta["pixel_microns"]
    try:
        px = float(pixel_size)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Unusable 'pixel_microns' value in ND2 metadata: {pixel_size!r}") from exc
    if px <= 0:
        raise ValueError(f"Non-positive pixel size in microns: {px}")
    return px


def tile_with_overlap(base: np.ndarray, target_shape: tuple[int, int], overlap: float = 0.05) -> np.ndarray:
    """
    Tile a smaller 2D image `base` into a larger `target_shape` with fractional overlap.
    Overlap creates a smooth blending between tiles using linear weighting.

    Parameters
    ----------
    base : 2D array
        The reference image to tile.
    target_shape : (H, W)
        Desired output size.
    overlap : float
        Fraction of tile size to overlap in each direction (0.05 = 5%).

    Returns
    -------
    tiled : 2D array
        New image of size `target_shape`.
    """
    bh, bw = base.shape
    th, tw = target_shape

    # compute stride with overlap
    stride_y = int(bh * (1 - overlap))
    stride_x = int(bw * (1 - overlap))

    # number of tiles needed
    ny = max(1, int(np.ceil((th - bh) / stride_y)) + 1)
    nx = max(1, int(np.ceil((tw - bw) / stride_x)) + 1)

    # prepare accumulator and weight map
    acc = np.zeros((th, tw), dtype=np.float32)
    wgt = np.zeros((th, tw), dtype=np.float32)

    # smooth blending mask
    yy = np.linspace(0, 1, bh)
    xx = np.linspace(0, 1, bw)
    wy = 1 - np.abs(yy - 0.5) * 2  # triangle windows for smooth edges
    wx = 1 - np.abs(xx - 0.5) * 2
    blend = np.outer(wy, wx).astype(np.float32)

    # tile placement
    for iy in range(ny):
        for ix in range(nx):
            y0 = iy * stride_y
            x0 = ix * stride_x
            y1 = min(y0 + bh, th)
            x1 = min(x0 + bw, tw)

            # compute cropping if the tile spills outside
            tile = base[: y1 - y0, : x1 - x0]
            bmask = blend[: y1 - y0, : x1 - x0]

            acc[y0:y1, x0:x1] += tile * bmask
            wgt[y0:y1, x0:x1] += bmask

    # Normalize blended region
    wgt = np.clip(wgt, 1e-6, None)
    return acc / wgt



def generate_reference_image(
    reference_file: str,
    output_path: str,
    blur_radius_microns: float,
    target_shape: tuple[int, int] | None = None,
    overlap: float = 0.05,
) -> np.ndarray:
    """
    Generate a normalized reference image from an ND2 file's CARS channel.

    Steps
    -----
    1. Load a single 2D frame from the CARS channel and replace NaNs with 0.
    2. Apply the directional "east-shadows" correlation filter.
    3. Convert the requested Gaussian blur radius (μm) to sigma (pixels).
    4. Apply Gaussian blur (preserve_range=True).
    5. Clip negative values (from directional correlation) to zero.
    6. Normalize by the image's maximum → values in [0, 1].
    7. Save as float32 TIFF to `output_path` and return the normalized array.

    Parameters
    ----------
    reference_file
        Path to the ND2 file containing the source image.
    output_path
        Path to write the float32 TIFF reference image. Parent dirs are created if missing.
    blur_radius_microns
        Gaussian blur radius in microns (> 0).

    Returns
    -------
    np.ndarray
        The normalized reference image as float32 with values in [0, 1].

    Raises
    ------
    ValueError
        If metadata is missing/invalid, blur radius is non-positive, or the processed
        image has no positive values to normalize.
    """
    if blur_radius_microns <= 0:
        raise ValueError(f"blur_radius_microns must be > 0, got {blur_radius_microns}")

    with ND2Reader(reference_file) as nd2:
        logger.info("Generating reference image from %s", reference_file)
        frame = nd2.get_frame_2D(c=CARS_CH)
        image = np.nan_to_num(frame, nan=0.0)
        pixel_size_microns = _get_pixel_size_microns(nd2)

    # Directional correlation filter
    filtered = apply_east_shadows_filter(image)

    # Convert blur radius (μm) to sigma (pixels): sigma = radius / pixel_size
    sigma_pixels = float(blur_radius_microns) / pixel_size_microns
    logger.info(
        "Applying Gaussian blur with sigma=%.3f px (%.3f μm radius / %.3f μm pixels)",
        sigma_pixels,
        blur_radius_microns,
        pixel_size_microns,
    )
    blurred = gaussian(filtered, sigma=sigma_pixels, preserve_range=True)

    # Clip negatives from directional filter so reference is non-negative
    blurred = np.clip(blurred, a_min=0, a_max=None)

    # Normalize to [0, 1] by max (guard against empty/zero images)
    max_val = float(np.max(blurred))
    if max_val <= 0:
        raise ValueError("Processed reference image has no positive values to normalize.")
    normalized = (blurred / max_val).astype(np.float32, copy=False)
    # If target_shape is given AND the reference is smaller → tile with overlap
    if target_shape is not None:
        ref_h, ref_w = normalized.shape
        tgt_h, tgt_w = target_shape
    
        if tgt_h > ref_h or tgt_w > ref_w:
            logger.info(
                "Reference image smaller than target; tiling with %.1f%% overlap to match %s",
                overlap * 100,
                target_shape,
            )
            normalized = tile_with_overlap(normalized, target_shape, overlap)


    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    imwrite(output_path, normalized)
    logger.info("Reference image written to %s", output_path)
    return normalized
