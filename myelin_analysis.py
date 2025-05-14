"""
myelin_analysis.py

Sub-module for detecting filamentous myelin signal in pre‑processed CARS images.
Designed to be imported by `lipid_analysis.py`.

Example
-------
>>> from myelin_analysis import detect_myelin
>>> myelin_mask = detect_myelin(corrected_cars_slice,
...                             gaussian_sigma=1.5,
...                             threshold_method='otsu',
...                             offset=0.7,
...                             min_size=500,
...                             closing_radius=3,
...                             debug=True)
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import (gaussian, threshold_otsu, threshold_li,
                             threshold_yen, threshold_triangle)
from skimage.morphology import closing, disk, remove_small_objects
from skimage import exposure
from typing import Tuple

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _compute_threshold(image: np.ndarray, method: str = "otsu", offset: float = 1.0) -> float:
    """Return a global threshold value for *image* using the chosen *method*.

    *offset* (<1 ⇒ more permissive) rescales the base threshold so that faint
    myelin signal is retained.
    """
    img_flat = image.ravel().astype(float)

    method = method.lower()
    if method == "otsu":
        base = threshold_otsu(img_flat)
    elif method == "li":
        base = threshold_li(img_flat)
    elif method == "triangle":
        base = threshold_triangle(img_flat)
    elif method == "yen":
        base = threshold_yen(img_flat)
    else:
        raise ValueError(f"Unsupported threshold_method: {method}")

    return base * offset


def _debug_plot(input_img: np.ndarray, mask: np.ndarray, thr_val: float) -> None:
    """Render side‑by‑side debug figures (raw, mask, overlay)."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Raw input
    axs[0].imshow(input_img, cmap="gray")
    axs[0].set_title("CARS (pre‑processed)")
    axs[0].axis("off")

    # Binary mask
    axs[1].imshow(mask, cmap="gray")
    axs[1].set_title(f"Threshold mask (> {thr_val:.3f})")
    axs[1].axis("off")

    # Overlay (mask in green)
    overlay = exposure.rescale_intensity(input_img, out_range=(0.0, 1.0))
    overlay_rgb = np.dstack([overlay, overlay, overlay])
    overlay_rgb[mask, 1] = 1.0  # highlight mask in green channel
    axs[2].imshow(overlay_rgb)
    axs[2].set_title("Overlay")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()
    plt.close(fig)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def detect_myelin(
    cars_image: np.ndarray,
    *,
    gaussian_sigma: float = 1.0,
    threshold_method: str = "otsu",
    offset: float = 0.7,
    min_size: int = 200,
    closing_radius: int = 2,
    debug: bool = False,
) -> Tuple[np.ndarray, float]:
    """Return a binary mask of candidate myelin sheaths in *cars_image* and the
    fraction of the image they occupy (0–1).

    Parameters
    ----------
    cars_image : ndarray
        2‑D pre‑processed CARS slice (already East‑shadow filtered and reference‑
        corrected) in *float* or *uint* format.
    gaussian_sigma : float, optional
        Sigma (pixels) for Gaussian smoothing before threshold.
    threshold_method : {'otsu', 'li', 'triangle', 'yen'}, optional
        Global thresholding algorithm.
    offset : float, optional
        Multiplier applied to the base threshold; <1 captures fainter signal.
    min_size : int, optional
        Remove connected components smaller than this (pixels).
    closing_radius : int, optional
        Radius for morphological closing to bridge gaps (set 0 to skip).
    debug : bool, optional
        When *True*, displays diagnostic figures via Matplotlib.

    Returns
    -------
    mask : ndarray (bool)
        Binary mask where *True* marks putative myelin structures.
    frac : float
        (mask.sum / mask.size) – e.g. 0.46 ⇒ 46 % of the plane.
    """
    img = np.nan_to_num(cars_image.astype(float))

    # 1. Optional Gaussian blur (denoise, emphasise continuous filaments)
    if gaussian_sigma > 0:
        img = gaussian(img, sigma=gaussian_sigma, preserve_range=True)

    # 2. Permissive global threshold
    thr = _compute_threshold(img, method=threshold_method, offset=offset)
    mask = img > thr

    # 3. Morphological cleanup (bridge gaps; drop tiny speckles)
    if closing_radius > 0:
        mask = closing(mask, disk(closing_radius))

    mask = remove_small_objects(mask, min_size=min_size)

    # 4. Optional debug view
    if debug:
        _debug_plot(img, mask, thr)

    # 5. Compute area fraction
    fraction = float(np.count_nonzero(mask)) / mask.size
    return mask, fraction


__all__ = ["detect_myelin"]
