"""Visualization helpers for debug overlays and saved composites."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.exposure import rescale_intensity
from skimage.segmentation import find_boundaries

from .constants import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


def debug_display_dapi(raw_dapi_slice: np.ndarray, dapi_mask: np.ndarray, pos_index: int) -> None:
    """
    Display a side-by-side DAPI max-projection and its mask overlay.

    Notes
    -----
    This function always shows a Matplotlib figure and closes it. Upstream code
    can control whether to call it (e.g., based on VERBOSE).
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(raw_dapi_slice, cmap="gray")
    axs[0].set_title(f"DAPI (pos={pos_index + 1}) - Max Projection")
    axs[0].axis("off")

    overlay = np.dstack([raw_dapi_slice] * 3).astype(np.float32)
    overlay = rescale_intensity(overlay, in_range="image", out_range=(0, 255)).astype(np.uint8)
    mask_red = np.zeros_like(overlay)
    mask_red[..., 0] = dapi_mask.astype(np.uint8) * 255

    axs[1].imshow(overlay)
    axs[1].imshow(mask_red, alpha=0.4)
    axs[1].set_title(f"DAPI Mask Overlay (pos={pos_index + 1})")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def debug_display_3way_segmentation(
    pure_lipid_mask: np.ndarray,
    lipid_lipofuscin_mask: np.ndarray,
    pure_lipofuscin_mask: np.ndarray,
    cell_mask: np.ndarray,
    auto_image: Optional[np.ndarray] = None,
    cars_image: Optional[np.ndarray] = None,
    pos_index: int = 0,
    title_suffix: str = "",
    myelin_mask: Optional[np.ndarray] = None,
    base_data_dir: Optional[str] = None,
    file_identifier: str = "",
    show_plots: bool = False,
) -> None:
    """
    Visualize and save a 3-way segmentation debug composite.

    Saves to "<base_data_dir>/Debug/<file_identifier>_posXX_debugseg_<suffix>.png".
    Displays the figure only if `show_plots` is True.

    Parameters
    ----------
    pure_lipid_mask, lipid_lipofuscin_mask, pure_lipofuscin_mask, cell_mask
        Boolean masks (H, W).
    auto_image, cars_image
        Optional 2D images shown in the top row; autoscaled to 8-bit for display.
    pos_index
        Zero-based position index; 1-based value is shown in titles.
    title_suffix
        String displayed in the overlay panel title and used in the saved filename.
    myelin_mask
        Optional myelin mask to include in the overlay.
    base_data_dir
        Base directory where the "Debug" folder is created for saving the figure.
        If None, the figure is not saved.
    file_identifier
        Filename stem used when saving the figure.
    show_plots
        If True, show the figure; it is always closed afterward.
    """
    import os  # local import to avoid unnecessary top-level dependency

    def to8(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if img is None:
            return None
        return rescale_intensity(img, in_range="image", out_range=(0, 255)).astype(np.uint8)

    def mask_rgb(mask: np.ndarray, rgb: tuple[int, int, int]) -> np.ndarray:
        out = np.zeros((*mask.shape, 3), dtype=np.uint8)
        m = mask.astype(np.uint8)
        out[..., 0] = m * rgb[0]
        out[..., 1] = m * rgb[1]
        out[..., 2] = m * rgb[2]
        return out

    auto8, cars8 = to8(auto_image), to8(cars_image)
    pure_rgb = mask_rgb(pure_lipid_mask, (0, 255, 0))
    lipo_rgb = mask_rgb(lipid_lipofuscin_mask, (255, 0, 0))
    lipofuscin_rgb = mask_rgb(pure_lipofuscin_mask, (255, 0, 255))
    cell_rgb = mask_rgb(cell_mask, (0, 0, 255))
    myelin_rgb = mask_rgb(myelin_mask, (0, 255, 255)) if myelin_mask is not None else None

    overlay_terms = [0.5 * pure_rgb, 0.5 * lipo_rgb, 0.5 * lipofuscin_rgb, 0.3 * cell_rgb]
    if myelin_rgb is not None:
        overlay_terms.append(0.5 * myelin_rgb)
    overlay = np.clip(sum(overlay_terms), 0, 255).astype(np.uint8)

    fig, axs = plt.subplots(2, 4, figsize=(18, 9))

    if auto8 is not None:
        axs[0, 0].imshow(auto8, cmap="gray")
        axs[0, 0].set_title(f"Autofluorescence (pos={pos_index + 1})")
    else:
        axs[0, 0].axis("off")

    if cars8 is not None:
        axs[0, 1].imshow(cars8, cmap="gray")
        axs[0, 1].set_title(f"CARS (pos={pos_index + 1})")
    else:
        axs[0, 1].axis("off")

    axs[0, 2].imshow(cell_rgb)
    axs[0, 2].set_title("Cell Mask")
    axs[0, 3].imshow(overlay)
    axs[0, 3].set_title(f"Overlay {title_suffix}")

    axs[1, 0].imshow(pure_rgb)
    axs[1, 0].set_title("Pure Lipid Mask")
    axs[1, 1].imshow(lipo_rgb)
    axs[1, 1].set_title("Lipid+Lipofuscin")
    axs[1, 2].imshow(lipofuscin_rgb)
    axs[1, 2].set_title("Pure Lipofuscin")

    if myelin_rgb is not None:
        axs[1, 3].imshow(myelin_rgb)
        axs[1, 3].set_title("Myelin (final)")
    else:
        axs[1, 3].axis("off")

    for row in axs:
        for ax in row:
            ax.axis("off")

    plt.tight_layout()

    # Always save to <base_data_dir>/Debug if provided
    if base_data_dir:
        debug_dir = os.path.join(base_data_dir, "Debug")
        os.makedirs(debug_dir, exist_ok=True)

        clean_suffix = title_suffix.replace("[", "").replace("]", "").replace(" ", "_")
        save_name = f"{file_identifier}_pos{pos_index + 1:02d}_debugseg_{clean_suffix}.png"
        out_path = os.path.join(debug_dir, save_name)
        fig.savefig(out_path, dpi=200)
        logger.info("Saved debug composite to: %s", out_path)

    if show_plots:
        plt.show()
    plt.close(fig)


def save_dapi_marker_overlay(
    dapi_mask: np.ndarray,
    marker_mask: np.ndarray,
    marker_name: str,
    out_path: str,
    config: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Save an RGBA PNG with DAPI (semi-transparent fill) and marker (opaque outline).

    - DAPI channel uses `config['colormaps']['DAPI']` if provided; else (0, 0, 255, 128).
    - Marker outline uses `config['colormaps'][marker_name]` if provided; else white.
    """
    height, width = dapi_mask.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    dapi_rgb = (config or {}).get("colormaps", {}).get("DAPI", (0, 0, 255))
    dapi_rgba = (int(dapi_rgb[0]), int(dapi_rgb[1]), int(dapi_rgb[2]), 128)

    marker_rgb = (config or {}).get("colormaps", {}).get(marker_name, (255, 255, 255))
    marker_rgba = (int(marker_rgb[0]), int(marker_rgb[1]), int(marker_rgb[2]), 255)

    outline = find_boundaries(marker_mask, mode="outer")

    rgba[dapi_mask.astype(bool), :4] = dapi_rgba
    sel = outline.astype(bool)
    rgba[sel, 0] = marker_rgba[0]
    rgba[sel, 1] = marker_rgba[1]
    rgba[sel, 2] = marker_rgba[2]
    rgba[sel, 3] = marker_rgba[3]

    Image.fromarray(rgba, mode="RGBA").save(out_path)
    logger.info("Saved overlay to: %s", out_path)
