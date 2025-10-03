# lipid_analysis/imaging.py
from typing import Dict, Mapping, Sequence

import numpy as np
from skimage.exposure import rescale_intensity


def grayscale_autoscale(image_2d: np.ndarray) -> np.ndarray:
    """Rescale a 2D image to 0..255 (uint8)."""
    if image_2d.ndim != 2:
        raise ValueError(f"grayscale_autoscale expects 2D, got {image_2d.shape}")
    scaled = rescale_intensity(image_2d, in_range="image", out_range=(0, 255))
    return scaled.astype(np.uint8, copy=False)


def blend_fluorescence_cars(
    fluor_rgb: np.ndarray, cars_gray: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """Blend color fluorescence (H×W×3 uint8/float) with grayscale CARS (H×W)."""
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


def colorize_channel(image_2d: np.ndarray, rgb_color: Sequence[float]) -> np.ndarray:
    """Rescale 2D image to [0..1] and apply rgb_color (len=3 floats in [0,1]). Returns H×W×3 float."""
    if image_2d.ndim != 2:
        raise ValueError(f"colorize_channel expects 2D, got {image_2d.shape}")
    col = np.asarray(rgb_color, dtype=np.float32)
    if col.shape != (3,):
        raise ValueError(f"rgb_color must have 3 components, got shape {col.shape}")
    scaled = rescale_intensity(image_2d, in_range="image", out_range=(0.0, 1.0)).astype(
        np.float32, copy=False
    )
    return np.stack((scaled * col[0], scaled * col[1], scaled * col[2]), axis=-1)


def composite_fluorescence(
    fluor_images: Dict[str, np.ndarray], config_dict: Mapping
) -> np.ndarray:
    """
    Build an RGB composite by colorizing each marker channel and summing (clipped).
    fluor_images: dict[str, 2D ndarray]; config_dict['colormaps'][marker] -> [R,G,B] 0..255
    Returns: uint8 image (H×W×3).
    """
    if not fluor_images:
        raise ValueError("composite_fluorescence received an empty fluor_images dict")
    first_key = next(iter(fluor_images))
    H, W = fluor_images[first_key].shape
    comp = np.zeros((H, W, 3), dtype=np.float32)
    colormaps = config_dict.get("colormaps", {})
    default_rgb = (
        np.asarray(colormaps.get("DEFAULT", (255, 255, 255)), dtype=np.float32) / 255.0
    )

    for marker, img in fluor_images.items():
        if img.ndim != 2:
            raise ValueError(
                f"Channel for marker '{marker}' must be 2D, got {img.shape}"
            )
        rgb255 = np.asarray(colormaps.get(marker, default_rgb * 255), dtype=np.float32)
        comp += colorize_channel(img, rgb255 / 255.0).astype(np.float32, copy=False)

    comp = np.clip(comp, 0.0, 1.0)
    return (comp * 255.0).astype(np.uint8)
