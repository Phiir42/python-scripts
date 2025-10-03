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


def process_fluorescence_channel(
    image_slice,
    cell_size,
    min_size,
    closing_radius,
    gaussian_sigma,
    fill_holes,
    threshold_method,
    offset,
    exclude_dark_regions=True,
    dark_threshold=50,
    min_hole_size=20000,
    debug=False,
):
    """(Unchanged docstring from original—thresholding + morphology to cell mask.)"""
    import matplotlib.pyplot as plt

    if image_slice.ndim != 2:
        raise ValueError(f"Expected a 2D array, but got shape {image_slice.shape}")

    image_slice = np.nan_to_num(image_slice)
    if gaussian_sigma and gaussian_sigma > 0:
        image_slice = gaussian(image_slice, sigma=gaussian_sigma, preserve_range=True)

    if exclude_dark_regions:
        preliminary_dark_mask = image_slice < dark_threshold
        labeled_dark = measure.label(preliminary_dark_mask)
        exclude_mask = np.zeros_like(labeled_dark, dtype=bool)
        for region in measure.regionprops(labeled_dark):
            if region.area >= min_hole_size:
                exclude_mask[tuple(region.coords.T)] = True
        valid_pixels = image_slice[~exclude_mask].ravel()
    else:
        exclude_mask = np.zeros_like(image_slice, dtype=bool)
        valid_pixels = image_slice.ravel()

    thr_m = threshold_method.lower()
    if len(valid_pixels) > 0:
        if thr_m == "otsu":
            base_threshold = threshold_otsu(valid_pixels)
        elif thr_m == "li":
            base_threshold = threshold_li(valid_pixels)
        elif thr_m == "triangle":
            base_threshold = threshold_triangle(valid_pixels)
        elif thr_m == "yen":
            base_threshold = threshold_yen(valid_pixels)
        else:
            base_threshold = threshold_otsu(valid_pixels)
    else:
        base_threshold = 999999

    final_threshold = base_threshold * offset
    binary_mask = image_slice > final_threshold
    binary_mask[exclude_mask] = False

    cleaned_mask = remove_small_objects(binary_mask, min_size=min_size)
    binary_closed = closing(cleaned_mask, disk(closing_radius))
    if fill_holes:
        binary_closed = ndi.binary_fill_holes(binary_closed)

    cell_mask = remove_small_objects(binary_closed, min_size=cell_size)

    if debug:
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


def robust_mad(a):
    """Median absolute deviation."""
    med = np.median(a)
    return np.median(np.abs(a - med))


def find_foci(
    image_slice,
    sigma,
    min_distance,
    min_size,
    std_dev_multiplier,
    remove_saturated,
    saturation_threshold,
    saturated_min_size,
    debug=False,
):
    """(Unchanged docstring from original—local maxima + watershed; re-include saturated regions.)"""
    if image_slice.ndim != 2:
        raise ValueError(f"find_foci expects a 2D array, got shape {image_slice.shape}")

    image_slice = np.nan_to_num(image_slice)

    # Exclude saturated pixels for thresholding only
    # (measure is already imported at top; no need to re-import)
    exclude_mask = np.zeros_like(image_slice, dtype=bool)
    if remove_saturated:
        labeled_sat = measure.label(image_slice > saturation_threshold)
        for region in measure.regionprops(labeled_sat):
            if region.area >= saturated_min_size:
                exclude_mask[tuple(region.coords.T)] = True

    smoothed = (
        gaussian(image_slice, sigma=sigma, preserve_range=True)
        if sigma > 0
        else image_slice.copy()
    )

    valid_pixels = smoothed[~exclude_mask].ravel()
    if len(valid_pixels) > 0:
        median_val = float(np.median(valid_pixels))
        mad_val = float(robust_mad(valid_pixels))  # <- call local function directly
        approx_std = 1.4826 * mad_val
        threshold_val = median_val + (std_dev_multiplier * approx_std)
    else:
        threshold_val = float("inf")

    mask_std = smoothed > threshold_val
    opened = opening(mask_std, disk(3))
    distance = ndi.distance_transform_edt(opened)
    local_maxi_coords = feature.peak_local_max(
        smoothed, min_distance=min_distance, labels=opened
    )
    local_maxi = np.zeros_like(opened, dtype=bool)
    if local_maxi_coords.size:
        local_maxi[tuple(local_maxi_coords.T)] = True

    markers = ndi.label(local_maxi)[0]
    labels_ws = segmentation.watershed(-distance, markers, mask=opened)

    final_mask = np.zeros_like(labels_ws, dtype=bool)
    for region in measure.regionprops(labels_ws):
        if region.area >= min_size:
            final_mask[tuple(region.coords.T)] = True

    # Reinstate saturated objects so they are not lost
    final_mask[exclude_mask] = True

    if debug:
        print(f"[DEBUG] threshold_val={threshold_val:.2f}")

    return final_mask
