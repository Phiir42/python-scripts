#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lipid_analysis.py

This module provides a workflow for analyzing lipid inclusions in microscopy images
using:
1) Fluorescence .nd2 files
2) CARS (Coherent Anti-Stokes Raman Scattering) .nd2 files
3) Optional hyperspectral series folders

Main steps:
-----------
1. Reference image generation:
   - A reference ND2 file is opened and used to create a normalized reference TIFF.

2. File pairing and offset logic:
   - The script identifies fluorescence vs. CARS .nd2 files (based on config-defined
     keywords).
   - Each file is given a "StacksX" key, with optional marker-based offsets for
     fluorescence, ensuring correct pairing with corresponding CARS images.

3. Image processing:
   - Fluorescence images are processed to generate a binary cell mask.
   - CARS images are processed to identify lipid droplets (foci).
   - The pipeline performs measurements of lipid inclusions (size, intensity) within
     each cell.

4. Hyperspectral analysis (if applicable):
   - Folders containing hyperspectral data are detected, and each ND2 in the folder is
     processed to build a series of corrected images and measure lipid intensities
     across different wavenumbers.

5. Results output:
   - The script saves a final Excel file containing detailed measurements for each
     cell (lipid objects, intensities) and a summary table.

Usage:
------
Execute lipid_analysis.py from within an environment where `nd2reader`, `scipy`,
`pandas`, `skimage`, and other dependencies are installed. The script reads paths
and parameters from a configuration file (e.g., `config_ovarianTissue.py`),
allowing easy adaptation for different tissue types or markers.

Example:
--------
python lipid_analysis.py

Dependencies:
-------------
- nd2reader
- scipy
- numpy
- pandas
- scikit-image (skimage)
- tifffile
- matplotlib
- opencv-python (cv2)
- pillow (PIL)

Note:
-----
The script assumes that all relevant .nd2 files follow a naming convention containing
the substring "StacksX" and a magnification keyword like "100X". The config file
governs how marker offsets are applied, and how hyperspectral data is processed.
Adjust the config file as needed for other tissue systems or filename patterns.
"""

import logging
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from nd2reader import ND2Reader
from PIL import Image
from scipy import ndimage as ndi
from skimage import feature, measure, segmentation
from skimage.exposure import rescale_intensity
from skimage.filters import (
    gaussian, threshold_otsu, threshold_li, threshold_triangle, threshold_yen
)
from skimage.morphology import closing, disk, opening, remove_small_objects
from skimage.segmentation import find_boundaries
from tifffile import imwrite

import argparse
import importlib.util

# Suppress excessive logs from nd2reader
logging.getLogger('nd2reader').setLevel(logging.ERROR)

EAST_SHADOWS_KERNEL = np.array(
    [
        [-1, 0, 1],
        [-2, 1, 2],
        [-1, 0, 1]
    ], dtype=np.float32
)


def load_config(py_file_path):
    """Dynamically load a Python config file as a module and return `config`."""
    spec = importlib.util.spec_from_file_location("cfg_module", py_file_path)
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)
    return cfg_mod.config


def apply_east_shadows_filter(image):
    """
    Apply the 'East' shadows filter (3x3 correlation) to the given 2D image.

    Parameters
    ----------
    image : ndarray
        The input 2D array (e.g., a single channel image).

    Returns
    -------
    filtered : ndarray
        The filtered image (float32).
    """
    img_float = image.astype(np.float32)
    filtered = ndi.correlate(img_float, EAST_SHADOWS_KERNEL, mode='reflect')
    return filtered


def find_nd2_files(directory):
    """
    Find and pair fluorescence and CARS .nd2 files in a directory based on matching keys.

    The key is derived from the filename substring "StacksX". Fluorescence files may
    have an offset (as determined by their marker) that modifies the stacks number so
    that they align with the matching CARS file.

    Also identifies folders containing hyperspectral data, using the keyword defined in
    config["file_keywords"]["hyperspectral_keyword"]. Any directory name matching that
    keyword is assumed to contain hyperspectral .nd2 series.

    Parameters
    ----------
    directory : str
        Path to the directory containing the .nd2 files.

    Returns
    -------
    paired_files : dict
        Mapping from the final pairing key (e.g. "Stacks3") to a dict with:
         - 'fluorescence': str  (path to the ND2 file)
         - 'CARS': str          (path to the ND2 file)
    hyperspectral_folders : list of str
        List of folder paths that contain the specified hyperspectral keyword.

    Raises
    ------
    ValueError
        If no valid 'StacksX' substring is found when parsing certain filenames.
    """
    hyperspectral_folders = []
    cars_list = []
    fluorescence_list = []

    file_kw = config["file_keywords"]
    mag_kw = file_kw["magnification_keyword"]
    hyperspec_kw = file_kw["hyperspectral_keyword"]

    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)

        # 1) If it's a folder with hyperspec keyword, store it
        if os.path.isdir(full_path) and hyperspec_kw in item:
            hyperspectral_folders.append(full_path)
            continue

        # 2) If it's not an ND2 or doesn't contain the magnification, skip
        if not (item.endswith(".nd2") and mag_kw in item):
            continue

        # 3) Parse it
        meta = parse_nd2_filename(item, config)
        print(f"File: {item} => meta: {meta}")
        if meta["contains_cars"]:
            cars_list.append((meta, full_path))
        else:
            # It's presumably a fluorescence .nd2
            # But we might confirm by checking if meta["markers_found"] is non-empty, etc.
            fluorescence_list.append((meta, full_path))

    # Now we want to pair them up
    paired_files = match_fluoro_and_cars(fluorescence_list, cars_list, config)
    return paired_files, hyperspectral_folders


def match_fluoro_and_cars(fluoro_list, cars_list, config):
    """
    Attempt to pair each fluorescence file with a corresponding CARS file
    based on:
      - same prefix
      - same stacks_label, stacks_number
      - (optional) marker overlap

    Returns a dict: { "someKey": {"fluorescence": path, "CARS": path}, ... }
    """
    paired_files = {}

    # If you need offsets, store them from config
    stack_offset_dict = config.get("stack_offset", {})

    for (f_meta, f_path) in fluoro_list:
        # 1) Possibly adjust f_meta["stacks_number"] if offset needed
        # If ANY marker in f_meta["markers_found"] -> sum offsets
        offset_total = 0
        for mk in f_meta["markers_found"]:
            offset_total += stack_offset_dict.get(mk, 0)
        f_stacks_num = None
        if f_meta["stacks_number"] is not None:
            f_stacks_num = f_meta["stacks_number"] + offset_total

        # We'll build an internal key to identify the "fluoro side"
        # ignoring the markers in the name, we unify by prefix + stacks_label
        # + corrected number
        for (c_meta, c_path) in cars_list:
            # Check prefix
            if f_meta["prefix"] != c_meta["prefix"]:
                continue
            # Check stacks_label
            if f_meta["stacks_label"] != c_meta["stacks_label"]:
                continue
            # Check stacks_number
            c_stacks_num = c_meta["stacks_number"]
            if f_stacks_num != c_stacks_num:
                continue

            # "marker overlap" is optional; only enforced if CARS has markers
            c_cars_markers = c_meta["markers_found"]
            if c_cars_markers:  # if not empty
                # Then require that these markers also appear in the fluoro file
                if not c_cars_markers.issubset(f_meta["markers_found"]):
                    continue

            # If we pass all checks, we have a match
            pair_key = (
                f"{f_meta['prefix']}-Stacks{f_meta['stacks_label']}"
                f"{f_stacks_num or ''}"
            )
            c_markers_sorted = "-".join(sorted(list(c_cars_markers))) or "NoMarkers"
            pair_key = (
                f"{f_meta['prefix']}-Stacks{f_meta['stacks_label']}"
                f"{f_stacks_num or ''}-{c_markers_sorted}"
            )
            paired_files[pair_key] = {
                "fluorescence": f_path,
                "CARS": c_path
            }
            # We can break here if we assume 1:1 matching
            break

    return paired_files


def parse_nd2_filename(filename, config):
    """
    Parse a filename like:
      "AD44-S1159-CARS2850-TUJ_Ck-100X-StacksNeurons3.nd2"

    Returns a dict with fields:
      {
        "base_no_ext": "AD44-S1159-CARS2850-TUJ_Ck-100X-StacksNeurons3",
        "prefix": "AD44-S1159",
        "markers_found": set(),  # e.g. {"TUJ_Ck"} if that substring is present
        "contains_cars": bool,   # True if "CARS2850" found
        "magnification": "100X" or None
        "stacks_label": "Neurons",  # the word after "Stacks"
        "stacks_number": 3 or None
      }
    """
    base_no_ext = filename.replace(".nd2", "")

    # 1) Identify if it contains the cars_keyword
    cars_keyword = config["file_keywords"]["cars_keyword"]
    contains_cars = (cars_keyword in base_no_ext)

    # 2) Identify magnification keyword if present
    mag_keyword = config["file_keywords"]["magnification_keyword"]
    magnification = mag_keyword if (mag_keyword in base_no_ext) else None

    # 3) Find all fluorescence markers that appear in the filename
    found_markers = set()
    for mk in config["file_keywords"]["fluorescence_markers"]:
        if mk in base_no_ext:
            found_markers.add(mk)

    # 4) Locate "Stacks..." substring at the end of the filename
    #    e.g. "StacksNeurons3" => label="Neurons", number=3
    match = re.search(r"(Stacks([A-Za-z]*)(\d*)$)", base_no_ext)
    stacks_label = None
    stacks_number = None
    if match:
        label_part = match.group(2)
        digit_part = match.group(3)
        stacks_label = label_part
        if digit_part:
            stacks_number = int(digit_part)

    # 5) Build the "prefix"
    #    Everything before the "Stacks..." portion, plus removing known magnification,
    #    cars keyword, and recognized markers
    prefix_candidate = base_no_ext
    if match:
        prefix_candidate = prefix_candidate[: match.start(1)]

    if magnification is not None:
        prefix_candidate = prefix_candidate.replace(mag_keyword, "")
    if contains_cars:
        prefix_candidate = prefix_candidate.replace(cars_keyword, "")

    for mk in found_markers:
        prefix_candidate = prefix_candidate.replace(mk, "")

    prefix_candidate = re.sub(r"[-_]+$", "", prefix_candidate)
    prefix_candidate = prefix_candidate.strip()

    meta = {
        "base_no_ext": base_no_ext,
        "prefix": prefix_candidate,
        "markers_found": found_markers,
        "contains_cars": contains_cars,
        "magnification": magnification,
        "stacks_label": stacks_label,
        "stacks_number": stacks_number
    }
    return meta


def get_file_key(filename):
    """
    Construct a pairing key from a filename containing a 'Stacks...' substring.

    1) Strips away .nd2 extension.
    2) Locates the 'Stacks...' portion at the end (Stacks + optional letters + optional digits).
    3) Takes everything before that as the 'prefix candidate'.
    4) Removes known keywords (fluorescence markers, cars keyword, magnification keyword)
       from the prefix candidate, plus any leftover dashes.
    5) Parses the 'Stacks' portion for optional label/digits (applying offsets if
       fluorescence).
    6) Rebuilds the final key with the cleaned prefix + Stacks suffix.

    Parameters
    ----------
    filename : str
        The ND2 filename (e.g. "Control-SCL1-CARS2850-100X-StacksAstrocytes3.nd2").

    Returns
    -------
    final_key : str
        e.g. "Control-SCL1-StacksAstrocytes3" or "AD33-S1178-StacksMicroglia".

    Raises
    ------
    ValueError
        If no valid 'Stacks...' substring is found.
    """
    base = filename.replace(".nd2", "")
    is_cars = config["file_keywords"]["cars_keyword"] in base

    match = re.search(r"(Stacks[A-Za-z]*\d*)$", base)
    if not match:
        raise ValueError(f"No valid 'Stacks...' found in filename: {filename}")

    stacks_part = match.group(1)
    prefix_candidate = base[: match.start(1)]

    removal_candidates = (
        config["file_keywords"]["fluorescence_markers"]
        + [
            config["file_keywords"]["cars_keyword"],
            config["file_keywords"]["magnification_keyword"]
        ]
    )
    for kw in removal_candidates:
        prefix_candidate = prefix_candidate.replace(kw, "")

    prefix_candidate = re.sub(r"-+", "-", prefix_candidate)
    prefix_candidate = prefix_candidate.strip("-")

    match_stacks = re.search(r"Stacks([A-Za-z]*)(\d*)", stacks_part)
    if not match_stacks:
        raise ValueError(f"Could not parse the 'Stacks' suffix: {stacks_part}")

    label_part = match_stacks.group(1)
    digit_part = match_stacks.group(2)

    if digit_part:
        stack_num = int(digit_part)
        if not is_cars:
            offset_total = 0
            for marker in config["file_keywords"]["fluorescence_markers"]:
                if marker in base:
                    offset_total += config["stack_offset"].get(marker, 0)
            stack_num += offset_total
        final_stacks_part = f"Stacks{stack_num}"
    else:
        final_stacks_part = f"Stacks{label_part}"

    if prefix_candidate:
        final_key = f"{prefix_candidate}-{final_stacks_part}"
    else:
        final_key = final_stacks_part

    return final_key


def create_custom_colormap(start_color, end_color):
    """
    Create a linear segmented colormap transitioning from black to the specified color.

    Parameters
    ----------
    start_color : tuple of int
        RGB color for the starting point (usually black, e.g., (0, 0, 0)).
    end_color : tuple of int
        RGB color for the ending point (e.g., green (0, 255, 0) or magenta (255, 0, 255)).

    Returns
    -------
    LinearSegmentedColormap
        A custom matplotlib colormap.
    """
    colors = [
        tuple(c / 255.0 for c in start_color),
        tuple(c / 255.0 for c in end_color)
    ]
    return LinearSegmentedColormap.from_list("custom_colormap", colors)


def process_fluorescence_channel(
    image_slice, cell_size, min_size, closing_radius, gaussian_sigma, fill_holes,
    threshold_method, offset, exclude_dark_regions=True,
    dark_threshold=50, min_hole_size=20000, debug=True
):
    """
    Process a single fluorescence image slice and create a binary mask.

    1) Replaces NaNs with zeros.
    2) Applies Gaussian smoothing.
    3) Thresholds using Otsu's method.
    4) Performs morphological closing and optionally fills holes.
    5) Removes small objects.

    Parameters
    ----------
    image_slice : ndarray
        2D fluorescence image data.
    min_size : int
        Minimum size (in pixels) for objects to be kept.
    closing_radius : int
        Radius for the morphological closing operation.
    gaussian_sigma : float
        Sigma for the Gaussian filter.
    fill_holes : bool
        Whether to fill holes in the binary mask.
    threshold_method : str
        The thresholding method to use ('otsu', 'li', 'triangle', 'yen').
    offset : float
        Multiplier for the threshold value (default 1.0).
    exclude_dark_regions : bool
        Whether to exclude large dark holes from thresholding.
    dark_threshold : float
        Pixel intensity below which pixels are considered "dark".
    min_hole_size : int
        Minimum connected area (in pixels) for a dark region to be a hole.

    Returns
    -------
    cleaned_mask : ndarray (bool)
        Binary mask after thresholding and morphological operations.

    Raises
    ------
    ValueError
        If input slice is not 2D.
    """
    if image_slice.ndim != 2:
        raise ValueError(f"Expected a 2D array, but got shape {image_slice.shape}")

    image_slice = np.nan_to_num(image_slice)

    if exclude_dark_regions:
        preliminary_dark_mask = image_slice < dark_threshold
        labeled_dark = measure.label(preliminary_dark_mask)
        props = measure.regionprops(labeled_dark)
        exclude_mask = np.zeros_like(labeled_dark, dtype=bool)
        for region in props:
            if region.area >= min_hole_size:
                coords = region.coords
                exclude_mask[coords[:, 0], coords[:, 1]] = True
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
        base_threshold = 999999;

    final_threshold = base_threshold * offset
    binary_mask = image_slice > final_threshold
    binary_mask[exclude_mask] = False
    
    cleaned_mask = remove_small_objects(binary_mask, min_size=min_size)

    binary_closed = closing(cleaned_mask, disk(closing_radius))
    if fill_holes:
        binary_closed = ndi.binary_fill_holes(binary_closed)

    cell_mask = remove_small_objects(binary_closed, min_size=cell_size)
    
    # =========== DEBUG PLOTS IF DESIRED ================
    if debug:
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        axes[0].imshow(image_slice, cmap='gray')
        axes[0].set_title("Raw Fluorescence")

        # Show thresholded mask (before morphological closing/fill)
        axes[1].imshow(binary_mask, cmap='gray')
        axes[1].set_title(f"Thresholded (> {final_threshold:.2f})")

        axes[2].imshow(cleaned_mask, cmap='gray')
        axes[2].set_title("After First Cleaning")

        axes[3].imshow(binary_closed, cmap='gray')
        axes[3].set_title("After Closing + Fill")
        
        axes[4].imshow(cell_mask, cmap='gray')
        axes[4].set_title("Final Cleaned Mask")

        for ax in axes:
            ax.axis('off')
        plt.suptitle(f"Method: {threshold_method}, offset={offset:.2f}, closing={closing_radius}")
        plt.tight_layout()
        plt.show()
    # ====================================================
    
    return cell_mask


def robust_mad(a):
    """
    Simple helper to compute median absolute deviation.
    Returns median(|x - median(x)|).
    """
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
    debug=False
):
    """
    Identify foci within a CARS image slice using local maxima and thresholding,
    optionally ignoring large saturated regions.

    Steps:
    1) (Optional) Remove large saturated objects above 'saturation_threshold' intensity
       if 'remove_saturated' is True.
    2) (Optional) Gaussian smoothing if sigma>0.
    3) Compute robust median+MAD for threshold => create 'mask_std'.
    4) Morphological opening, local maxima detection, watershed.
    5) Retain objects >= 'min_size'.
    6) If debug=True, display step-by-step figures.

    Parameters
    ----------
    image_slice : ndarray
        2D CARS image data.
    sigma : float
        Sigma for the Gaussian filter (if >0).
    min_distance : int
        Minimum distance between local maxima.
    min_size : int
        Minimum area (in pixels) for each focus to be retained.
    std_dev_multiplier : float
        Multiplier for the robust standard deviation threshold.
    remove_saturated : bool
        Whether to exclude very bright objects from thresholding logic.
    saturation_threshold : float
        Intensity above which we consider the object "saturated."
    saturated_min_size : int
        Minimum connected area to be considered a large saturated region.
    debug : bool
        If True, display debug prints and intermediate plots.

    Returns
    -------
    final_mask : ndarray (bool)
        Binary mask for the identified foci.
    """
    import matplotlib.pyplot as plt

    image_slice = np.nan_to_num(image_slice)
    original = image_slice.copy()

    # 1) Exclude large saturated objects
    exclude_mask = np.zeros_like(image_slice, dtype=bool)
    if remove_saturated:
        preliminary_sat_mask = image_slice > saturation_threshold
        labeled_sat = measure.label(preliminary_sat_mask)
        if debug:
            print(f"[DEBUG] Checking for saturated objects above {saturation_threshold}...")

        for region in measure.regionprops(labeled_sat):
            if region.area >= saturated_min_size:
                coords = region.coords
                exclude_mask[coords[:, 0], coords[:, 1]] = True

        if debug:
            sat_count = np.count_nonzero(exclude_mask)
            print(f"[DEBUG] Excluding {sat_count} pixels from saturated regions.")

    # 2) Optional smoothing
    if sigma > 0:
        smoothed = gaussian(image_slice, sigma=sigma, preserve_range=True)
    else:
        smoothed = image_slice.copy()

    # Use only non-excluded pixels for threshold calculation
    valid_pixels = smoothed[~exclude_mask].ravel()

    # 3) Compute robust median+MAD threshold => mask_std
    if len(valid_pixels) > 0:
        median_val = np.median(valid_pixels)
        mad_val = robust_mad(valid_pixels)
        approx_std = 1.4826 * mad_val
        threshold_val = median_val + (std_dev_multiplier * approx_std)
    else:
        threshold_val = 999999

    mask_std = smoothed > threshold_val

    # 4) Morphological opening => watershed => area filtering
    opened = opening(mask_std, disk(3))
    distance = ndi.distance_transform_edt(opened)
    local_maxi_coords = feature.peak_local_max(
        smoothed, min_distance=min_distance, labels=opened
    )
    local_maxi = np.zeros_like(opened, dtype=bool)
    local_maxi[tuple(local_maxi_coords.T)] = True

    markers = ndi.label(local_maxi)[0]
    labels_ws = segmentation.watershed(-distance, markers, mask=opened)

    final_mask = np.zeros_like(labels_ws, dtype=bool)
    for region in measure.regionprops(labels_ws):
        if region.area >= min_size:
            final_mask[tuple(region.coords.T)] = True

    # Exclude saturated
    final_mask[exclude_mask] = True

    # ===== DEBUG OUTPUTS =====
    if debug:
        print("[DEBUG] =============== find_foci DEBUG (No Otsu) ===============")
        print(f"[DEBUG] median_val={median_val:.2f}, MAD={mad_val:.2f}, "
              f"threshold_val={threshold_val:.2f}")

        fig, axs = plt.subplots(2, 3, figsize=(16, 8))
        axs = axs.ravel()

        # Panel 0: Original + saturated overlay
        axs[0].imshow(original, cmap='gray')
        axs[0].set_title("Raw Input")
        axs[0].axis('off')
        if remove_saturated and np.any(exclude_mask):
            sat_overlay = np.zeros((*original.shape, 3), dtype=np.uint8)
            sat_overlay[exclude_mask, 0] = 255  # color saturated in red
            axs[0].imshow(sat_overlay, alpha=0.4)

        # Panel 1: Smoothed
        axs[1].imshow(smoothed, cmap='gray')
        axs[1].set_title(f"Smoothed (sigma={sigma})")
        axs[1].axis('off')

        # Panel 2: mask_std
        axs[2].imshow(mask_std, cmap='gray')
        axs[2].set_title(f"mask_std > {threshold_val:.1f}")
        axs[2].axis('off')

        # Panel 3: opened
        axs[3].imshow(opened, cmap='gray')
        axs[3].set_title("After Opening")
        axs[3].axis('off')

        # Panel 4: local maxima
        local_show = np.dstack([opened*255, opened*255, opened*255]).astype(np.uint8)
        # Mark local maxima in green
        local_show[local_maxi, 1] = 255
        axs[4].imshow(local_show)
        axs[4].set_title("Local Maxima (green)")
        axs[4].axis('off')

        # Panel 5: watershed + final
        labeled_overlay = np.zeros((*final_mask.shape, 3), dtype=np.uint8)
        rng = np.random.default_rng(1234)
        for region in measure.regionprops(labels_ws):
            color = rng.integers(0, 255, size=3)
            for rr, cc in region.coords:
                labeled_overlay[rr, cc] = color
        labeled_overlay[~final_mask] //= 4
        axs[5].imshow(labeled_overlay)
        axs[5].set_title(f"Watershed + Filtered (min_size={min_size})")
        axs[5].axis('off')

        plt.tight_layout()
        plt.show()
        plt.close(fig)

    return final_mask


def process_hyperspectral_series(spectrum_folder, reference_image, output_path, foci_params):
    """
    Process a hyperspectral series to extract lipid droplet intensities.

    Steps:
    1) Reads 32 ND2 files in 'spectrum_folder' (each ~1 wavelength).
    2) East-shadows filter and reference correction.
    3) Use 9th corrected image to generate a lipid mask (via find_foci).
    4) For each droplet, extract mean intensities across all wavelengths.
    5) Save two sheets to 'output_path':
       - 'Raw Data' (Wavenumber 1..32)
       - 'Normalized Data' (each row normalized to its max,
         columns labeled by computed wavenumbers).
    6) Also compute ratio_map for wavenumber 19 vs 9, color-coded from yellow->red.

    Parameters
    ----------
    spectrum_folder : str
        Path to the folder containing the hyperspectral series.
    reference_image : ndarray
        Precomputed reference image for CARS correction.
    output_path : str
        Path to save the results spreadsheet.
    foci_params : dict
        Parameters for the find_foci function.

    Returns
    -------
    None
        Saves an Excel file with two sheets ('Raw Data' and 'Normalized Data').
    """
    nd2_files = sorted([
        os.path.join(spectrum_folder, f)
        for f in os.listdir(spectrum_folder)
        if f.endswith('.nd2')
    ])
    if len(nd2_files) != 32:
        raise ValueError(
            f"Expected 32 images in the series, but found {len(nd2_files)}."
        )

    corrected_images = []
    for nd2_file in nd2_files:
        with ND2Reader(nd2_file) as nd2:
            raw_image = np.nan_to_num(nd2.get_frame_2D(c=2))
            correlated_image = apply_east_shadows_filter(raw_image)
            c_image = np.nan_to_num(correlated_image / reference_image)
            corrected_images.append(c_image)

    mask_image = corrected_images[8]  # 9th image is index 8
    lipid_mask = find_foci(mask_image, **foci_params)

    # (Optional) visualize the mask overlay
    visualize_hyperspectral_mask_overlay(mask_image, lipid_mask)

    # Extract intensities for each droplet across all 32 corrected images
    lipid_labels = measure.label(lipid_mask)
    lipid_data = []

    for lipid in measure.regionprops(lipid_labels):
        lipid_id = lipid.label
        intensities = []
        for img in corrected_images:
            mean_intensity = np.mean(img[lipid.coords[:, 0], lipid.coords[:, 1]])
            intensities.append(mean_intensity)
        lipid_data.append([lipid_id] + intensities)

    columns_raw = ["Lipid ID"] + [f"Wavenumber {i+1}" for i in range(32)]
    lipid_df_raw = pd.DataFrame(lipid_data, columns=columns_raw)
    lipid_df_norm = lipid_df_raw.copy()

    def compute_wavenumber(lambda_nm):
        """Compute wavenumber from wavelength in nm using 1e7*(1/lambda_nm - 1/1031)."""
        return 1.0e7 * ((1.0 / lambda_nm) - (1.0 / 1031.0))

    wavelengths_nm = [801.0 - 0.5 * i for i in range(32)]
    wavenumbers = [compute_wavenumber(wl) for wl in wavelengths_nm]

    new_col_names = ["Lipid ID"] + [f"{wn:.2f}" for wn in wavenumbers]
    data_to_normalize = lipid_df_norm.iloc[:, 1:]
    row_maxes = data_to_normalize.max(axis=1).replace({0: 1})
    lipid_df_norm.iloc[:, 1:] = data_to_normalize.div(row_maxes, axis=0)
    lipid_df_norm.columns = new_col_names

    with pd.ExcelWriter(output_path) as writer:
        lipid_df_raw.to_excel(writer, sheet_name='Raw Data', index=False)
        lipid_df_norm.to_excel(writer, sheet_name='Normalized Data', index=False)

    print(f"Hyperspectral lipid intensities saved to {output_path}")

    idx_2850 = 1 + 8   # column index for "Wavenumber 9"
    idx_2930 = 1 + 18  # column index for "Wavenumber 19"

    ratio_map = np.full_like(lipid_labels, fill_value=-1, dtype=np.float32)
    ratio_values = []

    for row in lipid_df_raw.itertuples(index=False):
        lipid_id = row[0]
        intens_2850 = row[idx_2850]
        intens_2930 = row[idx_2930]

        if intens_2850 > 0:
            ratio_val = intens_2930 / intens_2850
        else:
            ratio_val = 0.0

        ratio_map[lipid_labels == lipid_id] = ratio_val
        ratio_values.append(ratio_val)

    if len(ratio_values) == 0:
        print("No droplets found, skipping ratio heatmap.")
        return

    ratio_min = np.min(ratio_values)
    ratio_max = np.max(ratio_values) if np.max(ratio_values) > 0 else 1.0
    ratio_norm = (ratio_map - ratio_min) / (ratio_max - ratio_min + 1e-9)
    ratio_norm_clipped = np.clip(ratio_norm, 0.0, 1.0)

    cmap = LinearSegmentedColormap.from_list(
        'yellow_red', [(1.0, 1.0, 0.0), (1.0, 0.0, 0.0)]
    )
    ratio_rgba = cmap(ratio_norm_clipped)
    ratio_rgb = (ratio_rgba[..., :3] * 255).astype(np.uint8)

    bg_mask = (ratio_map < 0)
    ratio_rgb[bg_mask] = [0, 0, 0]

    fig = plt.figure(figsize=(6, 6))
    plt.imshow(ratio_rgb)
    plt.title("Droplet Ratio Map (2930 / 2850)")
    plt.axis('off')
    plt.show()
    plt.close(fig)

    ratio_bgr = cv2.cvtColor(ratio_rgb, cv2.COLOR_RGB2BGR)
    out_path_ratio = os.path.join(spectrum_folder, "Ratio_2930_over_2850.png")
    cv2.imwrite(out_path_ratio, ratio_bgr)
    print(f"Ratio heatmap saved to {out_path_ratio}")


def visualize_hyperspectral_mask_overlay(cars_image, lipid_mask):
    """
    Visualize the overlay of the CARS image and lipid mask from the hyperspectral series.

    Parameters
    ----------
    cars_image : ndarray
        2D CARS image data (already corrected).
    lipid_mask : ndarray
        Binary mask for the identified lipid droplets.

    Returns
    -------
    None
        Displays matplotlib figures with the grayscale CARS image, lipid mask, and overlay.
    """

    def create_rgb_mask(mask, color):
        """
        Create an RGB image from a boolean mask, painting 'color' where mask is True.
        """
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for i in range(3):
            rgb[..., i] = mask * color[i]
        return rgb

    max_val = cars_image.max() if cars_image.max() > 0 else 1
    grayscale_8bit = (cars_image / max_val * 255).astype(np.uint8)
    lipid_mask_rgb = create_rgb_mask(lipid_mask, [255, 255, 0])

    grayscale_rgb = np.stack([grayscale_8bit] * 3, axis=-1)
    overlay_rgb = np.clip(
        0.5 * grayscale_rgb + 0.5 * lipid_mask_rgb, 0, 255
    ).astype(np.uint8)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(grayscale_8bit, cmap='gray')
    axs[0].set_title("CARS (9th Hyperspectral Image)")
    axs[0].axis('off')

    axs[1].imshow(lipid_mask_rgb)
    axs[1].set_title("Lipid Mask (Yellow)")
    axs[1].axis('off')

    axs[2].imshow(overlay_rgb)
    axs[2].set_title("Overlay")
    axs[2].axis('off')

    plt.show()
    plt.close(fig)


def analyze_intracellular_objects(cars_mask, cell_mask, cars_image,
                                  file_name, z_stack, pixel_size_microns):
    """
    Analyze lipid inclusions within each cell in the image.

    1) Labels cells from cell_mask, labels lipids from cars_mask.
    2) For each cell region, compute area, lipid objects in that region, etc.
    3) Summarizes lipid size, intensity, and proportion for each cell.

    Parameters
    ----------
    cars_mask : ndarray (bool)
        Mask for CARS lipid droplets.
    cell_mask : ndarray (bool)
        Mask for cells.
    cars_image : ndarray
        Original (or corrected) CARS image.
    file_name : str
        Name of the .nd2 file being processed.
    z_stack : int
        Current z-stack index being analyzed.
    pixel_size_microns : float
        Pixel size in microns.

    Returns
    -------
    results : list of dict
        Detailed results of lipid inclusions for each cell-lipid object pair.
    summary : list of dict
        One-row summary per cell, indicating total lipid count and area fraction.
    """
    labeled_cells = measure.label(cell_mask)
    labeled_lipids = measure.label(cars_mask)
    results = []
    summary = []

    for cell in measure.regionprops(labeled_cells):
        cell_id = cell.label
        cell_area = cell.area
        cell_area_um2 = cell_area * (pixel_size_microns ** 2)

        cell_mask_region = (labeled_cells == cell_id)
        lipid_objects_in_cell = labeled_lipids * cell_mask_region

        lipid_count = 0
        total_lipid_area = 0.0

        for lipid in measure.regionprops(
            lipid_objects_in_cell, intensity_image=cars_image
        ):
            lipid_size_pixels = lipid.area
            lipid_size_um2 = lipid_size_pixels * (pixel_size_microns ** 2)
            lipid_count += 1
            total_lipid_area += lipid_size_um2

            results.append({
                'file_name': file_name,
                'z_stack': z_stack,
                'cell_id': cell_id,
                'cell_area': cell_area,
                'cell_area_um2': cell_area_um2,
                'lipid_size_pixels': lipid_size_pixels,
                'lipid_size_um2': lipid_size_um2,
                'lipid_intensity': lipid.mean_intensity
            })

        if cell_area_um2 > 0:
            lipid_percentage = (total_lipid_area / cell_area_um2) * 100.0
        else:
            lipid_percentage = 0.0

        summary.append({
            'file_name': file_name,
            'z_stack': z_stack,
            'cell_id': cell_id,
            'cell_area': cell_area,
            'cell_area_um2': cell_area_um2,
            'lipid_count': lipid_count,
            'total_lipid_area_um2': total_lipid_area,
            'lipid_percentage': lipid_percentage
        })

    return results, summary


def generate_reference_image(reference_file, output_path, blur_radius_microns):
    """
    Generate a normalized reference image from the CARS signal in a reference ND2 file.

    1) Reads channel 2 from the ND2 (assumed to be CARS).
    2) Applies East shadows filter.
    3) Gaussian-blurs the image by the specified radius (in microns).
    4) Normalizes so the final reference image is in [0..1].
    5) Saves an 8-bit version of that reference as a TIFF.

    Parameters
    ----------
    reference_file : str
        Path to the reference .nd2 file.
    output_path : str
        Path to save the normalized reference image (as a TIFF).
    blur_radius_microns : float
        Desired Gaussian blur radius in microns.

    Returns
    -------
    reference_image : ndarray
        The normalized reference image (2D array scaled from 0 to 1).

    Raises
    ------
    ValueError
        If the blurred image has no valid data.
    """
    with ND2Reader(reference_file) as ref_nd2:
        print(f"Generating reference image from: {reference_file}")
        reference_img = np.nan_to_num(ref_nd2.get_frame_2D(c=2))
        pixel_size_microns = ref_nd2.metadata['pixel_microns']

    reference_img = apply_east_shadows_filter(reference_img)
    sigma_pixels = blur_radius_microns / pixel_size_microns
    print(
        f"Applying Gaussian blur (sigma={sigma_pixels:.2f} pixels) "
        f"from {blur_radius_microns} microns"
    )

    blurred_reference = gaussian(
        reference_img, sigma=sigma_pixels, preserve_range=True
    )
    blurred_ref_max = np.max(blurred_reference)
    original_max = np.max(reference_img)

    if blurred_ref_max <= 0:
        raise ValueError("Blurred reference has no valid data.")

    blurred_reference_scaled = blurred_reference * (original_max / blurred_ref_max)
    max_value = np.max(blurred_reference_scaled)
    if max_value <= 0:
        raise ValueError(
            "Reference image has no valid intensity data after preprocessing."
        )

    normalized_reference = blurred_reference_scaled / max_value
    imwrite(output_path, normalized_reference.astype(np.float32))
    print(f"Reference image saved to {output_path}")

    return normalized_reference


def get_marker_color(marker_name, config_colors):
    """
    Return a custom colormap for the given marker name.
    If marker is not in config_colors, return the default colormap.

    Parameters
    ----------
    marker_name : str
        Marker name, e.g. 'DAPI', 'IBA1'.
    config_colors : dict
        Dictionary of marker->(R,G,B) color tuples.

    Returns
    -------
    LinearSegmentedColormap
        A custom colormap from black to the specified color.
    """
    if marker_name in config_colors:
        end_color = config_colors[marker_name]
    else:
        end_color = config_colors["DEFAULT"]
    return create_custom_colormap((0, 0, 0), end_color)


def process_nd2_pair(fluorescence_path, cars_path, reference_image):
    """
    Process paired fluorescence and CARS .nd2 files with reference correction.

    Steps:
    1) Identify the "analysis marker" for the main fluorescence channel.
    2) Read & correct the CARS channel (max projection), find lipid droplets.
    3) For each cell marker in the config, build a cell mask (no union).
    4) Measure lipid objects inside the cell mask.
    5) Return combined results from all markers.
    6) Save a DAPI+Marker overlay (transparent background) for each marker
       if DAPI is available.
    7) Gather all fluorescence channels for a final color composite,
       plus grayscale CARS, plus 50/50 blend.

    Parameters
    ----------
    fluorescence_path : str
        Path to the fluorescence ND2 file.
    cars_path : str
        Path to the CARS ND2 file.
    reference_image : ndarray
        Precomputed reference image for CARS correction.

    Returns
    -------
    all_positions_results : list of dict
        Detailed results of lipid inclusions for each cell from each marker.
    all_positions_summary : list of dict
        Summary of lipid objects per cell from each marker.
    """
    foci_params = config["morphology_params"]["foci_params"]
    fluorescence_params = config["morphology_params"]["fluorescence_params"]

    # Identify which marker is in this filename (for analysis display)
    analysis_marker_hit = None
    for test_marker in config["file_keywords"]["fluorescence_markers"]:
        if test_marker in fluorescence_path:
            analysis_marker_hit = test_marker
            break

    if analysis_marker_hit is None:
        print(f"No recognized marker in {fluorescence_path}")
        return [], []

    channel_map = config["channel_map"]

    # Try to parse "Stacks..." from the filename with get_file_key, but if it fails:
    try:
        file_key = get_file_key(os.path.basename(fluorescence_path))
    except ValueError:
        file_key = os.path.basename(fluorescence_path)

    match_stacks = re.search(r"Stacks([A-Za-z]+)", file_key)
    if match_stacks:
        stacks_label = match_stacks.group(1)
    else:
        stacks_label = ""

    cell_marker_map = config.get("cell_marker_map", {})
    if stacks_label in cell_marker_map:
        chosen_cell_markers = cell_marker_map[stacks_label]
        print(f"Using custom marker set for '{stacks_label}': {chosen_cell_markers}")
    else:
        chosen_cell_markers = config.get("cell_markers", [])
        print(f"No custom marker map for '{stacks_label}', using: {chosen_cell_markers}")

    all_positions_results = []
    all_positions_summary = []

    with ND2Reader(fluorescence_path) as fluoro_nd2, ND2Reader(cars_path) as cars_nd2:
        print(f"File: {fluorescence_path}")
        fluoro_nd2.iter_axes = 'v'
        cars_nd2.iter_axes = 'v'
        pixel_size_microns = fluoro_nd2.metadata['pixel_microns']

        def max_project_cars(
            nd2obj,
            c_index,
            position,
            reference_image,
            foci_params
        ):
            """Read z-slices from channel c_index, apply East-shadows, do max-projection."""
            
            # Extract the blur sigma from foci_params
            gaussian_sigma = foci_params.get("sigma")
    
            z_stack_slices_cars = []
            total_z = nd2obj.sizes.get('z', 1)
        
            for z_slice in range(total_z):
                raw_sl = np.nan_to_num(
                    nd2obj.get_frame_2D(v=position, c=c_index, z=z_slice)
                )
                # 1) East shadows filter
                correlated_sl = apply_east_shadows_filter(raw_sl)
        
                # 2) Divide by reference
                slice_div = np.nan_to_num(correlated_sl / reference_image)
        
                # 3) Gaussian blur
                blurred_sl = gaussian(slice_div, sigma=gaussian_sigma, preserve_range=True)
        
                z_stack_slices_cars.append(blurred_sl)
        
            # 4) Finally, take maximum-intensity projection across Z
            final_mip = np.max(np.array(z_stack_slices_cars), axis=0)
            return final_mip

        def max_project_fluorescence(nd2obj, ch_index, position, fluoro_params):
            """
            Read each Z-slice from a fluorescence channel, apply an optional Gaussian blur
            (sigma>0), then return a maximum-intensity projection.
            """
            z_stack_slices = []
            total_z = nd2obj.sizes.get('z', 1)
        
            # Pull the sigma (or default to 0 => skip blur)
            gaussian_sigma = fluoro_params.get("gaussian_sigma", 0.0)
        
            for z_slice in range(total_z):
                raw_slice = nd2obj.get_frame_2D(v=position, c=ch_index, z=z_slice)
                raw_slice = np.nan_to_num(raw_slice)
        
                # If sigma>0, apply a blur to this slice before storing it
                if gaussian_sigma > 0:
                    blurred_slice = gaussian(raw_slice, sigma=gaussian_sigma, preserve_range=True)
                    z_stack_slices.append(blurred_slice)
                else:
                    # No blur
                    z_stack_slices.append(raw_slice)
        
            # Finally, do a max projection of the slice stack
            final_mip = np.max(np.array(z_stack_slices), axis=0)
            return final_mip

        for pos in range(fluoro_nd2.sizes['v']):
            fluoro_nd2.default_coords['v'] = pos
            cars_nd2.default_coords['v'] = pos

            file_stub = os.path.splitext(
                os.path.basename(fluorescence_path)
            )[0]
            file_stub += f"_pos{pos+1}"

            dapi_ch_idx = config["channel_map"].get("DAPI", None)
            if dapi_ch_idx is not None:
                dapi_slice = max_project_fluorescence(
                    nd2obj=fluoro_nd2,
                    ch_index=dapi_ch_idx,
                    position=pos,
                    fluoro_params=config["morphology_params"]["nuclei_params"]
                )
            
                dapi_mask = process_fluorescence_channel(
                    dapi_slice,
                    **config["morphology_params"]["nuclei_params"]
                )
                debug_display_dapi(dapi_slice, dapi_mask, pos)
            else:
                dapi_mask = None

            corrected_cars_slice = max_project_cars(
                nd2obj=cars_nd2,
                c_index=2,
                position=pos,
                reference_image=reference_image,
                foci_params=foci_params
            )

            fluor_images_for_display = {}
            for cm in chosen_cell_markers:
                if cm in fluorescence_path:
                    ch_idx = channel_map.get(cm, None)
                    if ch_idx is not None:
                        z_stack_fl = []
                        for z_idx in range(fluoro_nd2.sizes['z']):
                            raw_slice = np.nan_to_num(
                                fluoro_nd2.get_frame_2D(v=pos, c=ch_idx, z=z_idx)
                            )
                            z_stack_fl.append(raw_slice)
                        fluor_images_for_display[cm] = np.max(
                            np.array(z_stack_fl), axis=0
                        )

            if len(fluor_images_for_display) > 0:
                composite_fluor = composite_fluorescence(fluor_images_for_display, config)
            else:
                composite_fluor = np.zeros(
                    (
                        corrected_cars_slice.shape[0],
                        corrected_cars_slice.shape[1],
                        3
                    ), dtype=np.uint8
                )

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(composite_fluor)
            axs[0].set_title(f"Max Projected Fluorescence Overlay (pos={pos+1})")
            axs[0].axis('off')

            axs[1].imshow(corrected_cars_slice, cmap='gray')
            axs[1].set_title(f"Max Projected CARS (pos={pos+1})")
            axs[1].axis('off')
            plt.show()
            plt.close(fig)

            cars_foci_mask = find_foci(corrected_cars_slice, **foci_params)

            for cm in chosen_cell_markers:
                if cm in fluorescence_path:
                    cm_channel_idx = channel_map.get(cm, None)
                    if cm_channel_idx is None:
                        continue
                    cm_slice = max_project_fluorescence(
                        nd2obj=fluoro_nd2,
                        ch_index=cm_channel_idx,
                        position=pos,
                        fluoro_params=fluorescence_params
                    )

                    marker_thresholds = config.get("marker_thresholds", {})
                    cm_thresholds = marker_thresholds.get(cm, {})

                    threshold_method = cm_thresholds.get(
                        "threshold_method",
                        fluorescence_params.get("threshold_method", "otsu")
                    )
                    offset_val = cm_thresholds.get(
                        "offset",
                        fluorescence_params.get("offset", 1.0)
                    )

                    cm_mask = process_fluorescence_channel(
                        cm_slice,
                        cell_size=fluorescence_params["cell_size"],
                        min_size=fluorescence_params["min_size"],
                        closing_radius=fluorescence_params["closing_radius"],
                        gaussian_sigma=fluorescence_params["gaussian_sigma"],
                        fill_holes=fluorescence_params["fill_holes"],
                        threshold_method=threshold_method,
                        offset=offset_val
                    )

                    pos_results, pos_summary = analyze_intracellular_objects(
                        cars_foci_mask,
                        cm_mask,
                        corrected_cars_slice,
                        file_name=os.path.basename(fluorescence_path),
                        z_stack=pos + 1,
                        pixel_size_microns=pixel_size_microns
                    )

                    for r_item in pos_results:
                        r_item["cell_marker"] = cm
                    for s_item in pos_summary:
                        s_item["cell_marker"] = cm

                    all_positions_results.extend(pos_results)
                    all_positions_summary.extend(pos_summary)

                    def create_rgb_mask(bin_mask, rgb_color):
                        """Create an RGB mask from a binary mask."""
                        rgb_m = np.zeros((*bin_mask.shape, 3), dtype=np.uint8)
                        for i_col in range(3):
                            rgb_m[..., i_col] = bin_mask * rgb_color[i_col]
                        return rgb_m

                    green = [0, 255, 0]
                    yellow = [255, 255, 0]

                    cell_rgb_mask = create_rgb_mask(cm_mask, green)
                    cars_rgb_mask = create_rgb_mask(cars_foci_mask, yellow)
                    overlay_rgb_mask = np.clip(
                        0.5 * cell_rgb_mask + 0.5 * cars_rgb_mask,
                        0, 255
                    ).astype(np.uint8)

                    fig_mask, axs_mask = plt.subplots(1, 3, figsize=(18, 6))
                    axs_mask[0].imshow(cell_rgb_mask)
                    axs_mask[0].set_title(f"{cm} Cell Mask (pos={pos+1})")
                    axs_mask[0].axis('off')

                    axs_mask[1].imshow(cars_rgb_mask)
                    axs_mask[1].set_title(f"CARS Mask (pos={pos+1})")
                    axs_mask[1].axis('off')

                    axs_mask[2].imshow(overlay_rgb_mask)
                    axs_mask[2].set_title(f"Overlay (pos={pos+1}) [{cm}]")
                    axs_mask[2].axis('off')
                    plt.show()
                    plt.close(fig_mask)

                    if dapi_mask is not None:
                        images_dir = ensure_subdirectory(
                            config["paths"]["data_directory"], "Images"
                        )
                        out_overlay_path = os.path.join(
                            images_dir,
                            f"{file_stub}_DAPI_{cm}.png"
                        )
                        save_dapi_marker_overlay(
                            dapi_mask, cm_mask, cm, out_overlay_path
                        )

            fluor_images_for_composite = {}
            for marker_name, ch_idx in channel_map.items():
                if ch_idx is None:
                    continue
                z_stack_fl = []
                for z_slice in range(fluoro_nd2.sizes['z']):
                    raw_slice_fluor = np.nan_to_num(
                        fluoro_nd2.get_frame_2D(v=pos, c=ch_idx, z=z_slice)
                    )
                    z_stack_fl.append(raw_slice_fluor)
                marker_max = np.max(np.array(z_stack_fl), axis=0)
                fluor_images_for_composite[marker_name] = marker_max

            save_composite_images(
                fluor_images_for_composite,
                corrected_cars_slice,
                config,
                config["paths"]["data_directory"],
                file_stub
            )

    return all_positions_results, all_positions_summary


def debug_display_dapi(raw_dapi_slice, dapi_mask, pos_index):
    """
    Displays a quick debug figure of the DAPI slice vs. the final DAPI mask.
    Shows how 'liberal' the mask is compared to the raw intensity.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(raw_dapi_slice, cmap='gray')
    axs[0].set_title(f"DAPI (pos={pos_index+1}) - Max Projection")
    axs[0].axis('off')

    # We'll overlay the mask in red
    overlay = np.dstack([raw_dapi_slice]*3).astype(np.float32)
    # Normalize overlay for display
    overlay = rescale_intensity(overlay, in_range='image', out_range=(0, 255)).astype(np.uint8)
    mask_red = np.zeros_like(overlay)
    mask_red[..., 0] = dapi_mask * 255  # fill red channel
    alpha = 0.4

    axs[1].imshow(overlay)
    axs[1].imshow(mask_red, alpha=alpha)
    axs[1].set_title(f"DAPI Mask Overlay (pos={pos_index+1})")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def save_results_to_excel(results, summary, output_file):
    """
    Save analysis results and summary to an Excel file.

    Parameters
    ----------
    results : list of dict
        Detailed results of lipid inclusions for each cell.
    summary : list of dict
        Summary of lipid objects per cell.
    output_file : str
        Output .xlsx file path.
    """
    results_df = pd.DataFrame(results)
    summary_df = pd.DataFrame(summary)

    # If these columns exist, sort by them in order.
    if 'cell_marker' in results_df.columns and 'z_stack' in results_df.columns:
        results_df = results_df.sort_values(by=['file_name', 'cell_marker', 'z_stack'])
    if 'cell_marker' in summary_df.columns and 'z_stack' in summary_df.columns:
        summary_df = summary_df.sort_values(by=['file_name', 'cell_marker', 'z_stack'])

    with pd.ExcelWriter(output_file) as writer:
        results_df.to_excel(writer, sheet_name='Detailed Results', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)


def ensure_subdirectory(main_dir, sub_name="Images"):
    """
    Create (if needed) and return a subdirectory path inside main_dir.

    Parameters
    ----------
    main_dir : str
        Base directory path.
    sub_name : str
        Name for the subdirectory (default "Images").

    Returns
    -------
    out_dir : str
        The full path to the subdirectory.
    """
    out_dir = os.path.join(main_dir, sub_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def grayscale_autoscale(image_2d):
    """
    Rescale a 2D image to 0..255, returning an 8-bit grayscale.

    Parameters
    ----------
    image_2d : ndarray
        2D image.

    Returns
    -------
    scaled_8bit : ndarray (uint8)
        Grayscale image, auto-scaled to [0..255].
    """
    scaled = rescale_intensity(image_2d, in_range='image', out_range=(0, 255))
    return scaled.astype(np.uint8)


def blend_fluorescence_cars(fluor_rgb, cars_gray, alpha=0.5):
    """
    Blend a color fluorescence composite with a grayscale CARS image (HxW).

    Weighted sum: alpha*fluor + (1-alpha)*cars

    Parameters
    ----------
    fluor_rgb : ndarray, shape (H, W, 3)
        8-bit color image of the fluorescence composite.
    cars_gray : ndarray, shape (H, W)
        8-bit grayscale image of the CARS.
    alpha : float
        Blend factor for fluorescence (default 0.5).

    Returns
    -------
    blended : ndarray, shape (H, W, 3) (uint8)
        The blended composite in 8-bit color.
    """
    cars_rgb = np.stack([cars_gray, cars_gray, cars_gray], axis=-1)
    blended_float = (
        alpha * fluor_rgb.astype(np.float32)
        + (1 - alpha) * cars_rgb.astype(np.float32)
    )
    blended = np.clip(blended_float, 0, 255).astype(np.uint8)
    return blended


def composite_fluorescence(fluor_images, config_dict):
    """
    Build an RGB image by colorizing each 2D fluorescence channel and summing.

    Each channel is rescaled to [0..1], then multiplied by a (R,G,B)
    color from config, then added into a float accumulation array.

    Parameters
    ----------
    fluor_images : dict of { str: ndarray }
        e.g. {"DAPI": dapi_2d, "IBA1": iba1_2d, ...}
    config_dict : dict
        A config dictionary with "colormaps" containing marker->(R,G,B) definitions.

    Returns
    -------
    composite_8bit : ndarray, shape (H, W, 3), dtype=uint8
        The color composite.
    """
    first_key = next(iter(fluor_images))
    height, width = fluor_images[first_key].shape
    composite_float = np.zeros((height, width, 3), dtype=np.float32)

    for marker_name, img_2d in fluor_images.items():
        rgb_255 = config_dict["colormaps"].get(
            marker_name, config_dict["colormaps"]["DEFAULT"]
        )
        rgb_float = np.array(rgb_255) / 255.0
        colorized = colorize_channel(img_2d, rgb_float)
        composite_float += colorized.astype(np.float32)

    composite_float = np.clip(composite_float, 0.0, 1.0)
    composite_8bit = (composite_float * 255).astype(np.uint8)
    return composite_8bit


def colorize_channel(image_2d, rgb_color):
    """
    Rescale a single-channel image to [0..1] and colorize it.

    Parameters
    ----------
    image_2d : ndarray
        2D grayscale data.
    rgb_color : tuple of float
        (R, G, B) each in [0..1].

    Returns
    -------
    colorized : ndarray, shape (H, W, 3), dtype=float32
        The colorized channel in [0..1].
    """
    scaled = rescale_intensity(image_2d, in_range='image', out_range=(0, 1))
    colorized = np.stack([
        scaled * rgb_color[0],
        scaled * rgb_color[1],
        scaled * rgb_color[2]
    ], axis=-1)
    return colorized


def save_composite_images(fluor_images, cars_image, config_dict,
                          main_dir, file_stub):
    """
    Save three composite images to disk (PNG):
     1) [file_stub]_fluor.png  -> color composite of all fluorescence channels
     2) [file_stub]_cars.png   -> grayscale CARS channel
     3) [file_stub]_fluor_cars.png -> 50/50 blend of color fluor + grayscale CARS

    Parameters
    ----------
    fluor_images : dict
        Marker -> 2D array.
    cars_image : ndarray
        2D CARS channel data (already reference-corrected).
    config_dict : dict
        The main configuration dict, containing 'colormaps'.
    main_dir : str
        Base directory path.
    file_stub : str
        Filename prefix for the output images.

    Returns
    -------
    None
    """
    out_dir = ensure_subdirectory(main_dir, "Images")

    composite_fluor = composite_fluorescence(fluor_images, config_dict)
    cars_gray_8bit = grayscale_autoscale(cars_image)
    fluor_cars_blended = blend_fluorescence_cars(
        composite_fluor, cars_gray_8bit, alpha=0.5
    )

    fluor_bgr = cv2.cvtColor(composite_fluor, cv2.COLOR_RGB2BGR)
    blend_bgr = cv2.cvtColor(fluor_cars_blended, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(out_dir, f"{file_stub}_fluor.png"), fluor_bgr)
    cv2.imwrite(os.path.join(out_dir, f"{file_stub}_cars.png"), cars_gray_8bit)
    cv2.imwrite(os.path.join(out_dir, f"{file_stub}_fluor_cars.png"), blend_bgr)

    print(f"Saved composites to {out_dir}")


def save_dapi_marker_overlay(dapi_mask, marker_mask, marker_name, out_path):
    """
    Create a PNG with RGBA where:
      - Background is transparent
      - The DAPI region is semi-transparent (using config's DAPI color)
      - The marker_mask outline is fully opaque in the marker's config color

    Parameters
    ----------
    dapi_mask : ndarray (bool)
        Binary mask for the DAPI channel (True => region).
    marker_mask : ndarray (bool)
        Binary mask for the cell marker channel.
    marker_name : str
        Name of the cell marker, e.g. 'IBA1', 'GFAP'. This is used to fetch
        the color from config.
    out_path : str
        Output .png file path to save the overlay image.

    Returns
    -------
    None
        Saves a .png file with RGBA channels.
    """
    height, width = dapi_mask.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    dapi_color_255 = config["colormaps"].get("DAPI", (0, 0, 255))
    dapi_rgba = (
        dapi_color_255[0],
        dapi_color_255[1],
        dapi_color_255[2],
        128
    )

    marker_color_255 = config["colormaps"].get(
        marker_name, config["colormaps"]["DEFAULT"]
    )
    marker_rgba = (
        marker_color_255[0],
        marker_color_255[1],
        marker_color_255[2],
        255
    )

    outline = find_boundaries(marker_mask, mode='outer')
    rgba[dapi_mask, 0] = dapi_rgba[0]
    rgba[dapi_mask, 1] = dapi_rgba[1]
    rgba[dapi_mask, 2] = dapi_rgba[2]
    rgba[dapi_mask, 3] = dapi_rgba[3]

    outline_positions = (outline == 1)
    rgba[outline_positions, 0] = marker_rgba[0]
    rgba[outline_positions, 1] = marker_rgba[1]
    rgba[outline_positions, 2] = marker_rgba[2]
    rgba[outline_positions, 3] = marker_rgba[3]

    pil_img = Image.fromarray(rgba, mode='RGBA')
    pil_img.save(out_path)
    print(f"Saved overlay to: {out_path}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", required=True,
    #                     help="Path to a .py file containing `config` dict.")
    # args = parser.parse_args()

    # config = load_config(args.config)
    
    from config_AD4d import config
    
    DIRECTORY = config["paths"]["data_directory"]
    reference_file = os.path.join(DIRECTORY, 'Reference.nd2')
    reference_output_path = os.path.join(DIRECTORY, 'Reference.tif')
    output_file = os.path.join(DIRECTORY, 'analysis_results.xlsx')

    # 1) Generate reference image
    reference_image = generate_reference_image(
        reference_file,
        reference_output_path,
        blur_radius_microns=2
    )

    # 2) Process paired files
    paired_files, hyperspectral_folders = find_nd2_files(DIRECTORY)
    all_results_list = []
    all_summary_list = []

    for key_val, paths_dict in paired_files.items():
        print(f"Processing pair: {key_val}")
        pair_res, pair_sum = process_nd2_pair(
            paths_dict['fluorescence'],
            paths_dict['CARS'],
            reference_image
        )
        all_results_list.extend(pair_res)
        all_summary_list.extend(pair_sum)

    # 3) Process hyperspectral series
    hyperspectral_foci_params = config["morphology_params"]["foci_params_hyperspectral"]
    for folder in hyperspectral_folders:
        folder_name = os.path.basename(folder)
        hyperspectral_output = os.path.join(
            DIRECTORY,
            f"Hyperspectral_Results_{folder_name}.xlsx"
        )
        process_hyperspectral_series(
            folder,
            reference_image,
            hyperspectral_output,
            hyperspectral_foci_params
        )

    # 4) Save results to Excel
    save_results_to_excel(all_results_list, all_summary_list, output_file)
    print(f"Results saved to {output_file}")
