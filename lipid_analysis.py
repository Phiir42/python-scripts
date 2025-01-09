"""
lipid_analysis.py

This module provides a workflow for analyzing lipid inclusions in microscopy images using:
1) Fluorescence .nd2 files
2) CARS (Coherent Anti-Stokes Raman Scattering) .nd2 files
3) Optional hyperspectral series folders named "SpectrumCH..."

Main steps:
-----------
1. Reference image generation:
   - A reference ND2 file is opened and used to create a normalized reference TIFF.

2. File pairing and offset logic:
   - The script identifies fluorescence vs. CARS .nd2 files (based on config-defined keywords).
   - Each file is given a "StacksX" key, with optional marker-based offsets for fluorescence,
     ensuring correct pairing with corresponding CARS images.

3. Image processing:
   - Fluorescence images are processed to generate a binary cell mask.
   - CARS images are processed to identify lipid droplets (foci).
   - The pipeline performs measurements of lipid inclusions (size, intensity) within each cell.

4. Hyperspectral analysis (if applicable):
   - Folders containing "SpectrumCH" data are detected, and each ND2 in the folder is processed 
     to build a series of corrected images and measure lipid intensities across different
     wavenumbers.

5. Results output:
   - The script saves a final Excel file containing detailed measurements for each cell (lipid
     objects, intensities) and a summary table.

Usage:
------
Execute LipidAnalysis.py from within an environment where `nd2reader`, `scipy`, `pandas`,
`skimage`, and other dependencies are installed. The script reads paths and parameters from a 
configuration file (e.g., `config_ovarianTissue.py`), allowing easy adaptation for different tissue 
types or markers.

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

Note:
-----
The script assumes that all relevant .nd2 files follow a naming convention containing the substring 
"StacksX" and a magnification keyword like "100X". The config file governs how marker offsets 
are applied, and how hyperspectral data is processed. Adjust the config file as needed for 
other tissue systems or filename patterns.
"""

import logging
import os
import re


import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from nd2reader import ND2Reader
from scipy import ndimage as ndi
from skimage import feature, measure, segmentation
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import (
    closing, disk, opening, remove_small_objects
)
from tifffile import imsave
from config_ovarianTissue import config


logging.getLogger('nd2reader').setLevel(logging.ERROR)


def find_nd2_files(directory):
    """
    Find and pair fluorescence and CARS .nd2 files in a directory based on matching keys.

    The key is derived from the filename substring "StacksX". Fluorescence files may
    have an offset (as determined by their marker) that modifies the stacks number so
    that they align with the matching CARS file.

    Also identifies folders containing hyperspectral data, using the keyword defined in
    config["file_keywords"]["hyperspectral_keyword"]. Any directory name matching that
    keyword (e.g., "SpectrumCH") is assumed to contain hyperspectral .nd2 series.

    Parameters
    ----------
    directory : str
        Path to the directory containing the .nd2 files.

    Returns
    -------
    paired_files : dict
        Mapping from the final pairing key (e.g. "Stacks3") to a dict with two fields:
        'fluorescence' and 'CARS', each containing the file path to the respective ND2.
    hyperspectral_folders : list
        List of folder paths that contain the specified hyperspectral keyword.

    Raises
    ------
    ValueError
        If no valid 'StacksX' substring is found when parsing certain filenames.
    """
    fluorescence_files = {}
    cars_files = {}
    hyperspectral_folders = []
    
    hyperspectral_keyword = config["file_keywords"]["hyperspectral_keyword"]

    for file in os.listdir(directory):
        full_path = os.path.join(directory, file)

        # Identify hyperspectral folders
        if hyperspectral_keyword in file and os.path.isdir(full_path):
            hyperspectral_folders.append(full_path)

        # Only process .nd2 files that contain the desired magnification keyword
        if file.endswith('.nd2') and config["file_keywords"]["magnification_keyword"] in file:
            key = get_file_key(file)

            # Check if it's a CARS vs. fluorescence file
            if config["file_keywords"]["cars_keyword"] in file:
                cars_files[key] = full_path
            elif any(marker in file for marker in config["file_keywords"]["fluorescence_markers"]):
                fluorescence_files[key] = full_path

    # Now pair them up
    paired_files = {}
    for key, fluoro_path in fluorescence_files.items():
        if key in cars_files:
            paired_files[key] = {
                'fluorescence': fluoro_path,
                'CARS': cars_files[key]
            }


    return paired_files, hyperspectral_folders


def get_file_key(filename):
    """
    Construct a pairing key based on "StacksX" in the filename, applying marker offsets.

    The function extracts the integer from the substring "StacksX". If the file is
    identified as a fluorescence file (i.e., it does not contain the config's `cars_keyword`),
    it checks which marker is present. It then applies any marker-specific offset
    as specified in `config["stack_offset"]`. Finally, it returns a string of the form
    "Stacks<number>" that is used to match fluorescence and CARS files.

    Parameters
    ----------
    filename : str
        The ND2 filename (e.g., "Sample-Ki67-Stacks1.nd2").

    Returns
    -------
    str
        A standardized key string of the form "Stacks<number>", e.g. "Stacks3".

    Raises
    ------
    ValueError
        If no 'StacksX' substring is found in the filename (under the assumption that
        all relevant files should contain it).
    """
    base = filename.replace('.nd2', '')

    # Determine if this is a CARS file (nonlinear) or fluorescence
    is_cars = config["file_keywords"]["cars_keyword"] in base

    # Extract the integer from "Stacks(\d+)"
    match = re.search(r"Stacks(\d+)", base)
    if not match:
        raise ValueError(f"No valid 'StacksX' found in filename: {filename}")
    stack_num = int(match.group(1))

    # If it's not CARS, sum offsets for all fluorescence markers present
    if not is_cars:
        offset_total = 0
        for marker in config["file_keywords"]["fluorescence_markers"]:
            if marker in base:
                offset_total += config["stack_offset"].get(marker, 0)
        stack_num += offset_total

    # Final key: "Stacks3", for example
    final_key = f"Stacks{stack_num}"
    return final_key


def create_custom_colormap(start_color, end_color):
    """
    Create a linear segmented colormap transitioning from black to the specified color.

    Parameters
    ----------
    start_color : tuple
        RGB color for the starting point (usually black, e.g., (0, 0, 0)).
    end_color : tuple
        RGB color for the ending point (e.g., green (0, 255, 0) or magenta (255, 0, 255)).

    Returns
    -------
    LinearSegmentedColormap
        A custom colormap.
    """
    colors = [
        tuple(c / 255.0 for c in start_color),
        tuple(c / 255.0 for c in end_color)
    ]
    return LinearSegmentedColormap.from_list("custom_colormap", colors)


def process_fluorescence_channel(
    image_slice, min_size, closing_radius, gaussian_sigma, fill_holes
):
    """
    Process a single fluorescence image slice and create a binary mask.

    Parameters
    ----------
    image_slice : ndarray
        2D fluorescence image data.
    min_size : int
        Minimum size (in pixels) for objects to be kept in the mask.
    closing_radius : int
        Radius for the morphological closing operation.
    gaussian_sigma : float
        Sigma for the Gaussian filter.
    fill_holes : bool
        Whether to fill holes in the binary mask.

    Returns
    -------
    cleaned_mask : ndarray
        Binary mask after thresholding and morphological operations.
    """
    if image_slice.ndim != 2:
        raise ValueError(
            f"Expected a 2D array, but got shape {image_slice.shape}"
        )

    # Replace NaNs with zeros
    image_slice = np.nan_to_num(image_slice)

    # Apply Gaussian smoothing
    smoothed_image = gaussian(
        image_slice, sigma=gaussian_sigma, preserve_range=True
    )

    # Apply Otsu's threshold to generate binary mask
    threshold_value = threshold_otsu(smoothed_image)
    binary_mask = smoothed_image > threshold_value

    # Perform morphological closing to connect structures
    binary_closed = closing(binary_mask, disk(closing_radius))

    if fill_holes:
        binary_closed = ndi.binary_fill_holes(binary_closed)

    # Remove small objects to reduce background noise
    cleaned_mask = remove_small_objects(binary_closed, min_size=min_size)

    return cleaned_mask


def find_foci(image_slice, sigma, min_distance, min_size, std_dev_multiplier):
    """
    Identify foci within a CARS image slice using local maxima and thresholding.

    Parameters
    ----------
    image_slice : ndarray
        2D CARS image data.
    sigma : float
        Sigma for the Gaussian filter.
    min_distance : int
        Minimum distance between local maxima.
    min_size : int
        Minimum area (in pixels) for each focus to be retained.
    std_dev_multiplier : float
        Multiplier for the standard deviation to set a threshold.

    Returns
    -------
    final_mask : ndarray
        Binary mask for the identified foci.
    """
    image_slice = np.nan_to_num(image_slice)

    # Smooth the image
    data = {}
    data['smoothed'] = gaussian(image_slice, sigma=sigma, preserve_range=True)

    # Threshold using mean and standard deviation
    mean_val = np.mean(data['smoothed'])
    std_val = np.std(data['smoothed'])
    threshold_val = mean_val + (std_dev_multiplier * std_val)
    data['mask_std'] = data['smoothed'] > threshold_val

    # Apply global Otsu threshold
    global_thresh = threshold_otsu(data['smoothed'])
    data['mask_otsu'] = data['smoothed'] > global_thresh

    # Combine the two masks
    data['combined'] = data['mask_std'] & data['mask_otsu']

    # Morphologically open to clean up small noise
    data['opened'] = opening(data['combined'], disk(3))

    # Find local maxima
    distance = ndi.distance_transform_edt(data['opened'])
    local_maxi_coords = feature.peak_local_max(
        data['smoothed'],
        min_distance=min_distance,
        labels=data['opened']
    )
    local_maxi = np.zeros_like(data['opened'], dtype=bool)
    local_maxi[tuple(local_maxi_coords.T)] = True

    # Perform watershed segmentation
    markers = ndi.label(local_maxi)[0]
    labels = segmentation.watershed(-distance, markers, mask=data['opened'])

    # Filter by area to create the final mask
    final_mask = np.zeros_like(labels, dtype=bool)
    for region in measure.regionprops(labels):
        if region.area >= min_size:
            final_mask[tuple(region.coords.T)] = True

    return final_mask


def process_hyperspectral_series(
    spectrum_folder, reference_image, output_path, foci_params
):
    """
    Process a hyperspectral series to extract lipid droplet intensities.

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
        Saves the results to a spreadsheet.
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
    for file in nd2_files:
        with ND2Reader(file) as nd2:
            # Assuming channel 2 for CARS
            raw_image = np.nan_to_num(nd2.get_frame_2D(c=2))
            corrected_image = raw_image / reference_image
            corrected_image = np.nan_to_num(corrected_image)
            corrected_images.append(corrected_image)

    # Generate the lipid mask using the 9th corrected image
    mask_image = corrected_images[8]  # 9th image corresponds to index 8
    lipid_mask = find_foci(mask_image, **foci_params)
    
    # (Optional) Visualize the mask overlay on the 9th corrected image
    visualize_hyperspectral_mask_overlay(mask_image, lipid_mask)

    # Extract intensities for each lipid droplet across all 32 corrected images
    lipid_labels = measure.label(lipid_mask)
    lipid_data = []

    for lipid in measure.regionprops(lipid_labels):
        lipid_id = lipid.label
        intensities = []
        for img in corrected_images:
            mean_intensity = np.mean(
                img[lipid.coords[:, 0], lipid.coords[:, 1]])
            intensities.append(mean_intensity)
        lipid_data.append([lipid_id] + intensities)

    columns = ["Lipid ID"] + [f"Wavenumber {i+1}" for i in range(32)]
    lipid_df = pd.DataFrame(lipid_data, columns=columns)
    lipid_df.to_excel(output_path, index=False)
    print(f"Hyperspectral lipid intensities saved to {output_path}")
    
    
def visualize_hyperspectral_mask_overlay(cars_image, lipid_mask):
    """
    Visualize the overlay of the CARS image and lipid mask from the hyperspectral series.

    Parameters
    ----------
    cars_image : ndarray
        2D CARS image data (already corrected, e.g., the 9th hyperspectral image).
    lipid_mask : ndarray
        Binary mask for the identified lipid droplets.

    Returns
    -------
    None
        Displays matplotlib figures showing the grayscale CARS image, the lipid mask, and an overlay.
    """
    # Helper function to create an RGB mask
    def create_rgb_mask(mask, color):
        """
        Create an RGB image from a boolean mask, painting 'color' where mask is True.
        """
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for i in range(3):
            rgb[..., i] = mask * color[i]
        return rgb

    # Convert the CARS image to 8-bit grayscale for display
    # (Just for visualization; scale by max intensity)
    max_val = cars_image.max() if cars_image.max() > 0 else 1
    grayscale_8bit = (cars_image / max_val * 255).astype(np.uint8)

    # Create a yellow mask for lipids
    lipid_mask_rgb = create_rgb_mask(lipid_mask, [255, 255, 0])  # yellow

    # Blend the grayscale image and the mask with 50% opacity
    grayscale_rgb = np.stack([grayscale_8bit]*3, axis=-1)  # shape => (H, W, 3)
    overlay_rgb = np.clip(
        0.5 * grayscale_rgb + 0.5 * lipid_mask_rgb, 0, 255
    ).astype(np.uint8)

    # Display the three images side by side
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


def analyze_intracellular_objects(
    cars_mask, cell_mask, cars_image, file_name, z_stack, pixel_size_microns
):
    """
    Analyze lipid inclusions within each cell in the image.

    Parameters
    ----------
    cars_mask : ndarray
        Mask for CARS lipid droplets.
    cell_mask : ndarray
        Mask for cells.
    cars_image : ndarray
        Original CARS image.
    file_name : str
        Name of the .nd2 file being processed.
    z_stack : int
        Current z-stack being analyzed.
    pixel_size_microns : float
        Pixel size in microns.

    Returns
    -------
    results : list of dict
        List of dictionaries containing lipid properties with file and z-stack context.
    summary : list of dict
        List of dictionaries summarizing lipid objects per cell.
    """
    labeled_cells = measure.label(cell_mask)
    labeled_lipids = measure.label(cars_mask)
    results = []
    summary = []

    for cell in measure.regionprops(labeled_cells):
        cell_id = cell.label
        cell_area = cell.area
        cell_mask_region = labeled_cells == cell_id
        lipid_objects_in_cell = labeled_lipids * cell_mask_region

        lipid_count = 0
        total_lipid_area = 0

        # Calculate properties of each lipid object within the cell
        for lipid in measure.regionprops(
            lipid_objects_in_cell,
            intensity_image=cars_image
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
                'lipid_size_pixels': lipid_size_pixels,
                'lipid_size_um2': lipid_size_um2,
                'lipid_intensity': lipid.mean_intensity
            })

        summary.append({
            'file_name': file_name,
            'z_stack': z_stack,
            'cell_id': cell_id,
            'cell_area': cell_area,
            'lipid_count': lipid_count,
            'total_lipid_area_um2': total_lipid_area
        })

    return results, summary


def generate_reference_image(reference_file, output_path, blur_radius_microns):
    """
    Generate a normalized reference image from the CARS signal in a reference ND2 file.

    Parameters
    ----------
    reference_file : str
        Path to the reference .nd2 file.
    output_path : str
        Path to save the normalized reference image.
    blur_radius_microns : float
        Desired Gaussian blur radius in microns.

    Returns
    -------
    reference_image : ndarray
        The normalized reference image (2D array scaled from 0 to 1).
    """
    with ND2Reader(reference_file) as ref_nd2:
        print(f"Generating reference image from: {reference_file}")

        # Ensure we're working with the CARS channel (assumed to be channel 2)
        reference_image = np.nan_to_num(ref_nd2.get_frame_2D(c=2))
        pixel_size_microns = ref_nd2.metadata['pixel_microns']

    sigma_pixels = blur_radius_microns / pixel_size_microns
    print(
        f"Applying Gaussian blur with sigma = {sigma_pixels} pixels "
        f"(from {blur_radius_microns} microns)"
    )

    # Apply Gaussian blur
    blurred_reference = gaussian(reference_image, sigma=sigma_pixels,
                                 preserve_range=True)

    # Scale the blurred image to preserve the original range
    blurred_ref_max = np.max(blurred_reference)
    original_max = np.max(reference_image)

    if blurred_ref_max > 0:
        blurred_reference_scaled = (
            blurred_reference * (original_max / blurred_ref_max)
        )
    else:
        raise ValueError("Blurred reference has no valid data.")

    max_value = np.max(blurred_reference_scaled)
    if max_value > 0:
        normalized_reference = blurred_reference_scaled / max_value
    else:
        raise ValueError("Reference image has no valid intensity data after "
                         "preprocessing.")

    imsave(output_path, (normalized_reference * 255).astype(np.uint8))
    print(f"Reference image saved to: {output_path}")

    return normalized_reference


def get_marker_color(marker_name, config_colors):
    """
    Returns a custom colormap for the given marker name.
    If marker is not in config, returns a default colormap.
    """
    if marker_name in config_colors:
        end_color = config_colors[marker_name]
    else:
        end_color = config_colors["DEFAULT"]

    return create_custom_colormap((0, 0, 0), end_color)


def process_nd2_pair(fluorescence_path, cars_path, reference_image):
    """
    Process paired fluorescence and CARS .nd2 files with reference correction and a flexible
    cell mask.

    The function performs the following steps:
    
    1) Identifies which fluorescence marker is in the filename (the "analysis marker") so we know
       which channel index to analyze for fluorescence-based insights.
    2) Determines which marker(s), if any, are designated as cell markers
       (using config["cell_markers"]).
       - If no cell markers are designated (or none found in the filename), we treat the entire
         field of view as a single "cell."
       - Otherwise, we read each cell-marker channel, generate a binary mask, and combine (union)
         those masks into a single cell_mask.
    3) Reads and corrects the CARS channel using the provided reference_image.
    4) Locates lipid inclusions (via 'find_foci') and then calls 'analyze_intracellular_objects'
       to measure lipid properties within each cell-defined region.
    5) Returns a detailed list of measurements (results) and a summary of lipid objects per cell.

    Parameters
    ----------
    fluorescence_path : str
        Path to the fluorescence .nd2 file.
    cars_path : str
        Path to the CARS .nd2 file.
    reference_image : ndarray
        Precomputed reference image for CARS correction.

    Returns
    -------
    results : list of dict
        Detailed results of lipid inclusions for each cell.
    summary : list of dict
        Summary of lipid objects per cell.

    Notes
    -----
    - If multiple designated cell markers exist in the same file (e.g., "DAPI" and "aSMA"),
      this function will generate a union mask of all cell-marker channels present.
    - If none of the designated cell markers appears in the filename, or if config["cell_markers"]
      is empty, the entire image is treated as one cell.
    """

    # Retrieve morphology parameters from config
    fluorescence_params = config["morphology_params"]["fluorescence_params"]
    foci_params = config["morphology_params"]["foci_params"]

    # Identify which marker is in this filename (the "analysis marker" for general fluorescence)
    analysis_marker_hit = None
    for marker in config["file_keywords"]["fluorescence_markers"]:
        if marker in fluorescence_path:
            analysis_marker_hit = marker
            break

    # If we found no recognized marker, skip
    if analysis_marker_hit is None:
        print(f"No recognized marker in {fluorescence_path}")
        return [], []

    # Determine the channel index for the analysis marker
    channel_map = config["channel_map"]
    analysis_channel_index = channel_map.get(analysis_marker_hit, 0)

    # Figure out which markers are designated as cell markers
    cell_markers = config.get("cell_markers", [])  # e.g. ["DAPI"]
    # It's possible this list is empty, meaning "no designated cell marker"

    # Open both ND2 files
    with ND2Reader(fluorescence_path) as fluoro_nd2, ND2Reader(cars_path) as cars_nd2:
        print(f"File: {fluorescence_path}")

        # For multi-position ND2, we iterate over 'v'
        fluoro_nd2.iter_axes = 'v'
        cars_nd2.iter_axes = 'v'

        # Retrieve pixel size from metadata (assume same for both files)
        pixel_size_microns = fluoro_nd2.metadata['pixel_microns']

        # Prepare data structures to accumulate results
        all_positions_results = []
        all_positions_summary = []

        # Iterate over positions (v)
        for pos in range(fluoro_nd2.sizes['v']):
            fluoro_nd2.default_coords['v'] = pos
            cars_nd2.default_coords['v'] = pos

            # Helper function to max-project the specified channel
            def max_project_channel(nd2obj, channel_index, position):
                """Returns the max-projected stack for the specified channel index."""
                z_stack_slices = [
                    np.nan_to_num(nd2obj.get_frame_2D(v=position, c=channel_index, z=z_slice))
                    for z_slice in range(nd2obj.sizes['z'])
                ]
                return np.max(np.array(z_stack_slices), axis=0)

            # 1) Read the analysis-marker channel (for optional display or additional processing)
            fluorescence_slice = max_project_channel(fluoro_nd2, analysis_channel_index, pos)

            # 2) Construct the cell_mask
            if not cell_markers:
                # If no cell markers are designated => entire field is a "cell"
                cell_mask = np.ones_like(fluorescence_slice, dtype=bool)
            else:
                # Union of all designated cell markers that appear in the file name
                cell_mask = np.zeros_like(fluorescence_slice, dtype=bool)
                for cm in cell_markers:
                    if cm in fluorescence_path:
                        cm_channel_idx = channel_map.get(cm, None)
                        if cm_channel_idx is not None:
                            cm_slice = max_project_channel(fluoro_nd2, cm_channel_idx)
                            cm_mask = process_fluorescence_channel(cm_slice, **fluorescence_params)
                            # Combine them with logical OR
                            cell_mask |= cm_mask

                # If cell_mask is still all False (e.g. because the markers in cell_markers
                # weren't found), treat entire FOV as single cell
                if not cell_mask.any():
                    cell_mask = np.ones_like(fluorescence_slice, dtype=bool)

            # 3) Read & correct the CARS channel
            cars_slice = max_project_channel(cars_nd2, 2, pos)  # channel 2 for CARS
            corrected_cars_slice = np.nan_to_num(cars_slice / reference_image)

            # 4) Display max-projected slices (optional)
            marker_colormap = get_marker_color(analysis_marker_hit, config["colormaps"])
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(fluorescence_slice, cmap=marker_colormap)
            axs[0].set_title(f"Max Projected Fluorescence (pos={pos+1})")
            axs[0].axis('off')

            axs[1].imshow(corrected_cars_slice, cmap='gray')
            axs[1].set_title(f"Max Projected CARS (pos={pos+1})")
            axs[1].axis('off')
            plt.show()

            # 5) Identify lipid droplets using the foci-finding function
            cars_foci_mask = find_foci(corrected_cars_slice, **foci_params)

            # 6) Analyze lipid objects within the cell_mask
            pos_results, pos_summary = analyze_intracellular_objects(
                cars_foci_mask,
                cell_mask,
                corrected_cars_slice,
                file_name=os.path.basename(fluorescence_path),
                z_stack=pos + 1,
                pixel_size_microns=pixel_size_microns
            )

            all_positions_results.extend(pos_results)
            all_positions_summary.extend(pos_summary)

            # 7) (Optional) Visualization of the final masks
            def create_rgb_mask(mask, rgb_color):
                rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
                for i in range(3):
                    rgb_mask[:, :, i] = mask * rgb_color[i]
                return rgb_mask

            green = [0, 255, 0]   # for cell mask
            yellow = [255, 255, 0]  # for CARS mask

            cell_rgb_mask = create_rgb_mask(cell_mask, green)
            cars_rgb_mask = create_rgb_mask(cars_foci_mask, yellow)

            # Blend with 50% opacity
            overlay_rgb_mask = np.clip(
                0.5 * cell_rgb_mask + 0.5 * cars_rgb_mask,
                0, 255
            ).astype(np.uint8)

            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            axs[0].imshow(cell_rgb_mask)
            axs[0].set_title(f"Cell Mask (Green) pos={pos+1}")
            axs[0].axis('off')

            axs[1].imshow(cars_rgb_mask)
            axs[1].set_title(f"CARS Mask (Yellow) pos={pos+1}")
            axs[1].axis('off')

            axs[2].imshow(overlay_rgb_mask)
            axs[2].set_title(f"Overlay of Cell & CARS pos={pos+1}")
            axs[2].axis('off')
            plt.show()

    # Return the compiled results for all positions
    return all_positions_results, all_positions_summary



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

    with pd.ExcelWriter(output_file) as writer:
        results_df.to_excel(writer, sheet_name='Detailed Results', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)


# Main Workflow
if __name__ == "__main__":
    DIRECTORY = config["paths"]["data_directory"]
    reference_file = os.path.join(DIRECTORY, 'Reference.nd2')
    reference_output_path = os.path.join(DIRECTORY, 'Reference.tif')
    output_file = os.path.join(DIRECTORY, 'analysis_results.xlsx')

    # Generate reference image
    reference_image = generate_reference_image(
        reference_file,
        reference_output_path,
        blur_radius_microns=2
    )

    # Process paired files with reference correction
    paired_files, hyperspectral_folders = find_nd2_files(DIRECTORY)
    all_results = []
    all_summary = []

    for key, files in paired_files.items():
        print(f"Processing pair: {key}")
        pair_results, pair_summary = process_nd2_pair(
            files['fluorescence'],
            files['CARS'],
            reference_image
        )
        all_results.extend(pair_results)
        all_summary.extend(pair_summary)

    # Process hyperspectral series
    foci_params = config["morphology_params"]["foci_params"]
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
            foci_params
        )

    # Save results to Excel
    save_results_to_excel(all_results, all_summary, output_file)
    print(f"Results saved to {output_file}")
