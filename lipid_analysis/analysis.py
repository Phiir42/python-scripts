import os
import re

import matplotlib.pyplot as plt
import numpy as np
from nd2reader import ND2Reader
from skimage import measure
from skimage.morphology import dilation, disk

from .config_utils import resolve_marker_name
from .constants import CARS_CH, VERBOSE
from .filters import apply_east_shadows_filter
from .imaging import composite_fluorescence
from .io_utils import ensure_subdirectory, save_composite_images
from .segmentation import find_foci, process_fluorescence_channel
from .visualize import debug_display_3way_segmentation, debug_display_dapi


def max_project_fluorescence(nd2obj, ch_index, position, fluoro_params):
    """Max-project a fluorescence channel at a given position (v index)."""
    from skimage.filters import gaussian

    z_stack_slices = []
    total_z = nd2obj.sizes.get("z", 1)
    gaussian_sigma = float(fluoro_params.get("gaussian_sigma", 0.0) or 0.0)
    for z_slice in range(total_z):
        raw_slice = nd2obj.get_frame_2D(v=position, c=ch_index, z=z_slice)
        raw_slice = np.nan_to_num(raw_slice)
        if gaussian_sigma > 0:
            z_stack_slices.append(
                gaussian(raw_slice, sigma=gaussian_sigma, preserve_range=True)
            )
        else:
            z_stack_slices.append(raw_slice)
    return np.max(np.array(z_stack_slices), axis=0)


def analyze_3way_intracellular_objects(
    labeled_pure_lipid,
    labeled_lipo_lipid,
    labeled_pure_lipo,
    cell_mask,
    cars_image,
    auto_image,
    file_name,
    z_stack,
    pixel_size_microns,
    lamp2_mask=None,
):
    """Quantify pure lipid, lipid+lipofuscin, and pure lipofuscin objects per cell."""
    labeled_cells = measure.label(cell_mask)
    results, summary = [], []

    for cell in measure.regionprops(labeled_cells):
        cell_id = cell.label
        cell_area = cell.area
        cell_area_um2 = cell_area * (pixel_size_microns**2)
        cell_mask_region = labeled_cells == cell_id
        pure_lipid_lamp2_count = lipid_lipo_lamp2_count = pure_lipo_lamp2_count = 0

        def _overlaps_lamp2(rr_cc_coords):
            if lamp2_mask is None:
                return False
            rr, cc = rr_cc_coords[:, 0], rr_cc_coords[:, 1]
            return bool(np.any(lamp2_mask[rr, cc]))

        # A) pure lipid
        pure_lipid_in_cell = labeled_pure_lipid * cell_mask_region
        pure_lipid_count = 0
        pure_lipid_area_um2 = 0.0
        for region_lipid in measure.regionprops(
            pure_lipid_in_cell, intensity_image=cars_image
        ):
            area_px = region_lipid.area
            area_um2 = area_px * (pixel_size_microns**2)
            pure_lipid_count += 1
            pure_lipid_area_um2 += area_um2
            lamp2_hit = _overlaps_lamp2(region_lipid.coords)
            if lamp2_hit:
                pure_lipid_lamp2_count += 1
            results.append(
                {
                    "file_name": file_name,
                    "z_stack": z_stack,
                    "cell_id": cell_id,
                    "cell_area": cell_area,
                    "cell_area_um2": cell_area_um2,
                    "feature_type": "pure_lipid",
                    "feature_size_pixels": area_px,
                    "feature_size_um2": area_um2,
                    "feature_intensity": region_lipid.mean_intensity,
                    "lamp2_colocalized": lamp2_hit,
                }
            )

        # B) lipid+lipofuscin
        lipo_lipid_in_cell = labeled_lipo_lipid * cell_mask_region
        lipid_lipo_count = 0
        lipid_lipo_area_um2 = 0.0
        for region_ll in measure.regionprops(
            lipo_lipid_in_cell, intensity_image=cars_image
        ):
            area_px = region_ll.area
            area_um2 = area_px * (pixel_size_microns**2)
            lipid_lipo_count += 1
            lipid_lipo_area_um2 += area_um2
            lamp2_hit = _overlaps_lamp2(region_ll.coords)
            if lamp2_hit:
                lipid_lipo_lamp2_count += 1
            results.append(
                {
                    "file_name": file_name,
                    "z_stack": z_stack,
                    "cell_id": cell_id,
                    "cell_area": cell_area,
                    "cell_area_um2": cell_area_um2,
                    "feature_type": "lipid_lipofuscin",
                    "feature_size_pixels": area_px,
                    "feature_size_um2": area_um2,
                    "feature_intensity": region_ll.mean_intensity,
                    "lamp2_colocalized": lamp2_hit,
                }
            )

        # C) pure lipofuscin
        pure_lipo_in_cell = labeled_pure_lipo * cell_mask_region
        pure_lipo_count = 0
        pure_lipo_area_um2 = 0.0
        for region_pure_l in measure.regionprops(
            pure_lipo_in_cell, intensity_image=auto_image
        ):
            area_px = region_pure_l.area
            area_um2 = area_px * (pixel_size_microns**2)
            pure_lipo_count += 1
            pure_lipo_area_um2 += area_um2
            lamp2_hit = _overlaps_lamp2(region_pure_l.coords)
            if lamp2_hit:
                pure_lipo_lamp2_count += 1
            results.append(
                {
                    "file_name": file_name,
                    "z_stack": z_stack,
                    "cell_id": cell_id,
                    "cell_area": cell_area,
                    "cell_area_um2": cell_area_um2,
                    "feature_type": "pure_lipofuscin",
                    "feature_size_pixels": area_px,
                    "feature_size_um2": area_um2,
                    "feature_intensity": region_pure_l.mean_intensity,
                    "lamp2_colocalized": lamp2_hit,
                }
            )

        pure_lipid_pct = (
            100.0 * (pure_lipid_area_um2 / cell_area_um2) if cell_area_um2 > 0 else 0
        )
        lipid_lipo_pct = (
            100.0 * (lipid_lipo_area_um2 / cell_area_um2) if cell_area_um2 > 0 else 0
        )
        pure_lipo_pct = (
            100.0 * (pure_lipo_area_um2 / cell_area_um2) if cell_area_um2 > 0 else 0
        )

        summary.append(
            {
                "file_name": file_name,
                "z_stack": z_stack,
                "cell_id": cell_id,
                "cell_area": cell_area,
                "cell_area_um2": cell_area_um2,
                "pure_lipid_count": pure_lipid_count,
                "pure_lipid_area_um2": pure_lipid_area_um2,
                "pure_lipid_percentage": pure_lipid_pct,
                "lipid_lipofuscin_count": lipid_lipo_count,
                "lipid_lipofuscin_area_um2": lipid_lipo_area_um2,
                "lipid_lipofuscin_percentage": lipid_lipo_pct,
                "lipofuscin_count": pure_lipo_count,
                "lipofuscin_area_um2": pure_lipo_area_um2,
                "lipofuscin_percentage": pure_lipo_pct,
                "pure_lipid_lamp2_count": pure_lipid_lamp2_count,
                "lipid_lipofuscin_lamp2_count": lipid_lipo_lamp2_count,
                "lipofuscin_lamp2_count": pure_lipo_lamp2_count,
            }
        )
    return results, summary


def process_nd2_pair(fluorescence_path, cars_path, reference_image):
    """Process a fluorescence/CARS ND2 pair; return (results, summary)."""
    global config
    foci_params = config["morphology_params"]["foci_params"]
    fluorescence_params = config["morphology_params"]["fluorescence_params"]
    autofluorescence_params = config["morphology_params"]["autofluorescence_params"]

    analysis_marker_hit = None
    for test_marker in config["file_keywords"]["fluorescence_markers"]:
        if test_marker in fluorescence_path:
            analysis_marker_hit = test_marker
            break
    if analysis_marker_hit is None:
        print(f"No recognized marker in {fluorescence_path}")
        return [], []

    try:
        file_key = __import__(
            "lipid_analysis.filepairing", fromlist=["get_file_key"]
        ).get_file_key(os.path.basename(fluorescence_path), config)
    except ValueError:
        file_key = os.path.basename(fluorescence_path)

    m = re.search(r"Stacks([A-Za-z]+)", file_key)
    stacks_label = m.group(1) if m else ""
    cell_marker_map = config.get("cell_marker_map", {})
    chosen_cell_markers = cell_marker_map.get(
        stacks_label, config.get("cell_markers", [])
    )
    print(
        f"{'Using custom' if stacks_label in cell_marker_map else 'Using default'} marker set for '{stacks_label}': {chosen_cell_markers}"
    )

    all_positions_results, all_positions_summary = [], []

    with ND2Reader(fluorescence_path) as fluoro_nd2, ND2Reader(cars_path) as cars_nd2:
        fluoro_nd2.iter_axes = "v"
        cars_nd2.iter_axes = "v"
        pixel_size_microns = fluoro_nd2.metadata["pixel_microns"]

        def max_project_cars(nd2obj, c_index, position, reference_image, foci_params):
            from skimage.filters import gaussian

            z_stack_slices_cars = []
            total_z = nd2obj.sizes.get("z", 1)
            blur_sigma = float(foci_params.get("sigma", 0.0) or 0.0)
            for z_slice in range(total_z):
                raw_sl = np.nan_to_num(
                    nd2obj.get_frame_2D(v=position, c=c_index, z=z_slice)
                )
                correlated_sl = apply_east_shadows_filter(raw_sl)
                den = np.clip(reference_image, 1e-6, None)
                slice_div = correlated_sl / den
                blurred_sl = gaussian(slice_div, sigma=blur_sigma, preserve_range=True)
                z_stack_slices_cars.append(blurred_sl)
            return np.max(np.array(z_stack_slices_cars), axis=0)

        for pos in range(fluoro_nd2.sizes["v"]):
            fluoro_nd2.default_coords["v"] = pos
            cars_nd2.default_coords["v"] = pos
            file_stub = (
                os.path.splitext(os.path.basename(fluorescence_path))[0]
                + f"_pos{pos+1}"
            )

            # DAPI (optional)
            dapi_ch_idx = config["channel_map"].get("DAPI", None)
            if dapi_ch_idx is not None:
                dapi_slice = max_project_fluorescence(
                    fluoro_nd2,
                    dapi_ch_idx,
                    pos,
                    config["morphology_params"]["nuclei_params"],
                )
                dapi_mask = process_fluorescence_channel(
                    dapi_slice,
                    **config["morphology_params"]["nuclei_params"],
                    debug=VERBOSE,
                )
                if VERBOSE:
                    debug_display_dapi(dapi_slice, dapi_mask, pos)
            else:
                dapi_mask = None

            # CARS
            corrected_cars_slice = max_project_cars(
                cars_nd2, CARS_CH, pos, reference_image, foci_params
            )

            # Quick fluorescence preview composite (alias-aware; no filename gating)
            fluor_images_for_display = {}
            H, W = corrected_cars_slice.shape
            for cm in chosen_cell_markers:
                try:
                    cm_key = resolve_marker_name(cm, config)
                except KeyError:
                    continue
                ch_idx = config["channel_map"].get(cm_key)
                if ch_idx is None:
                    continue
                z_stack_fl = [
                    np.nan_to_num(fluoro_nd2.get_frame_2D(v=pos, c=ch_idx, z=z_idx))
                    for z_idx in range(fluoro_nd2.sizes.get("z", 1))
                ]
                img = np.max(np.array(z_stack_fl), axis=0)
                if img.shape == (H, W):
                    fluor_images_for_display[cm_key] = (
                        img  # only include matching shapes
                    )

            if len(fluor_images_for_display) > 0:
                composite_fluor = composite_fluorescence(
                    fluor_images_for_display, config
                )
            else:
                composite_fluor = np.zeros((H, W, 3), dtype=np.uint8)

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(composite_fluor)
            axs[0].set_title(f"Max Projected Fluorescence Overlay (pos={pos+1})")
            axs[0].axis("off")
            axs[1].imshow(corrected_cars_slice, cmap="gray")
            axs[1].set_title(f"Max Projected CARS (pos={pos+1})")
            axs[1].axis("off")
            plt.show()
            plt.close(fig)

            # Optional: myelin
            try:
                from . import myelin_analysis

                myelin_mask, myelin_pct = myelin_analysis.detect_myelin(
                    corrected_cars_slice,
                    gaussian_sigma=1.0,
                    offset=0.9,
                    min_size=300,
                    closing_radius=1,
                    debug=VERBOSE,
                )
            except Exception:
                myelin_pct = 0.0

            # Lipid-ish foci from CARS
            cars_foci_mask = find_foci(corrected_cars_slice, **foci_params)

            # Saturation/amyloid-like mask
            sat_thresh = foci_params["saturation_threshold"]
            sat_min = foci_params["saturated_min_size"]
            saturated_pixels = corrected_cars_slice >= sat_thresh
            labeled_sat = measure.label(saturated_pixels)
            amyloid_mask = np.zeros_like(corrected_cars_slice, dtype=bool)
            for region in measure.regionprops(labeled_sat):
                if region.area >= sat_min:
                    amyloid_mask[tuple(region.coords.T)] = True
            amyloid_pct = amyloid_mask.sum() / amyloid_mask.size

            # Autofluorescence channel (optional)
            auto_ch_idx = config["channel_map"].get("Autofluorescence")
            auto_mask = np.zeros_like(corrected_cars_slice, dtype=bool)
            auto_slice = None
            if auto_ch_idx is not None:
                auto_slice = max_project_fluorescence(
                    fluoro_nd2,
                    auto_ch_idx,
                    pos,
                    config["morphology_params"]["autofluorescence_params"],
                )
                auto_mask = find_foci(
                    auto_slice, **autofluorescence_params, debug=VERBOSE
                )

            # LAMP2 (optional)
            lamp2_mask = None
            lamp2_ch_idx = config["channel_map"].get("LAMP2", None)
            if lamp2_ch_idx is not None:
                try:
                    lamp2_mip = max_project_fluorescence(
                        fluoro_nd2,
                        lamp2_ch_idx,
                        pos,
                        config["morphology_params"]["fluorescence_params"],
                    )
                    lamp2_params = config["morphology_params"].get(
                        "lamp2_params",
                        config["morphology_params"]["autofluorescence_params"],
                    )
                    lamp2_mask = find_foci(lamp2_mip, **lamp2_params, debug=VERBOSE)
                    lamp2_mask = dilation(lamp2_mask, disk(1))
                except Exception:
                    lamp2_mask = None

            # 3-way masks
            pure_lipid_mask = cars_foci_mask & ~auto_mask
            lipid_lipofuscin_mask = cars_foci_mask & auto_mask
            pure_lipofuscin_mask = auto_mask & ~cars_foci_mask

            labeled_pure_lipid = measure.label(pure_lipid_mask)
            labeled_lipo_lipid = measure.label(lipid_lipofuscin_mask)
            labeled_pure_lipo = measure.label(pure_lipofuscin_mask)

            # Per-marker cell masks (alias-aware; no filename gating)
            for cm in chosen_cell_markers:
                try:
                    cm_key = resolve_marker_name(cm, config)
                except KeyError:
                    continue
                cm_channel_idx = config["channel_map"].get(cm_key, None)
                if cm_channel_idx is None:
                    continue

                cm_slice = max_project_fluorescence(
                    fluoro_nd2, cm_channel_idx, pos, fluorescence_params
                )

                marker_thresholds = config.get("marker_thresholds", {})
                cm_thresholds = marker_thresholds.get(cm_key, {})
                threshold_method = cm_thresholds.get(
                    "threshold_method",
                    fluorescence_params.get("threshold_method", "otsu"),
                )
                offset_val = cm_thresholds.get(
                    "offset", fluorescence_params.get("offset", 1.0)
                )

                cm_mask = process_fluorescence_channel(
                    cm_slice,
                    cell_size=fluorescence_params["cell_size"],
                    min_size=fluorescence_params["min_size"],
                    closing_radius=fluorescence_params["closing_radius"],
                    gaussian_sigma=fluorescence_params["gaussian_sigma"],
                    fill_holes=fluorescence_params["fill_holes"],
                    threshold_method=threshold_method,
                    offset=offset_val,
                )

                if VERBOSE:
                    debug_display_3way_segmentation(
                        pure_lipid_mask,
                        lipid_lipofuscin_mask,
                        pure_lipofuscin_mask,
                        cm_mask,
                        auto_image=auto_slice,
                        cars_image=corrected_cars_slice,
                        pos_index=pos,
                        title_suffix=f"[{cm_key}]",
                    )

                pos_results, pos_summary = analyze_3way_intracellular_objects(
                    labeled_pure_lipid,
                    labeled_lipo_lipid,
                    labeled_pure_lipo,
                    cm_mask,
                    corrected_cars_slice,
                    auto_slice,
                    file_name=os.path.basename(fluorescence_path),
                    z_stack=pos + 1,
                    pixel_size_microns=pixel_size_microns,
                    lamp2_mask=lamp2_mask,
                )
                for r_item in pos_results:
                    r_item["cell_marker"] = cm_key
                for s_item in pos_summary:
                    s_item["cell_marker"] = cm_key
                    s_item["myelination_percentage"] = myelin_pct
                    s_item["amyloid_percentage"] = amyloid_pct

                all_positions_results.extend(pos_results)
                all_positions_summary.extend(pos_summary)

                # Quick binary-mask overlays for sanity check
                def create_rgb_mask(bin_mask, rgb_color):
                    rgb_m = np.zeros((*bin_mask.shape, 3), dtype=np.uint8)
                    for i_col in range(3):
                        rgb_m[..., i_col] = bin_mask * rgb_color[i_col]
                    return rgb_m

                green, yellow = [0, 255, 0], [255, 255, 0]
                cell_rgb_mask = create_rgb_mask(cm_mask, green)
                cars_rgb_mask = create_rgb_mask(cars_foci_mask, yellow)
                overlay_rgb_mask = np.clip(
                    0.5 * cell_rgb_mask + 0.5 * cars_rgb_mask, 0, 255
                ).astype(np.uint8)

                fig_mask, axs_mask = plt.subplots(1, 3, figsize=(18, 6))
                axs_mask[0].imshow(cell_rgb_mask)
                axs_mask[0].set_title(f"{cm_key} Cell Mask (pos={pos+1})")
                axs_mask[0].axis("off")
                axs_mask[1].imshow(cars_rgb_mask)
                axs_mask[1].set_title("CARS Mask (pos={pos+1})")
                axs_mask[1].axis("off")
                axs_mask[2].imshow(overlay_rgb_mask)
                axs_mask[2].set_title(f"Overlay (pos={pos+1}) [{cm_key}]")
                axs_mask[2].axis("off")
                plt.show()
                plt.close(fig_mask)

                if dapi_mask is not None:
                    images_dir = ensure_subdirectory(
                        config["paths"]["data_directory"], "Images"
                    )
                    out_overlay_path = os.path.join(
                        images_dir, f"{file_stub}_DAPI_{cm_key}.png"
                    )
                    from .visualize import save_dapi_marker_overlay

                    save_dapi_marker_overlay(
                        dapi_mask, cm_mask, cm_key, out_overlay_path, config=config
                    )

            # Save composite images (all channels)
            fluor_images_for_composite = {}
            for marker_name, ch_idx in config["channel_map"].items():
                if ch_idx is None:
                    continue
                z_stack_fl = [
                    np.nan_to_num(fluoro_nd2.get_frame_2D(v=pos, c=ch_idx, z=z_slice))
                    for z_slice in range(fluoro_nd2.sizes.get("z", 1))
                ]
                marker_max = np.max(np.array(z_stack_fl), axis=0)
                fluor_images_for_composite[marker_name] = marker_max
            save_composite_images(
                fluor_images_for_composite,
                corrected_cars_slice,
                config,
                config["paths"]["data_directory"],
                file_stub,
            )

    return all_positions_results, all_positions_summary
