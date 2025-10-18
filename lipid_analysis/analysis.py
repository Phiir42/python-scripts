"""High-level analysis for pairing and quantifying intracellular lipid features.

This module:
- Projects fluorescence and CARS stacks to 2D.
- Builds preview composites (gated by VERBOSE).
- Segments myelin, lipid, and lipofuscin masks.
- Optionally incorporates LAMP2 and DAPI.
- Quantifies per-cell feature counts/areas and aggregates summaries.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from nd2reader import ND2Reader
from skimage import measure

from .config_utils import resolve_marker_name
from .constants import CARS_CH, LOG_LEVEL, VERBOSE
from .imaging import composite_fluorescence, get_corrected_cars_stack, get_fluorescence_stack
from .io_utils import ensure_subdirectory, save_composite_images
from .segmentation import find_foci, process_fluorescence_channel, process_fluorescence_stack, assess_low_feature, colocalize_objects_3d
from .visualize import debug_display_3way_segmentation, debug_display_dapi


logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

Results = List[MutableMapping[str, object]]
Summary = List[MutableMapping[str, object]]


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #


def _get_pixel_size_microns(nd2: ND2Reader) -> float:
    """Extract a scalar pixel size in microns from ND2 metadata, with validation."""
    meta = getattr(nd2, "metadata", None)
    if not isinstance(meta, dict) or "pixel_microns" not in meta:
        raise ValueError("ND2 metadata missing 'pixel_microns'.")
    try:
        px = float(meta["pixel_microns"])
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Unusable 'pixel_microns' value in ND2 metadata: {meta['pixel_microns']!r}"
        ) from exc
    if px <= 0:
        raise ValueError(f"Non-positive pixel size in microns: {px}")
    return px


def max_project_fluorescence(
    nd2obj: ND2Reader,
    ch_index: int,
    position: int,
    fluoro_params: Mapping[str, object],
) -> np.ndarray:
    """
    Max-project a fluorescence channel at a given stage position (v index).

    Applies an optional Gaussian blur on each z-slice before the max projection.

    Parameters
    ----------
    nd2obj
        Open ND2Reader instance configured for iteration by positions.
    ch_index
        Channel index to read.
    position
        Stage position index (v).
    fluoro_params
        Dict-like with optional key 'gaussian_sigma' (float).

    Returns
    -------
    np.ndarray
        2D float array (H, W) of the max-projected fluorescence.
    """
    from skimage.filters import gaussian

    total_z = nd2obj.sizes.get("z", 1)
    gaussian_sigma = float(fluoro_params.get("gaussian_sigma", 0.0) or 0.0)
    slices: List[np.ndarray] = []

    for z_slice in range(total_z):
        raw = nd2obj.get_frame_2D(v=position, c=ch_index, z=z_slice)
        raw = np.nan_to_num(raw)
        if gaussian_sigma > 0:
            raw = gaussian(raw, sigma=gaussian_sigma, preserve_range=True)
        slices.append(raw)

    return np.max(np.stack(slices, axis=0), axis=0)


def analyze_3way_intracellular_objects(
    labeled_pure_lipid_3d: np.ndarray,
    labeled_lipo_lipid_3d: np.ndarray,
    labeled_pure_lipo_3d: np.ndarray,
    cell_mask_3d: np.ndarray,
    cars_stack: np.ndarray,
    auto_stack: Optional[np.ndarray],
    file_name: str,
    z_stack: int,
    pixel_size_microns: float,
    lamp2_mask_3d: Optional[np.ndarray] = None,
) -> Tuple[Results, Summary]:
    """
    Quantify pure lipid, lipid+lipofuscin, and pure lipofuscin objects per cell.

    For each cell region:
      - counts/areas of objects per category,
      - mean intensity per object (CARS for lipid-containing, autofluorescence
        for pure lipofuscin),
      - optional LAMP2 colocalization counts,
      - per-cell percentages (area / cell area).

    Returns both per-object records ('results') and per-cell summaries ('summary').
    """
    # Label cells directly in 3-D.
    if cell_mask_3d.ndim != 3:
        raise ValueError(f"cell_mask_3d must be (Z,H,W), got {cell_mask_3d.shape}")
    labeled_cells_3d = measure.label(cell_mask_3d.astype(bool, copy=False), connectivity=1)
    
    results: Results = []
    summary: Summary = []
    
    for cell in measure.regionprops(labeled_cells_3d):
        cell_id = int(cell.label)
        # 3-D mask and counts
        cell_mask_region_3d = (labeled_cells_3d == cell_id)
        # Keep legacy 2-D "area" fields by projecting the 3-D cell to MIP:
        cell_area_px = int(np.max(cell_mask_region_3d, axis=0).sum())
        cell_area_um2 = float(cell_area_px) * (pixel_size_microns**2)
    
        pure_lipid_lamp2_count = 0
        lipid_lipo_lamp2_count = 0
        pure_lipo_lamp2_count = 0
    
        def _overlaps_lamp2(coords_3d: np.ndarray) -> bool:
            if lamp2_mask_3d is None:
                return False
            zz, yy, xx = coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2]
            return bool(np.any(lamp2_mask_3d[zz, yy, xx]))
    
        # A) pure lipid (3-D)
        pure_lipid_in_cell = labeled_pure_lipid_3d * cell_mask_region_3d
        pure_lipid_count = 0
        pure_lipid_area_um2 = 0.0
        for region_lipid in measure.regionprops(pure_lipid_in_cell, intensity_image=cars_stack):
            # regionprops in 3-D: area = voxel count
            voxels = int(region_lipid.area)
            # Report "area" in µm² as before (legacy); optionally add a true volume later
            area_um2 = float(voxels) * (pixel_size_microns**2)
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
                    "cell_area": cell_area_px,
                    "cell_area_um2": cell_area_um2,
                    "feature_type": "pure_lipid",
                    "feature_size_pixels": voxels,
                    "feature_size_um2": area_um2,
                    "feature_intensity": float(region_lipid.mean_intensity),
                    "lamp2_colocalized": lamp2_hit,
                }
            )
    
        # B) lipid + lipofuscin (3-D; cars intensity)
        lipo_lipid_in_cell = labeled_lipo_lipid_3d * cell_mask_region_3d
        lipid_lipo_count = 0
        lipid_lipo_area_um2 = 0.0
        for region_ll in measure.regionprops(lipo_lipid_in_cell, intensity_image=cars_stack):
            voxels = int(region_ll.area)
            area_um2 = float(voxels) * (pixel_size_microns**2)
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
                    "cell_area": cell_area_px,
                    "cell_area_um2": cell_area_um2,
                    "feature_type": "lipid_lipofuscin",
                    "feature_size_pixels": voxels,
                    "feature_size_um2": area_um2,
                    "feature_intensity": float(region_ll.mean_intensity),
                    "lamp2_colocalized": lamp2_hit,
                }
            )
    
        # C) pure lipofuscin (3-D; AF intensity if available)
        pure_lipo_in_cell = labeled_pure_lipo_3d * cell_mask_region_3d
        pure_lipo_count = 0
        pure_lipo_area_um2 = 0.0
        if auto_stack is None:
            logger.debug("No autofluorescence stack provided for intensity stats.")
        for region_pure_l in measure.regionprops(
            pure_lipo_in_cell, intensity_image=(auto_stack if auto_stack is not None else cars_stack)
        ):
            voxels = int(region_pure_l.area)
            area_um2 = float(voxels) * (pixel_size_microns**2)
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
                    "cell_area": cell_area_px,
                    "cell_area_um2": cell_area_um2,
                    "feature_type": "pure_lipofuscin",
                    "feature_size_pixels": voxels,
                    "feature_size_um2": area_um2,
                    "feature_intensity": float(region_pure_l.mean_intensity),
                    "lamp2_colocalized": lamp2_hit,
                }
            )
    
        def _pct(area_um2: float) -> float:
            return 100.0 * area_um2 / cell_area_um2 if cell_area_um2 > 0 else 0.0
    
        summary.append(
            {
                "file_name": file_name,
                "z_stack": z_stack,
                "cell_id": cell_id,
                "cell_area": cell_area_px,
                "cell_area_um2": cell_area_um2,
                "pure_lipid_count": pure_lipid_count,
                "pure_lipid_area_um2": pure_lipid_area_um2,
                "pure_lipid_percentage": _pct(pure_lipid_area_um2),
                "lipid_lipofuscin_count": lipid_lipo_count,
                "lipid_lipofuscin_area_um2": lipid_lipo_area_um2,
                "lipid_lipofuscin_percentage": _pct(lipid_lipo_area_um2),
                "lipofuscin_count": pure_lipo_count,
                "lipofuscin_area_um2": pure_lipo_area_um2,
                "lipofuscin_percentage": _pct(pure_lipo_area_um2),
                "pure_lipid_lamp2_count": pure_lipid_lamp2_count,
                "lipid_lipofuscin_lamp2_count": lipid_lipo_lamp2_count,
                "lipofuscin_lamp2_count": pure_lipo_lamp2_count,
            }
        )
    return results, summary


# --------------------------------------------------------------------------- #
# Main ND2 pair processing
# --------------------------------------------------------------------------- #


def process_nd2_pair(
    fluorescence_path: str,
    cars_path: str,
    reference_image: np.ndarray,
    config: Mapping[str, object],
) -> Tuple[Results, Summary]:
    """
    Process a fluorescence/CARS ND2 pair and return (results, summary).

    Notes
    -----
    - Uses max-projection of z-stacks for both fluorescence and CARS.
    - Corrects CARS using east-shadows correlation and per-pixel division by a
      precomputed reference image (clipped to avoid divide-by-zero).
    - Builds per-marker cell masks and quantifies 3-way (pure lipid / lipid+lipofuscin
      / pure lipofuscin) features per cell.
    - Optional channels: DAPI (nuclei), Autofluorescence, LAMP2.
    - Creates preview plots only when VERBOSE is True.
    """
    foci_params = config["morphology_params"]["foci_params"]  # type: ignore[index]
    fluorescence_params = config["morphology_params"]["fluorescence_params"]  # type: ignore[index]
    autofluorescence_params = config["morphology_params"]["autofluorescence_params"]  # type: ignore[index]

    # Determine analysis marker from fluorescence filename (first token hit)
    analysis_marker_hit: Optional[str] = None
    for test_marker in config["file_keywords"]["fluorescence_markers"]:  # type: ignore[index]
        if test_marker in os.path.basename(fluorescence_path):
            analysis_marker_hit = test_marker
            break
    if analysis_marker_hit is None:
        logger.warning("No recognized marker in %s", fluorescence_path)
        return [], []

    # Try to produce a readable file key (no hard dependency if parse fails)
    try:
        from .filepairing import get_file_key

        file_key = get_file_key(os.path.basename(fluorescence_path), config)
    except Exception:
        file_key = os.path.basename(fluorescence_path)

    # Per-stacks label selection of cell markers
    m = re.search(r"Stacks([A-Za-z]+)", file_key)
    stacks_label = m.group(1) if m else ""
    cell_marker_map = config.get("cell_marker_map", {})  # type: ignore[assignment]
    chosen_cell_markers = cell_marker_map.get(  # type: ignore[call-arg]
        stacks_label, config.get("cell_markers", [])  # type: ignore[arg-type]
    )
    logger.info(
        "%s marker set for '%s': %s",
        "Using custom" if stacks_label in cell_marker_map else "Using default",
        stacks_label,
        chosen_cell_markers,
    )

    all_positions_results: Results = []
    all_positions_summary: Summary = []

    with ND2Reader(fluorescence_path) as fluoro_nd2, ND2Reader(cars_path) as cars_nd2:
        fluoro_nd2.iter_axes = "v"
        cars_nd2.iter_axes = "v"
        pixel_size_microns = _get_pixel_size_microns(fluoro_nd2)

        for pos in range(fluoro_nd2.sizes["v"]):
            fluoro_nd2.default_coords["v"] = pos
            cars_nd2.default_coords["v"] = pos
            file_stub = (
                os.path.splitext(os.path.basename(fluorescence_path))[0]
                + f"_pos{pos + 1}"
            )

            # DAPI (optional)
            dapi_ch_idx = config["channel_map"].get("DAPI", None)  # type: ignore[index]
            if dapi_ch_idx is not None:
                dapi_slice = max_project_fluorescence(
                    fluoro_nd2,
                    int(dapi_ch_idx),
                    pos,
                    config["morphology_params"]["nuclei_params"],  # type: ignore[index]
                )
                dapi_mask = process_fluorescence_channel(
                    dapi_slice,
                    **config["morphology_params"]["nuclei_params"],  # type: ignore[arg-type]
                    debug=VERBOSE,
                )
                if VERBOSE:
                    debug_display_dapi(dapi_slice, dapi_mask, pos)
            else:
                dapi_mask = None

            # 3-D corrected CARS stack + 2-D MIP for displays/quick finders
            corrected_cars_stack = get_corrected_cars_stack(
                cars_nd2, CARS_CH, pos, reference_image, foci_params
            )  # (Z, H, W)
            corrected_cars_slice = corrected_cars_stack.max(axis=0)

            # Quick fluorescence preview composite (alias-aware; no filename gating)
            fluor_images_for_display: Dict[str, np.ndarray] = {}
            H, W = corrected_cars_slice.shape
            for cm in chosen_cell_markers:  # type: ignore[assignment]
                try:
                    cm_key = resolve_marker_name(cm, config)
                except KeyError:
                    continue
                ch_idx = config["channel_map"].get(cm_key)  # type: ignore[index]
                if ch_idx is None:
                    continue
                z_stack_fl = [
                    np.nan_to_num(fluoro_nd2.get_frame_2D(v=pos, c=int(ch_idx), z=z_idx))
                    for z_idx in range(fluoro_nd2.sizes.get("z", 1))
                ]
                img = np.max(np.stack(z_stack_fl, axis=0), axis=0)
                if img.shape == (H, W):
                    fluor_images_for_display[cm_key] = img

            if fluor_images_for_display:
                composite_fluor = composite_fluorescence(fluor_images_for_display, config)
            else:
                composite_fluor = np.zeros((H, W, 3), dtype=np.uint8)

            if VERBOSE:
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                axs[0].imshow(composite_fluor)
                axs[0].set_title(f"Max Projected Fluorescence Overlay (pos={pos + 1})")
                axs[0].axis("off")
                axs[1].imshow(corrected_cars_slice, cmap="gray")
                axs[1].set_title(f"Max Projected CARS (pos={pos + 1})")
                axs[1].axis("off")
                plt.show()
                plt.close(fig)

            # Build per-slice myelin masks → 3-D myelin array (Z, H, W)
            try:
                myelin_masks_z: List[np.ndarray] = []
                for z in range(corrected_cars_stack.shape[0]):
                    z_img = corrected_cars_stack[z]
            
                    # Per-slice low-feature assessment (same thresholds as before)
                    low_feat, metrics = assess_low_feature(
                        z_img,
                        sigma=1.0,
                        remove_saturated=True,
                        sat_thresh=float(foci_params["saturation_threshold"]),  # type: ignore[index]
                        sat_min=int(foci_params["saturated_min_size"]),         # type: ignore[index]
                        edge_min_frac=0.16,
                        lap_var_thresh=4.8e-3,
                        snr_thresh=0.285,
                        debug=False,  # set VERBOSE for slice-level prints
                    )
                    sdmul = 1.6 if low_feat else 0.8
            
                    mask_z = find_foci(
                        z_img,
                        sigma=1.0,
                        min_distance=8,
                        min_size=300,
                        std_dev_multiplier=sdmul,
                        remove_saturated=True,
                        saturation_threshold=float(foci_params["saturation_threshold"]),  # type: ignore[index]
                        saturated_min_size=int(foci_params["saturated_min_size"]),        # type: ignore[index]
                        separate_objects=False,
                        morph_op="closing",
                        morph_radius=2,
                        debug=False,
                    )
                    myelin_masks_z.append(mask_z.astype(bool, copy=False))
            
                myelin_3d = np.stack(myelin_masks_z, axis=0)  # (Z, H, W) bool
            
            except Exception:
                logger.exception("Per-slice myelin detection failed; continuing with empty 3-D mask.")
                myelin_3d = np.zeros_like(corrected_cars_stack, dtype=bool)

            # Per-slice CARS droplet/foci mask (3-D)
            cars_foci_masks_z: List[np.ndarray] = []
            for z in range(corrected_cars_stack.shape[0]):
                mask_z = find_foci(corrected_cars_stack[z], **foci_params)
                cars_foci_masks_z.append(mask_z.astype(bool, copy=False))
            cars_foci_mask_3d = np.stack(cars_foci_masks_z, axis=0)  # (Z, H, W) bool

            # Saturation/amyloid-like mask
            sat_thresh = float(foci_params["saturation_threshold"])  # type: ignore[index]
            sat_min = int(foci_params["saturated_min_size"])  # type: ignore[index]
            saturated_pixels = corrected_cars_slice >= sat_thresh
            labeled_sat = measure.label(saturated_pixels)
            amyloid_mask = np.zeros_like(corrected_cars_slice, dtype=bool)
            for region in measure.regionprops(labeled_sat):
                if region.area >= sat_min:
                    amyloid_mask[tuple(region.coords.T)] = True
            amyloid_pct = float(amyloid_mask.sum()) / float(amyloid_mask.size)

            # Autofluorescence channel (optional)
            auto_ch_idx = config["channel_map"].get("Autofluorescence")  # type: ignore[index]
            auto_slice: Optional[np.ndarray] = None            # 2-D MIP (for debug)
            auto_mask_3d = np.zeros_like(corrected_cars_stack, dtype=bool)  # (Z,H,W) default
            if auto_ch_idx is not None:
                # 3-D AF stack
                auto_stack = get_fluorescence_stack(
                    fluoro_nd2, int(auto_ch_idx), pos, config["morphology_params"]["autofluorescence_params"]  # type: ignore[index]
                )  # (Z, H, W)
                # 2-D MIP only for displays/debug
                auto_slice = auto_stack.max(axis=0)
            
                # Per-slice AF mask (3-D)
                af_masks_z: List[np.ndarray] = []
                for z in range(auto_stack.shape[0]):
                    af_z = find_foci(auto_stack[z], **autofluorescence_params)  # type: ignore[arg-type]
                    af_masks_z.append(af_z.astype(bool, copy=False))
                auto_mask_3d = np.stack(af_masks_z, axis=0)

            # LAMP2 (optional)
            lamp2_mask_3d: Optional[np.ndarray] = None
            lamp2_ch_idx = config["channel_map"].get("LAMP2", None)  # type: ignore[index]
            if lamp2_ch_idx is not None:
                try:
                    # 3-D LAMP2 stack
                    lamp2_stack = get_fluorescence_stack(
                        fluoro_nd2,
                        int(lamp2_ch_idx),
                        pos,
                        config["morphology_params"]["fluorescence_params"],  # type: ignore[index]
                    )
                    lamp2_params = config["morphology_params"].get(  # type: ignore[index]
                        "lamp2_params",
                        config["morphology_params"]["autofluorescence_params"],  # type: ignore[index]
                    )
                    lamp2_masks_z: List[np.ndarray] = []
                    from skimage.morphology import dilation
                    for z in range(lamp2_stack.shape[0]):
                        lz = find_foci(lamp2_stack[z], **lamp2_params)  # type: ignore[arg-type]
                        # light 2-D dilation like before (disk(1)); keep 2-D to match prior behavior
                        from skimage.morphology import disk
                        lz = dilation(lz, disk(1))
                        lamp2_masks_z.append(lz.astype(bool, copy=False))
                    lamp2_mask_3d = np.stack(lamp2_masks_z, axis=0)  # (Z,H,W)
                except Exception:
                    logger.exception("LAMP2 detection failed; continuing without it.")
                    lamp2_mask_3d = None
            
            min_overlap = int(foci_params.get("min_size", 20))
            labeled_pure_lipid_3d, labeled_lipo_lipid_3d, labeled_pure_lipo_3d = colocalize_objects_3d(
                cars_foci_mask_3d,
                auto_mask_3d,
                min_overlap=min_overlap,
            )
            
            # Boolean views for visualization and myelin exclusion (3-D)
            pure_lipid_3d        = (labeled_pure_lipid_3d > 0)
            lipid_lipofuscin_3d  = (labeled_lipo_lipid_3d > 0)
            pure_lipofuscin_3d   = (labeled_pure_lipo_3d > 0)

            # Strict 3-D exclusion: subtract all other features (3-D union) from myelin_3d
            other_features_3d = (pure_lipid_3d | lipid_lipofuscin_3d | pure_lipofuscin_3d)
            myelin_3d_refined = myelin_3d & (~other_features_3d)
            
            # Volume fraction (voxels) → percentage
            myelin_vol_frac = float(myelin_3d_refined.sum()) / float(myelin_3d_refined.size) if myelin_3d_refined.size else 0.0
            myelin_pct = 100.0 * myelin_vol_frac
            
            # 2-D MIPs for unchanged debug display
            pure_lipid_mask        = pure_lipid_3d.max(axis=0)
            lipid_lipofuscin_mask  = lipid_lipofuscin_3d.max(axis=0)
            pure_lipofuscin_mask   = pure_lipofuscin_3d.max(axis=0)
            myelin_mask_refined    = myelin_3d_refined.max(axis=0)
            cars_foci_mask         = cars_foci_mask_3d.max(axis=0)  # for the quick CARS overlay panel

            # Per-marker cell masks (alias-aware; no filename gating)
            for cm in chosen_cell_markers:  # type: ignore[assignment]
                try:
                    cm_key = resolve_marker_name(cm, config)
                except KeyError:
                    continue
                cm_channel_idx = config["channel_map"].get(cm_key, None)  # type: ignore[index]
                if cm_channel_idx is None:
                    continue

                # Build a (Z,H,W) fluorescence stack for the marker
                cm_stack = get_fluorescence_stack(
                    fluoro_nd2, int(cm_channel_idx), pos, fluorescence_params
                )

                marker_thresholds = config.get("marker_thresholds", {})  # type: ignore[assignment]
                cm_thresholds = marker_thresholds.get(cm_key, {})  # type: ignore[call-arg]
                threshold_method = cm_thresholds.get(
                    "threshold_method",
                    fluorescence_params.get("threshold_method", "otsu"),  # type: ignore[arg-type]
                )
                offset_val = cm_thresholds.get(
                    "offset", fluorescence_params.get("offset", 1.0)  # type: ignore[arg-type]
                )

                # 3-D cell mask from slice-wise thresholding
                cm_mask_3d = process_fluorescence_stack(
                    cm_stack,
                    cell_size=fluorescence_params["cell_size"],            # type: ignore[index]
                    min_size=fluorescence_params["min_size"],              # type: ignore[index]
                    closing_radius=fluorescence_params["closing_radius"],  # type: ignore[index]
                    gaussian_sigma=fluorescence_params["gaussian_sigma"],  # type: ignore[index]
                    fill_holes=fluorescence_params["fill_holes"],          # type: ignore[index]
                    threshold_method=threshold_method,
                    offset=offset_val,
                    exclude_dark_regions=fluorescence_params.get("exclude_dark_regions", True),  # type: ignore[index]
                    dark_threshold=fluorescence_params.get("dark_threshold", 50),                # type: ignore[index]
                    min_hole_size=fluorescence_params.get("min_hole_size", 20_000),              # type: ignore[index]
                    min_voxels_3d=None,  # set if you want strict 3-D cleanup
                    debug=False,
                )
                # 2-D MIPs for displays/overlays stay exactly as before
                cm_mask = cm_mask_3d.max(axis=0)

                debug_display_3way_segmentation(
                    pure_lipid_mask,
                    lipid_lipofuscin_mask,
                    pure_lipofuscin_mask,
                    cm_mask,
                    auto_image=auto_slice,
                    cars_image=corrected_cars_slice,
                    pos_index=pos,
                    title_suffix=f"[{cm_key}]",
                    myelin_mask=myelin_mask_refined,
                    base_data_dir=config["paths"]["data_directory"],  # type: ignore[index]
                    file_identifier=os.path.splitext(os.path.basename(fluorescence_path))[0],
                    show_plots=VERBOSE,
                )

                pos_results, pos_summary = analyze_3way_intracellular_objects(
                    labeled_pure_lipid_3d,
                    labeled_lipo_lipid_3d,
                    labeled_pure_lipo_3d,
                    cm_mask_3d,                    # 3-D cell mask
                    corrected_cars_stack,       # 3-D CARS
                    auto_stack if auto_ch_idx is not None else None,  # 3-D AF or None
                    file_name=os.path.basename(fluorescence_path),
                    z_stack=pos + 1,
                    pixel_size_microns=pixel_size_microns,
                    lamp2_mask_3d=lamp2_mask_3d,
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
                def create_rgb_mask(bin_mask: np.ndarray, rgb_color: Iterable[int]) -> np.ndarray:
                    rgb = np.zeros((*bin_mask.shape, 3), dtype=np.uint8)
                    for i_col, val in enumerate(rgb_color):
                        rgb[..., i_col] = bin_mask * int(val)
                    return rgb

                if VERBOSE:
                    green, yellow = [0, 255, 0], [255, 255, 0]
                    cell_rgb_mask = create_rgb_mask(cm_mask, green)
                    cars_rgb_mask = create_rgb_mask(cars_foci_mask, yellow)
                    overlay_rgb_mask = np.clip(
                        0.5 * cell_rgb_mask + 0.5 * cars_rgb_mask, 0, 255
                    ).astype(np.uint8)

                    fig_mask, axs_mask = plt.subplots(1, 3, figsize=(18, 6))
                    axs_mask[0].imshow(cell_rgb_mask)
                    axs_mask[0].set_title(f"{cm_key} Cell Mask (pos={pos + 1})")
                    axs_mask[0].axis("off")
                    axs_mask[1].imshow(cars_rgb_mask)
                    axs_mask[1].set_title(f"CARS Mask (pos={pos + 1})")
                    axs_mask[1].axis("off")
                    axs_mask[2].imshow(overlay_rgb_mask)
                    axs_mask[2].set_title(f"Overlay (pos={pos + 1}) [{cm_key}]")
                    axs_mask[2].axis("off")
                    plt.show()
                    plt.close(fig_mask)

                if dapi_mask is not None:
                    images_dir = ensure_subdirectory(
                        config["paths"]["data_directory"], "Images"  # type: ignore[index]
                    )
                    out_overlay_path = os.path.join(
                        images_dir, f"{file_stub}_DAPI_{cm_key}.png"
                    )
                    from .visualize import save_dapi_marker_overlay

                    save_dapi_marker_overlay(
                        dapi_mask, cm_mask, cm_key, out_overlay_path, config=config
                    )

            # Save composite images (all channels)
            fluor_images_for_composite: Dict[str, np.ndarray] = {}
            for marker_name, ch_idx in config["channel_map"].items():  # type: ignore[index]
                if ch_idx is None:
                    continue
                z_stack_fl = [
                    np.nan_to_num(
                        fluoro_nd2.get_frame_2D(v=pos, c=int(ch_idx), z=z_slice)
                    )
                    for z_slice in range(fluoro_nd2.sizes.get("z", 1))
                ]
                marker_max = np.max(np.stack(z_stack_fl, axis=0), axis=0)
                fluor_images_for_composite[str(marker_name)] = marker_max

            save_composite_images(
                fluor_images_for_composite,
                corrected_cars_slice,
                config,
                config["paths"]["data_directory"],  # type: ignore[index]
                file_stub,
            )

    return all_positions_results, all_positions_summary
