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
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple, Set

import matplotlib.pyplot as plt
import numpy as np
from nd2reader import ND2Reader
from skimage import measure

from .config_utils import resolve_marker_name
from .constants import CARS_CH, LOG_LEVEL, VERBOSE
from .filters import apply_east_shadows_filter
from .imaging import composite_fluorescence
from .io_utils import ensure_subdirectory, save_composite_images
from .segmentation import find_foci, process_fluorescence_channel
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
    labeled_pure_lipid: np.ndarray,
    labeled_lipo_lipid: np.ndarray,
    labeled_pure_lipo: np.ndarray,
    cell_mask: np.ndarray,
    cars_image: np.ndarray,
    auto_image: Optional[np.ndarray],
    file_name: str,
    z_stack: int,
    pixel_size_microns: float,
    lamp2_mask: Optional[np.ndarray] = None,
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
    labeled_cells = measure.label(cell_mask)
    results: Results = []
    summary: Summary = []

    for cell in measure.regionprops(labeled_cells):
        cell_id = int(cell.label)
        cell_area = int(cell.area)
        cell_area_um2 = float(cell_area) * (pixel_size_microns**2)
        cell_mask_region = labeled_cells == cell_id

        pure_lipid_lamp2_count = 0
        lipid_lipo_lamp2_count = 0
        pure_lipo_lamp2_count = 0

        def _overlaps_lamp2(coords: np.ndarray) -> bool:
            if lamp2_mask is None:
                return False
            rr, cc = coords[:, 0], coords[:, 1]
            return bool(np.any(lamp2_mask[rr, cc]))

        # A) pure lipid (CARS+ AF-)
        pure_lipid_in_cell = labeled_pure_lipid * cell_mask_region
        pure_lipid_count = 0
        pure_lipid_area_um2 = 0.0
        for region_lipid in measure.regionprops(
            pure_lipid_in_cell, intensity_image=cars_image
        ):
            area_px = int(region_lipid.area)
            area_um2 = float(area_px) * (pixel_size_microns**2)
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
                    "feature_intensity": float(region_lipid.mean_intensity),
                    "lamp2_colocalized": lamp2_hit,
                }
            )

        # B) lipid + lipofuscin (CARS+ AF+)
        lipo_lipid_in_cell = labeled_lipo_lipid * cell_mask_region
        lipid_lipo_count = 0
        lipid_lipo_area_um2 = 0.0
        for region_ll in measure.regionprops(
            lipo_lipid_in_cell, intensity_image=cars_image
        ):
            area_px = int(region_ll.area)
            area_um2 = float(area_px) * (pixel_size_microns**2)
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
                    "feature_intensity": float(region_ll.mean_intensity),
                    "lamp2_colocalized": lamp2_hit,
                }
            )

        # C) pure lipofuscin (CARS- AF+)
        pure_lipo_in_cell = labeled_pure_lipo * cell_mask_region
        pure_lipo_count = 0
        pure_lipo_area_um2 = 0.0
        if auto_image is None:
            logger.debug("No autofluorescence image provided for intensity stats.")
        for region_pure_l in measure.regionprops(
            pure_lipo_in_cell, intensity_image=auto_image
        ):
            area_px = int(region_pure_l.area)
            area_um2 = float(area_px) * (pixel_size_microns**2)
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
                    "feature_intensity": float(region_pure_l.mean_intensity),
                    "lamp2_colocalized": lamp2_hit,
                }
            )

        # Per-cell percentages
        def _pct(area_um2: float) -> float:
            return 100.0 * area_um2 / cell_area_um2 if cell_area_um2 > 0 else 0.0

        summary.append(
            {
                "file_name": file_name,
                "z_stack": z_stack,
                "cell_id": cell_id,
                "cell_area": cell_area,
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

        def max_project_cars(
            nd2obj: ND2Reader,
            c_index: int,
            position: int,
            ref_image: np.ndarray,
            fparams: Mapping[str, object],
        ) -> np.ndarray:
            from skimage.filters import gaussian

            total_z = nd2obj.sizes.get("z", 1)
            blur_sigma = float(fparams.get("sigma", 0.0) or 0.0)
            slices: List[np.ndarray] = []
            den = np.clip(ref_image.astype(np.float32, copy=False), 1e-6, None)

            for z_slice in range(total_z):
                raw = np.nan_to_num(nd2obj.get_frame_2D(v=position, c=c_index, z=z_slice))
                correlated = apply_east_shadows_filter(raw)
                div = correlated / den
                if blur_sigma > 0:
                    div = gaussian(div, sigma=blur_sigma, preserve_range=True)
                slices.append(div)

            return np.max(np.stack(slices, axis=0), axis=0)

        def _assess_low_feature(
            img2d: np.ndarray,
            sigma: float,
            remove_saturated: bool,
            sat_thresh: float,
            sat_min: int,
            rr_thresh: float = 0.12,
            edge_q: float = 1.5,
            edge_min_frac: float = 0.16,   # tuned default
            lap_var_thresh: float = 4.8e-3,
            snr_thresh: Optional[float] = 0.285,
            debug: bool = False,
        ) -> Tuple[bool, Dict[str, float]]:
            """
            Heuristically decide if the image is "low-feature" (flat/poor contrast).

            Cues after smoothing & saturation exclusion (mirrors find_foci preproc):
              - robust_range = (p99 - p1) / median
              - edge_density = fraction of Sobel magnitudes > med + edge_q * 1.4826*MAD
              - lap_var      = var(laplace(img / median))
              - snr          = approx_std / median  (approx_std from 1.4826 * MAD)

            Flags low-feature if at least 2 (of up to 4) cues fail thresholds.
            """
            from skimage.filters import gaussian, sobel
            import scipy.ndimage as ndi  # type: ignore

            im = np.nan_to_num(img2d.astype(np.float32), copy=False)
            if sigma > 0:
                im = gaussian(im, sigma=sigma, preserve_range=True)

            # Exclude saturated
            exclude = np.zeros_like(im, dtype=bool)
            if remove_saturated:
                labeled_sat = measure.label(im > sat_thresh)
                for reg in measure.regionprops(labeled_sat):
                    if reg.area >= sat_min:
                        exclude[tuple(reg.coords.T)] = True

            vp = im[~exclude].ravel()
            if vp.size == 0:
                if debug:
                    logger.debug("[LOWFEAT] empty valid region → low-feature")
                return True, {
                    "robust_range": 0.0,
                    "edge_density": 0.0,
                    "lap_var": 0.0,
                    "snr": 0.0,
                }

            med = float(np.median(vp))

            # Robust range
            p1, p99 = np.percentile(vp, [1.0, 99.0])
            robust_range = float((p99 - p1) / max(med, 1e-9))

            # Edge density (Sobel + robust threshold)
            gmag = np.abs(sobel(im))
            gv = gmag[~exclude].ravel()
            gmed = float(np.median(gv)) if gv.size else 0.0
            gmad = float(np.median(np.abs(gv - gmed))) if gv.size else 0.0
            gstd = 1.4826 * gmad
            gthr = gmed + edge_q * gstd
            edge_density = float(np.mean(gv > gthr)) if gv.size else 0.0

            # Laplacian variance (scale-normalized)
            imn = im / max(med, 1e-9)
            lap = ndi.laplace(imn)
            lap_var = float(np.var(lap[~exclude])) if vp.size else 0.0

            # SNR (same definition you've been printing)
            mad = float(np.median(np.abs(vp - med)))
            approx_std = 1.4826 * mad
            snr = approx_std / max(med, 1e-9)

            # Two-(of up to four)-fails rule
            fails = 0
            fails += int(robust_range < rr_thresh)
            fails += int(edge_density < edge_min_frac)
            fails += int(lap_var < lap_var_thresh)
            if snr_thresh is not None:
                fails += int(snr < snr_thresh)

            low_feature = fails >= 2

            if debug:
                logger.debug(
                    "[LOWFEAT] rr=%.4f (<%.3f) | ed=%.4f (<%.3f) | "
                    "lapvar=%.4e (<%.3e) | snr=%.3f (<%s) → low=%s",
                    robust_range,
                    rr_thresh,
                    edge_density,
                    edge_min_frac,
                    lap_var,
                    lap_var_thresh,
                    snr,
                    f"{snr_thresh:.3f}" if snr_thresh is not None else "—",
                    low_feature,
                )

            return low_feature, {
                "robust_range": robust_range,
                "edge_density": edge_density,
                "lap_var": lap_var,
                "snr": snr,
            }

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

            # CARS: max projection with reference correction
            corrected_cars_slice = max_project_cars(
                cars_nd2, CARS_CH, pos, reference_image, foci_params
            )

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

            # Assess low-feature status for this frame (myelin only)
            try:
                low_feat, metrics = _assess_low_feature(
                    corrected_cars_slice,
                    sigma=1.0,
                    remove_saturated=True,
                    sat_thresh=float(foci_params["saturation_threshold"]),  # type: ignore[index]
                    sat_min=int(foci_params["saturated_min_size"]),  # type: ignore[index]
                    edge_min_frac=0.16,
                    lap_var_thresh=4.8e-3,
                    snr_thresh=0.285,  # set to None to disable SNR
                    debug=VERBOSE,
                )
                sdmul = 1.6 if low_feat else 0.8  # adjust std-dev multiplier gently
                logger.info(
                    "[MYELIN] pos=%d low_feature=%s (rr=%.3f, ed=%.3f, lv=%.3e, snr=%.3f) "
                    "→ sdmul=%.2f",
                    pos + 1,
                    low_feat,
                    metrics["robust_range"],
                    metrics["edge_density"],
                    metrics["lap_var"],
                    metrics["snr"],
                    sdmul,
                )

                myelin_mask = find_foci(
                    corrected_cars_slice,
                    sigma=1.0,
                    min_distance=8,
                    min_size=300,
                    std_dev_multiplier=sdmul,
                    remove_saturated=True,
                    saturation_threshold=float(foci_params["saturation_threshold"]),  # type: ignore[index]
                    saturated_min_size=int(foci_params["saturated_min_size"]),  # type: ignore[index]
                    separate_objects=False,
                    morph_op="closing",
                    morph_radius=2,
                    debug=VERBOSE,
                )
            except Exception:
                logger.exception("Myelin detection failed; continuing with empty mask.")
                myelin_mask = np.zeros_like(corrected_cars_slice, dtype=bool)

            # Lipid-ish foci from CARS
            cars_foci_mask = find_foci(corrected_cars_slice, **foci_params)  # type: ignore[arg-type]

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
            auto_mask = np.zeros_like(corrected_cars_slice, dtype=bool)
            auto_slice: Optional[np.ndarray] = None
            if auto_ch_idx is not None:
                auto_slice = max_project_fluorescence(
                    fluoro_nd2,
                    int(auto_ch_idx),
                    pos,
                    config["morphology_params"]["autofluorescence_params"],  # type: ignore[index]
                )
                auto_mask = find_foci(auto_slice, **autofluorescence_params)  # type: ignore[arg-type]

            # LAMP2 (optional)
            lamp2_mask = None
            lamp2_ch_idx = config["channel_map"].get("LAMP2", None)  # type: ignore[index]
            if lamp2_ch_idx is not None:
                try:
                    lamp2_mip = max_project_fluorescence(
                        fluoro_nd2,
                        int(lamp2_ch_idx),
                        pos,
                        config["morphology_params"]["fluorescence_params"],  # type: ignore[index]
                    )
                    lamp2_params = config["morphology_params"].get(  # type: ignore[index]
                        "lamp2_params",
                        config["morphology_params"]["autofluorescence_params"],  # type: ignore[index]
                    )
                    lamp2_mask = find_foci(lamp2_mip, **lamp2_params)  # type: ignore[arg-type]
                except Exception:
                    logger.exception("LAMP2 detection failed; continuing without it.")
                    lamp2_mask = None

            def _colocalize_objects(
                cars_mask: np.ndarray,
                af_mask: np.ndarray,
                min_overlap: int,
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
                """
                Return labeled masks (pure_lipid, lipid+lipofuscin, pure_lipofuscin)
                using object-level colocalization with voxel-overlap >= min_overlap.
                """
                # Label both masks
                cars_labels = measure.label(cars_mask)
                af_labels = measure.label(af_mask)
            
                # Prepare output label images
                labeled_pure_lipid = np.zeros_like(cars_labels, dtype=np.int32)
                labeled_lipo_lipid = np.zeros_like(cars_labels, dtype=np.int32)
                labeled_pure_lipo  = np.zeros_like(af_labels, dtype=np.int32)
            
                # Track AF labels already consumed by colocalization to prevent double counting
                consumed_af: Set[int] = set()
            
                # Re-label counters for clean, consecutive IDs in outputs
                next_pure_lipid_id = 1
                next_lipo_lipid_id = 1
                next_pure_lipo_id  = 1
            
                # For each CARS object, decide pure lipid vs lipid+lipofuscin by overlap
                max_cid = int(cars_labels.max())
                for cid in range(1, max_cid + 1):
                    cid_mask = (cars_labels == cid)
                    if not np.any(cid_mask):
                        continue
            
                    # Which AF labels overlap this CARS object (and by how many voxels)?
                    # We can do this efficiently by looking up af_labels only where cid_mask is True.
                    af_touch = af_labels[cid_mask]
                    if af_touch.size == 0:
                        # No pixels → treat as pure lipid
                        labeled_pure_lipid[cid_mask] = next_pure_lipid_id
                        next_pure_lipid_id += 1
                        continue
            
                    # Count overlaps per AF id
                    # bincount index 0 corresponds to background, we ignore it.
                    counts = np.bincount(af_touch, minlength=int(af_labels.max()) + 1)
                    counts[0] = 0  # ignore background
                    # Find the AF object with the largest overlap
                    best_af = int(np.argmax(counts))
                    best_overlap = int(counts[best_af])
            
                    if best_af > 0 and best_overlap >= int(min_overlap):
                        # Colocalized: label CARS object as lipid+lipofuscin
                        labeled_lipo_lipid[cid_mask] = next_lipo_lipid_id
                        next_lipo_lipid_id += 1
                        # Consume that AF object so it won't be counted again as pure lipofuscin
                        consumed_af.add(best_af)
                    else:
                        # Not enough overlap with any AF object → pure lipid
                        labeled_pure_lipid[cid_mask] = next_pure_lipid_id
                        next_pure_lipid_id += 1
            
                # Any AF objects not consumed become pure lipofuscin
                max_aid = int(af_labels.max())
                for aid in range(1, max_aid + 1):
                    if aid in consumed_af:
                        continue
                    aid_mask = (af_labels == aid)
                    if not np.any(aid_mask):
                        continue
                    labeled_pure_lipo[aid_mask] = next_pure_lipo_id
                    next_pure_lipo_id += 1
            
                return labeled_pure_lipid, labeled_lipo_lipid, labeled_pure_lipo
            
            min_overlap = int(foci_params.get("min_size", 20))
            labeled_pure_lipid, labeled_lipo_lipid, labeled_pure_lipo = _colocalize_objects(
                cars_foci_mask,
                auto_mask if auto_slice is not None else np.zeros_like(cars_foci_mask, dtype=bool),
                min_overlap=min_overlap,
            )
            
            # Boolean masks for visualization and myelin exclusion
            pure_lipid_mask = (labeled_pure_lipid > 0)
            lipid_lipofuscin_mask = (labeled_lipo_lipid > 0)
            pure_lipofuscin_mask = (labeled_pure_lipo > 0)

            # Exclude any other-feature pixels from myelin % calculation
            other_features_mask = (
                pure_lipid_mask | lipid_lipofuscin_mask | pure_lipofuscin_mask
            )
            myelin_mask_refined = myelin_mask & (~other_features_mask)
            myelin_pct = (
                float(myelin_mask_refined.sum()) / float(myelin_mask_refined.size)
                if myelin_mask_refined.size
                else 0.0
            )

            # Per-marker cell masks (alias-aware; no filename gating)
            for cm in chosen_cell_markers:  # type: ignore[assignment]
                try:
                    cm_key = resolve_marker_name(cm, config)
                except KeyError:
                    continue
                cm_channel_idx = config["channel_map"].get(cm_key, None)  # type: ignore[index]
                if cm_channel_idx is None:
                    continue

                cm_slice = max_project_fluorescence(
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

                cm_mask = process_fluorescence_channel(
                    cm_slice,
                    cell_size=fluorescence_params["cell_size"],  # type: ignore[index]
                    min_size=fluorescence_params["min_size"],  # type: ignore[index]
                    closing_radius=fluorescence_params["closing_radius"],  # type: ignore[index]
                    gaussian_sigma=fluorescence_params["gaussian_sigma"],  # type: ignore[index]
                    fill_holes=fluorescence_params["fill_holes"],  # type: ignore[index]
                    threshold_method=threshold_method,
                    offset=offset_val,
                )

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
