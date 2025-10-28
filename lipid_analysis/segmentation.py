"""Segmentation utilities for fluorescence and CARS images."""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Dict, List, Set

import numpy as np
from scipy import ndimage as ndi
from skimage import feature, segmentation, measure
from skimage.measure import label
from skimage.filters import (
    gaussian,
    threshold_li,
    threshold_local,
    threshold_otsu,
    threshold_sauvola,
    threshold_triangle,
    threshold_yen,
    sobel
)
from skimage.morphology import closing, disk, opening, remove_small_objects

from .constants import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


def process_fluorescence_channel(
    image_slice: np.ndarray,
    cell_size: int,
    min_size: int,
    closing_radius: int,
    gaussian_sigma: float,
    fill_holes: bool,
    threshold_method: str,
    offset: float,
    exclude_dark_regions: bool = True,
    dark_threshold: float = 50,
    min_hole_size: int = 20_000,
    debug: bool = False,
) -> np.ndarray:
    """
    Threshold a fluorescence image and apply morphology to obtain a cell mask.

    Steps
    -----
    1) Optional Gaussian smoothing.
    2) Optionally exclude large dark regions from thresholding.
    3) Global threshold (Otsu/Li/Triangle/Yen) applied to valid pixels, then
       multiplied by `offset`.
    4) Remove small objects, morphological closing, optional hole filling.
    5) Final size filter to keep sufficiently large cell masks.

    Parameters
    ----------
    image_slice
        2D fluorescence image.
    cell_size
        Minimum area (in pixels) for the final cell mask objects.
    min_size
        Minimum area (in pixels) after initial thresholding (pre-closing).
    closing_radius
        Disk radius for the closing operation (post-cleaning).
    gaussian_sigma
        Smoothing sigma before thresholding; 0 disables.
    fill_holes
        If True, fill interior holes after closing.
    threshold_method
        One of {"otsu", "li", "triangle", "yen"} (case-insensitive).
    offset
        Multiplier applied to the base threshold.
    exclude_dark_regions
        If True, exclude large dark regions from determining the threshold.
    dark_threshold
        Pixel-value threshold used to seed dark regions.
    min_hole_size
        Minimum area for a dark region to be excluded from thresholding.
    debug
        If True, show a diagnostic figure via matplotlib.

    Returns
    -------
    np.ndarray
        Boolean mask of detected cells.
    """
    if image_slice.ndim != 2:
        raise ValueError(f"Expected a 2D array, but got shape {image_slice.shape}")

    image_slice = np.nan_to_num(image_slice)
    if gaussian_sigma and gaussian_sigma > 0:
        image_slice = gaussian(image_slice, sigma=float(gaussian_sigma), preserve_range=True)

    if exclude_dark_regions:
        preliminary_dark_mask = image_slice < dark_threshold
        labeled_dark = measure.label(preliminary_dark_mask)
        exclude_mask = np.zeros_like(labeled_dark, dtype=bool)
        for region in measure.regionprops(labeled_dark):
            if region.area >= int(min_hole_size):
                exclude_mask[tuple(region.coords.T)] = True
        valid_pixels = image_slice[~exclude_mask].ravel()
    else:
        exclude_mask = np.zeros_like(image_slice, dtype=bool)
        valid_pixels = image_slice.ravel()

    thr_m = (threshold_method or "otsu").lower()
    if valid_pixels.size > 0 and thr_m not in ("local", "sauvola"):
        if thr_m == "otsu":
            base_threshold = float(threshold_otsu(valid_pixels))
        elif thr_m == "li":
            base_threshold = float(threshold_li(valid_pixels))
        elif thr_m == "triangle":
            base_threshold = float(threshold_triangle(valid_pixels))
        elif thr_m == "yen":
            base_threshold = float(threshold_yen(valid_pixels))
        else:
            logger.warning("Unknown threshold_method '%s'; falling back to 'otsu'.", threshold_method)
            base_threshold = float(threshold_otsu(valid_pixels))
    else:
        base_threshold = float("inf")

    # --- Apply either global or local threshold ---
    thr_m = (threshold_method or "otsu").lower()
    use_local = thr_m.startswith("local")  # "local" means adaptive local mean/gaussian
    use_sauvola = (thr_m == "sauvola")

    if use_sauvola:
        # Slightly larger window; helps avoid local noise gluing neurites together
        block_size = max(75, int(2.5 * np.sqrt(max(cell_size, 1))))
        if block_size % 2 == 0:
            block_size += 1
    
        # Robust normalize to [0,1] so Sauvola's k, R behave consistently
        v = image_slice.astype(np.float32, copy=False)
        p1, p99 = np.percentile(v[~np.isnan(v)], [1, 99]) if v.size else (0.0, 1.0)
        denom = max(p99 - p1, 1e-6)
        v_norm = np.clip((v - p1) / denom, 0.0, 1.0)
    
        k = float(np.clip(offset, 0.05, 0.8))
        # Modestly smaller r makes the threshold less permissive
        local_thresh = threshold_sauvola(v_norm, window_size=block_size, k=k, r=0.35)
    
        # Compare in the normalized domain, then bring mask back
        binary_mask = v_norm > local_thresh
    elif use_local:
        # Make window scale with expected object size (odd number)
        # Larger window => more conservative (less sensitive to tiny fluctuations).
        block_size = int(2 * np.sqrt(cell_size))
        if block_size % 2 == 0:
            block_size += 1
    
        # Interpret 'offset' the same way as the global path:
        # larger offset => *more conservative* (higher effective threshold).
        # threshold_local uses (local_mean - offset), so to RAISE the threshold
        # we must pass a NEGATIVE value.
        if valid_pixels.size > 0:
            p1, p99 = np.percentile(valid_pixels, [1, 99])
            robust_scale = max((p99 - p1) / 16.0, 1.0)
        else:
            robust_scale = 10.0
        
        # Map user 'offset' to a negative bias so larger values are stricter.
        # Example: offset=1.0 -> -1.0*scale (stricter than 0), offset=0.5 -> -0.5*scale (mild),
        # offset=2.0 -> -2.0*scale (much stricter).
        local_offset = -abs(float(offset)) * robust_scale
        
        local_thresh = threshold_local(image_slice, block_size, method="gaussian", offset=local_offset)
        binary_mask = image_slice > local_thresh

    else:
        final_threshold = base_threshold * float(offset)
        binary_mask = image_slice > final_threshold
    
    binary_mask[exclude_mask] = False

    cleaned_mask = remove_small_objects(binary_mask, min_size=int(min_size))
    binary_closed = closing(cleaned_mask, disk(int(closing_radius)))
    if fill_holes:
        binary_closed = ndi.binary_fill_holes(binary_closed)

    cell_mask = remove_small_objects(binary_closed, min_size=int(cell_size))

    if debug:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        axes[0].imshow(image_slice, cmap="gray")
        axes[0].set_title("Raw Fluorescence")
        if use_sauvola:
            thr_label = f"sauvola (k={k:.2f}, block={block_size})"
        elif use_local:
            thr_label = f"local (block={block_size}, off≈{local_offset:.2f})"
        else:
            thr_label = f"> {final_threshold:.2f}"
        axes[1].imshow(binary_mask, cmap="gray")
        axes[1].set_title(f"Thresholded ({thr_label})")
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


def process_fluorescence_stack(
    stack_3d: np.ndarray,
    *,
    cell_size: int,
    min_size: int,
    closing_radius: int,
    gaussian_sigma: float,
    fill_holes: bool,
    threshold_method: str,
    offset: float,
    exclude_dark_regions: bool = True,
    dark_threshold: float = 50,
    min_hole_size: int = 20_000,
    min_voxels_3d: Optional[int] = None,
    debug: bool = False,
    # NEW: optional fallback knobs (all disabled by default)
    bad_slice_frac_threshold: Optional[float] = None,
    bad_slice_max_components: Optional[int] = None,
    bad_slice_use_mip_if_fraction_over: Optional[float] = None,
    clip_to_mip_mask: bool = False,
) -> np.ndarray:
    """
    Build a 3-D cell mask by applying `process_fluorescence_channel` to each z-slice
    of a (Z,H,W) fluorescence stack, then optional 3-D small-object removal.
    """
    if stack_3d.ndim != 3:
        raise ValueError(f"Expected a (Z,H,W) stack, got shape {stack_3d.shape}")

    masks = []
    per_slice_fracs: list[float] = []
    per_slice_comps: list[int] = []
    for z in range(stack_3d.shape[0]):
        mz = process_fluorescence_channel(
            stack_3d[z],
            cell_size=cell_size,
            min_size=min_size,
            closing_radius=closing_radius,
            gaussian_sigma=gaussian_sigma,
            fill_holes=fill_holes,
            threshold_method=threshold_method,
            offset=offset,
            exclude_dark_regions=exclude_dark_regions,
            dark_threshold=dark_threshold,
            min_hole_size=min_hole_size,
            debug=(debug and z == 0),
        )
        mz = mz.astype(bool, copy=False)
        masks.append(mz)
        # badness metrics (only if thresholds provided later)
        frac = float(np.count_nonzero(mz)) / float(mz.size) if mz.size else 0.0
        per_slice_fracs.append(frac)
        # component count to catch explosions
        labeled = measure.label(mz, connectivity=1)
        per_slice_comps.append(int(labeled.max()))
    m3d = np.stack(masks, axis=0)

    # --- NEW: Pathological-slice fallback using a robust MIP mask ---
    use_frac = (bad_slice_frac_threshold is not None)
    use_comp = (bad_slice_max_components is not None)
    use_many = (bad_slice_use_mip_if_fraction_over is not None)
    if use_frac or use_comp or clip_to_mip_mask:
        # Build MIP mask once with the same params
        mip = np.max(stack_3d, axis=0) if stack_3d.size else np.zeros_like(stack_3d[0])
        mip_mask = process_fluorescence_channel(
            mip,
            cell_size=cell_size,
            min_size=min_size,
            closing_radius=closing_radius,
            gaussian_sigma=gaussian_sigma,
            fill_holes=fill_holes,
            threshold_method=threshold_method,
            offset=offset,
            exclude_dark_regions=exclude_dark_regions,
            dark_threshold=dark_threshold,
            min_hole_size=min_hole_size,
            debug=False,
        ).astype(bool, copy=False)

        bad_idx: np.ndarray | None = None
        if use_frac or use_comp:
            import numpy as _np
            bad_by_frac = _np.array(per_slice_fracs) >= (bad_slice_frac_threshold or 1.1)  # off if None
            bad_by_comp = _np.array(per_slice_comps) >= (bad_slice_max_components or _np.iinfo(_np.int32).max)
            bad_idx = _np.where(bad_by_frac | bad_by_comp)[0]

            if bad_idx.size > 0:
                if use_many and (bad_idx.size / m3d.shape[0]) >= float(bad_slice_use_mip_if_fraction_over):
                    m3d[:, :, :] = mip_mask[None, :, :]
                else:
                    m3d[bad_idx, :, :] = mip_mask[None, :, :]

        if clip_to_mip_mask:
            m3d = m3d & mip_mask[None, :, :]

    if min_voxels_3d and min_voxels_3d > 1:
        m3d = remove_small_objects(m3d, min_size=int(min_voxels_3d), connectivity=1)
    return m3d


def robust_mad(a: np.ndarray) -> float:
    """Median absolute deviation (MAD) of a 1D array."""
    med = float(np.median(a))
    return float(np.median(np.abs(a - med)))


def find_foci(
    image_slice: np.ndarray,
    sigma: float,
    min_distance: int,
    min_size: int,
    std_dev_multiplier: float,
    remove_saturated: bool,
    saturation_threshold: float,
    saturated_min_size: int,
    separate_objects: bool = True,
    morph_op: str = "opening",
    morph_radius: int = 3,
    min_snr: Optional[float] = None,
    debug: bool = False,
) -> np.ndarray:
    """
    Detect foci using robust global thresholding and optional watershed splitting.

    Steps
    -----
    1) Optional smoothing with Gaussian(sigma).
    2) Exclude saturated objects from threshold estimation.
    3) Robust global threshold: median + std_dev_multiplier * (1.4826 * MAD).
    4) Optional early exit if approximate SNR < min_snr.
    5) Morphological regularization (opening/closing).
    6) Split touching objects with watershed if `separate_objects=True`.
    7) Area filter by `min_size`.
    8) Reinstate saturated objects into the final mask.

    Parameters
    ----------
    image_slice
        2D image.
    sigma
        Gaussian sigma for pre-smoothing (0 disables).
    min_distance
        Minimum peak separation (pixels) for seed detection.
    min_size
        Minimum area (pixels) for final objects.
    std_dev_multiplier
        Multiplier on approx. std (1.4826*MAD) for thresholding.
    remove_saturated
        If True, exclude saturated objects from threshold estimation.
    saturation_threshold
        Pixel value regarded as saturated for exclusion.
    saturated_min_size
        Minimum area for an object to be considered saturated.
    separate_objects
        If True, split touching objects via watershed.
    morph_op
        "opening" (default), "closing", or "none".
    morph_radius
        Disk radius for morphological operation.
    min_snr
        If provided, return an empty mask when approx SNR < min_snr.
    debug
        If True, emit debug logs.

    Returns
    -------
    np.ndarray
        Boolean mask of detected foci.
    """
    if image_slice.ndim != 2:
        raise ValueError(f"find_foci expects a 2D array, got shape {image_slice.shape}")

    image_slice = np.nan_to_num(image_slice)

    # Exclude saturated pixels for thresholding only
    exclude_mask = np.zeros_like(image_slice, dtype=bool)
    if remove_saturated:
        labeled_sat = measure.label(image_slice > saturation_threshold)
        for region in measure.regionprops(labeled_sat):
            if region.area >= int(saturated_min_size):
                exclude_mask[tuple(region.coords.T)] = True

    smoothed = (
        gaussian(image_slice, sigma=float(sigma), preserve_range=True)
        if sigma > 0
        else image_slice.copy()
    )

    valid_pixels = smoothed[~exclude_mask].ravel()
    if valid_pixels.size > 0:
        median_val = float(np.median(valid_pixels))
        mad_val = float(robust_mad(valid_pixels))
        approx_std = 1.4826 * mad_val
        threshold_val = median_val + (float(std_dev_multiplier) * approx_std)

        if debug:
            snr_dbg = approx_std / max(median_val, 1e-9)
            logger.debug(
                "[FOCI] median=%.4f, std≈%.4f, SNR=%.4f, threshold=%.4f",
                median_val,
                approx_std,
                snr_dbg,
                threshold_val,
            )

        # Optional early low-contrast guard
        if min_snr is not None:
            snr = approx_std / max(median_val, 1e-9)
            if snr < float(min_snr):
                if debug:
                    logger.debug("[FOCI] Low-contrast frame: SNR=%.4f < %.4f → empty", snr, min_snr)
                return np.zeros_like(image_slice, dtype=bool)
    else:
        threshold_val = float("inf")

    mask_std = smoothed > threshold_val

    # Morphological regularization
    work = mask_std.copy()
    if morph_op == "opening" and morph_radius > 0:
        work = opening(work, disk(int(morph_radius)))
    elif morph_op == "closing" and morph_radius > 0:
        work = closing(work, disk(int(morph_radius)))
    # else "none": leave as-is

    if separate_objects:
        # Blob mode: split touching objects with watershed
        distance = ndi.distance_transform_edt(work)
        local_maxi_coords = feature.peak_local_max(
            smoothed, min_distance=int(min_distance), labels=work
        )
        local_maxi = np.zeros_like(work, dtype=bool)
        if local_maxi_coords.size:
            local_maxi[tuple(local_maxi_coords.T)] = True
        markers = ndi.label(local_maxi)[0]
        labels_ws = segmentation.watershed(-distance, markers, mask=work)
    else:
        # Filament mode: connected components (avoid over-segmentation)
        labels_ws = measure.label(work)

    final_mask = np.zeros_like(labels_ws, dtype=bool)
    for region in measure.regionprops(labels_ws):
        if region.area >= int(min_size):
            final_mask[tuple(region.coords.T)] = True

    # Reinstate saturated objects so they are not lost
    final_mask[exclude_mask] = True

    if debug:
        logger.debug(
            "[FOCI] thr=%.3f | morph_op=%s r=%d | separate=%s | kept_px=%d",
            threshold_val,
            morph_op,
            int(morph_radius),
            separate_objects,
            int(final_mask.sum()),
        )

    return final_mask


def assess_low_feature(
    img2d: np.ndarray,
    *,
    sigma: float,
    remove_saturated: bool,
    sat_thresh: float,
    sat_min: int,
    rr_thresh: float = 0.12,
    edge_q: float = 1.5,
    edge_min_frac: float = 0.16,
    lap_var_thresh: float = 4.8e-3,
    snr_thresh: Optional[float] = 0.285,
    debug: bool = False,
) -> Tuple[bool, dict]:
    """
    Heuristically decide if the image is "low-feature" (flat/poor contrast).

    Cues after smoothing & saturation exclusion (mirrors find_foci preproc):
      - robust_range = (p99 - p1) / median
      - edge_density = fraction of Sobel magnitudes > med + edge_q * 1.4826*MAD
      - lap_var      = var(laplace(img / median))
      - snr          = approx_std / median  (approx_std from 1.4826 * MAD)

    Flags low-feature if at least 2 (of up to 4) cues fail thresholds.
    """
    im = np.nan_to_num(img2d.astype(np.float32), copy=False)
    if sigma > 0:
        im = gaussian(im, sigma=sigma, preserve_range=True)

    # Exclude saturated regions from stats
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
        return True, {"robust_range": 0.0, "edge_density": 0.0, "lap_var": 0.0, "snr": 0.0}

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

    # SNR (same definition as in find_foci logging)
    mad = float(np.median(np.abs(vp - med)))
    approx_std = 1.4826 * mad
    snr = approx_std / max(med, 1e-9)

    # Two-(of up to four)-fails rule
    fails = (
        int(robust_range < rr_thresh)
        + int(edge_density < edge_min_frac)
        + int(lap_var < lap_var_thresh)
        + (int(snr < snr_thresh) if snr_thresh is not None else 0)
    )
    low_feature = (fails >= 2)

    if debug:
        logger.debug(
            "[LOWFEAT] rr=%.4f (<%.3f) | ed=%.4f (<%.3f) | lapvar=%.4e (<%.3e) | snr=%.3f (<%s) → low=%s",
            robust_range, rr_thresh,
            edge_density, edge_min_frac,
            lap_var, lap_var_thresh,
            snr, f"{snr_thresh:.3f}" if snr_thresh is not None else "—",
            low_feature,
        )

    return low_feature, {
        "robust_range": robust_range,
        "edge_density": edge_density,
        "lap_var": lap_var,
        "snr": snr,
    }


def colocalize_objects_3d(
    cars_mask_3d: np.ndarray,
    af_mask_3d: np.ndarray,
    *,
    min_overlap: int,
    af_multi_min_count: int = 3,
    af_cover_frac: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Label and colocalize in 3-D.
    Returns: (labeled_pure_lipid_3d, labeled_lipo_lipid_3d, labeled_pure_lipo_3d).

    Extended logic identical to the 2-D path:
    - If one AF component touches many CARS components (>= af_multi_min_count)
      or the total overlapped voxels cover >= af_cover_frac of the AF volume,
      assign *all* touching CARS to lipidated lipofuscin (no size gate).
    - Otherwise, fall back to one-to-one pairing with the original size-ratio gate.
    """
    cars_labels = label(cars_mask_3d, connectivity=2)
    af_labels   = label(af_mask_3d,   connectivity=2)

    labeled_pure_lipid = np.zeros_like(cars_labels, dtype=np.int32)
    labeled_lipo_lipid = np.zeros_like(cars_labels, dtype=np.int32)
    labeled_pure_lipo  = np.zeros_like(af_labels,   dtype=np.int32)

    max_cid = int(cars_labels.max())
    max_aid = int(af_labels.max())

    if max_cid == 0 and max_aid == 0:
        return labeled_pure_lipid, labeled_lipo_lipid, labeled_pure_lipo

    cars_sizes = np.bincount(cars_labels.ravel(), minlength=max_cid + 1)
    af_sizes   = np.bincount(af_labels.ravel(),   minlength=max_aid + 1)

    # Build AF -> list of (cid, overlap)
    af_to_cids: Dict[int, List[Tuple[int, int]]] = {aid: [] for aid in range(1, max_aid + 1)}
    for cid in range(1, max_cid + 1):
        cid_mask = (cars_labels == cid)
        if not np.any(cid_mask):
            continue
        af_touch = af_labels[cid_mask]
        if af_touch.size == 0:
            continue
        counts = np.bincount(af_touch, minlength=max_aid + 1)
        counts[0] = 0
        for aid in np.nonzero(counts >= int(min_overlap))[0]:
            af_to_cids[int(aid)].append((cid, int(counts[int(aid)])))

    consumed_af: Set[int] = set()
    assigned_cids: Set[int] = set()
    next_pure_lipid_id = 1
    next_lipo_lipid_id = 1
    next_pure_lipo_id  = 1

    # Pass 1: multi-pair rule
    for aid, pairs in af_to_cids.items():
        if not pairs:
            continue
        total_overlap = sum(ov for _, ov in pairs)
        n_touching = len(pairs)
        size_af = int(af_sizes[aid])
        if (n_touching >= af_multi_min_count) or (size_af > 0 and (total_overlap / size_af) >= af_cover_frac):
            for cid, _ in pairs:
                if cid in assigned_cids:
                    continue
                labeled_lipo_lipid[cars_labels == cid] = next_lipo_lipid_id
                next_lipo_lipid_id += 1
                assigned_cids.add(cid)
            consumed_af.add(aid)

    # Pass 2: standard size-ratio-gated matching
    for cid in range(1, max_cid + 1):
        if cid in assigned_cids:
            continue
        cid_mask = (cars_labels == cid)
        if not np.any(cid_mask):
            continue
        size_lipid = int(cars_sizes[cid])
        af_touch = af_labels[cid_mask]
        if af_touch.size == 0:
            labeled_pure_lipid[cid_mask] = next_pure_lipid_id
            next_pure_lipid_id += 1
            continue
        counts = np.bincount(af_touch, minlength=max_aid + 1)
        counts[0] = 0
        best_af = int(np.argmax(counts))
        best_overlap = int(counts[best_af])
        if best_af > 0 and best_overlap >= int(min_overlap):
            size_af = int(af_sizes[best_af])
            if (size_af >= 0.5 * size_lipid) and (size_af <= 2.0 * size_lipid):
                labeled_lipo_lipid[cid_mask] = next_lipo_lipid_id
                next_lipo_lipid_id += 1
                consumed_af.add(best_af)
            else:
                labeled_pure_lipid[cid_mask] = next_pure_lipid_id
                next_pure_lipid_id += 1
        else:
            labeled_pure_lipid[cid_mask] = next_pure_lipid_id
            next_pure_lipid_id += 1

    for aid in range(1, max_aid + 1):
        if aid in consumed_af:
            continue
        aid_mask = (af_labels == aid)
        if np.any(aid_mask):
            labeled_pure_lipo[aid_mask] = next_pure_lipo_id
            next_pure_lipo_id += 1

    return labeled_pure_lipid, labeled_lipo_lipid, labeled_pure_lipo
