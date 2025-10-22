"""Hyperspectral analysis: mapping, masking, spectra averaging, peak fitting, and QC."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Set

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from nd2reader import ND2Reader
from skimage import measure
from skimage.morphology import dilation, disk

from .config_utils import resolve_marker_name
from .constants import CARS_CH, LOG_LEVEL, PEAKFIT_DEBUG, VERBOSE
from .debug_utils import save_alignment_triptych
from .filters import apply_east_shadows_filter
from .segmentation import find_foci, process_fluorescence_channel
from .peakfit import fit_cars_peaks, _plot_peak_fit_debug

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

# This is intentionally set from the CLI before calling module functions.
# e.g. in cli.py:  import lipid_analysis.hyperspec as hyperspec; hyperspec.config = config
config: Optional[Dict[str, Any]] = None


# --- Glitch repair helpers ----------------------------------------------------
def _repair_zero_glitches(
    y: np.ndarray, z_abs: float = 1e-9, z_rel: float = 0.02, win: int = 3
) -> np.ndarray:
    """
    Linearly interpolate random zero/near-zero points in a 1D spectrum.

    A point is considered a glitch if it is <= z_abs OR <= z_rel * local_median,
    where the local median is computed in a ±win neighborhood (excluding self).
    """
    y = np.asarray(y, dtype=float).copy()
    n = y.size
    if n == 0:
        return y

    # Local median of neighbors for each index
    med = np.empty_like(y)
    for i in range(n):
        lo = max(0, i - win)
        hi = min(n, i + win + 1)
        neigh = np.concatenate([y[lo:i], y[i + 1 : hi]])  # exclude self
        med[i] = np.median(neigh) if neigh.size else y[i]

    glitch = (y <= z_abs) | (y <= (z_rel * np.maximum(med, z_abs)))
    if not np.any(glitch):
        return y

    # Interpolate over glitches
    x = np.arange(n, dtype=float)
    ok = ~glitch
    if ok.sum() >= 2:
        y[glitch] = np.interp(x[glitch], x[ok], y[ok])
    else:
        # Fallback: if almost everything is bad, fill with median
        y[glitch] = np.nanmedian(y[ok]) if ok.any() else 0.0
    return y


def _normalize_row_max(y: np.ndarray, eps: float = 1e-9) -> tuple[np.ndarray, float]:
    """Normalize a 1D array by its (nan-safe) max; returns (normalized, scale)."""
    y = np.asarray(y, dtype=float)
    y_clean = np.nan_to_num(y, nan=0.0)
    m = float(np.max(y_clean))
    m = m if m > eps else 1.0
    return y_clean / m, m


def _colocalize_objects(
    cars_mask: np.ndarray,
    af_mask: np.ndarray,
    min_overlap: int,
    *,
    af_multi_min_count: int = 3,
    af_cover_frac: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return labeled masks (pure_lipid, lipid+lipofuscin, pure_lipofuscin)
    using object-level colocalization with voxel-overlap >= min_overlap.

    Extended logic:
    - If a single AF object touches >= `af_multi_min_count` CARS objects
      (each meeting `min_overlap`) OR the union of overlaps covers at least
      `af_cover_frac` of the AF object's voxels, we pair *all* touching CARS
      objects to that AF object (lipidated lipofuscin) and consume the AF object,
      **without** enforcing the size-ratio gate. This handles merged AF granules.
    - Otherwise we fall back to one-to-one pairing with the original size-ratio gate.
    """
    cars_labels = measure.label(cars_mask)
    af_labels   = measure.label(af_mask)

    labeled_pure_lipid = np.zeros_like(cars_labels, dtype=np.int32)
    labeled_lipo_lipid = np.zeros_like(cars_labels, dtype=np.int32)
    labeled_pure_lipo  = np.zeros_like(af_labels,   dtype=np.int32)

    # Sizes
    max_cid = int(cars_labels.max())
    max_aid = int(af_labels.max())
    if max_cid == 0 and max_aid == 0:
        return labeled_pure_lipid, labeled_lipo_lipid, labeled_pure_lipo

    cars_sizes = np.bincount(cars_labels.ravel(), minlength=max_cid + 1)
    af_sizes   = np.bincount(af_labels.ravel(),   minlength=max_aid + 1)

    # Build AF -> list of (cid, overlap) for cid that touch this AF with >= min_overlap
    af_to_cids: Dict[int, List[Tuple[int, int]]] = {aid: [] for aid in range(1, max_aid + 1)}
    for cid in range(1, max_cid + 1):
        cid_mask = (cars_labels == cid)
        if not cid_mask.any():
            continue
        af_touch = af_labels[cid_mask]
        if af_touch.size == 0:
            continue
        counts = np.bincount(af_touch, minlength=max_aid + 1)
        counts[0] = 0
        for aid in np.nonzero(counts >= int(min_overlap))[0]:
            af_to_cids[int(aid)].append((cid, int(counts[int(aid)])))

    consumed_af: Set[int] = set()
    next_pure_lipid_id = 1
    next_lipo_lipid_id = 1
    next_pure_lipo_id  = 1

    assigned_cids: Set[int] = set()

    # Pass 1: multi-pair rule for merged AF objects
    for aid, pairs in af_to_cids.items():
        if not pairs:
            continue
        total_overlap = sum(ov for _, ov in pairs)
        n_touching = len(pairs)
        size_af = int(af_sizes[aid])
        if (n_touching >= af_multi_min_count) or (size_af > 0 and (total_overlap / size_af) >= af_cover_frac):
            # Assign all touching CARS objects as lipidated lipofuscin (no size gate)
            for cid, _ in pairs:
                if cid in assigned_cids:
                    continue
                labeled_lipo_lipid[cars_labels == cid] = next_lipo_lipid_id
                next_lipo_lipid_id += 1
                assigned_cids.add(cid)
            consumed_af.add(aid)

    # Pass 2: standard best-match with size-ratio gate for remaining CARS objects
    for cid in range(1, max_cid + 1):
        if cid in assigned_cids:
            continue
        cid_mask = (cars_labels == cid)
        if not cid_mask.any():
            continue
        size_lipid = int(cars_sizes[cid])
        af_touch = af_labels[cid_mask]
        if af_touch.size == 0:
            labeled_pure_lipid[cid_mask] = next_pure_lipid_id
            next_pure_lipid_id += 1
            continue
        counts = np.bincount(af_touch, minlength=max_aid + 1)
        counts[0] = 0
        best_aid = int(np.argmax(counts))
        best_overlap = int(counts[best_aid])
        if best_aid > 0 and best_overlap >= int(min_overlap):
            size_af = int(af_sizes[best_aid])
            if (size_af >= 0.5 * size_lipid) and (size_af <= 2.0 * size_lipid):
                labeled_lipo_lipid[cid_mask] = next_lipo_lipid_id
                next_lipo_lipid_id += 1
                consumed_af.add(best_aid)
            else:
                labeled_pure_lipid[cid_mask] = next_pure_lipid_id
                next_pure_lipid_id += 1
        else:
            labeled_pure_lipid[cid_mask] = next_pure_lipid_id
            next_pure_lipid_id += 1

    # Pass 3: any AF not consumed becomes pure lipofuscin
    for aid in range(1, max_aid + 1):
        if aid in consumed_af:
            continue
        aid_mask = (af_labels == aid)
        if aid_mask.any():
            labeled_pure_lipo[aid_mask] = next_pure_lipo_id
            next_pure_lipo_id += 1

    return labeled_pure_lipid, labeled_lipo_lipid, labeled_pure_lipo


def _save_labeled_mask_images(
    labeled_mask: np.ndarray, base_gray: np.ndarray, out_dir: str
) -> tuple[Optional[str], Optional[str]]:
    """
    Save two PNGs into `out_dir`:
      (1) a color-by-label image with numeric IDs drawn at region centroids
      (2) an overlay of the labels on a grayscale background image
    """
    import matplotlib
    import matplotlib.colors as mcolors
    from skimage.segmentation import find_boundaries

    H, W = labeled_mask.shape
    n = int(labeled_mask.max())
    if n == 0:
        return None, None

    # Label color image
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    nz = labeled_mask > 0
    norm = mcolors.Normalize(vmin=1, vmax=n)
    rgba = matplotlib.colormaps.get_cmap("turbo")(norm(labeled_mask[nz]))
    rgb[nz] = (rgba[:, :3] * 255).astype(np.uint8)

    # White boundaries
    borders = find_boundaries(labeled_mask, mode="outer")
    rgb[borders] = [255, 255, 255]

    # Draw numeric IDs
    img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for r in measure.regionprops(labeled_mask):
        y, x = (int(round(r.centroid[0])), int(round(r.centroid[1])))
        txt = str(r.label)
        (tw, th), _ = cv2.getTextSize(txt, font, 0.5, 1)
        x0, y0 = max(0, x - 1), max(0, y - th - 2)
        x1, y1 = min(W - 1, x + tw + 1), min(H - 1, y + 2)
        cv2.rectangle(img_bgr, (x0, y0), (x1, y1), (0, 0, 0), -1)
        cv2.putText(img_bgr, txt, (x, y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    out_path_mask = os.path.join(out_dir, "Hyperspec_LabeledObjects.png")
    cv2.imwrite(out_path_mask, img_bgr)

    # Overlay on grayscale context
    base = base_gray.astype(np.float32)
    bmax = base.max() if base.max() > 0 else 1.0
    base8 = (base / bmax * 255).astype(np.uint8)
    base_rgb = cv2.cvtColor(base8, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base_rgb, 0.6, img_bgr, 0.8, 0)
    out_path_overlay = os.path.join(out_dir, "Hyperspec_LabeledObjects_overlay.png")
    cv2.imwrite(out_path_overlay, overlay)
    return out_path_mask, out_path_overlay


def save_batch_peak_summary(
    peak_df: Optional[pd.DataFrame],
    lipid_df_norm: pd.DataFrame,
    wavenumbers: List[float],
    out_png_path: str,
) -> None:
    """Create and save a 2×2 summary figure for one hyperspectral batch."""
    spec_cols = [
        c for c in lipid_df_norm.columns if c.replace(".", "", 1).replace("-", "", 1).isdigit()
    ]
    Y = lipid_df_norm[spec_cols].to_numpy(dtype=float)
    if Y.size == 0:
        Y = np.zeros((1, len(wavenumbers)), dtype=float)
    mean_spec = np.nanmean(Y, axis=0)
    std_spec = np.nanstd(Y, axis=0)

    if peak_df is None or peak_df.empty:
        centers_by_peak = {k: [] for k in range(1, 8)}
        amps_by_peak = {k: [] for k in range(1, 8)}
        fit_success_pct = 0.0
        n_droplets_fit = 0
        n_droplets_total = 0
    else:
        succ_per_lipid = peak_df.groupby("Lipid ID")["FitSuccess"].max()
        n_droplets_total = succ_per_lipid.shape[0]
        n_droplets_fit = int(succ_per_lipid.sum())
        fit_success_pct = 100.0 * n_droplets_fit / n_droplets_total if n_droplets_total > 0 else 0.0
        centers_by_peak = {
            k: peak_df.loc[peak_df["Peak"] == k, "Center_cm^-1"].dropna().to_list()
            for k in range(1, 8)
        }
        amps_by_peak = {
            k: peak_df.loc[peak_df["Peak"] == k, "Amplitude"].dropna().to_list()
            for k in range(1, 8)
        }

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)

    # (A) Mean ±1 SD normalized spectrum
    axA = fig.add_subplot(gs[0, 0])
    axA.plot(wavenumbers, mean_spec, lw=2)
    axA.fill_between(wavenumbers, mean_spec - std_spec, mean_spec + std_spec, alpha=0.25)
    axA.set_title("Mean ±1 SD (Normalized Spectra)")
    axA.set_xlabel("Raman shift (cm$^{-1}$)")
    axA.set_ylabel("Normalized intensity")
    axA.grid(alpha=0.3)

    # (B) Fitted centers
    axB = fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(0)
    for k in range(1, 8):
        xs = np.full(len(centers_by_peak[k]), k, dtype=float) + rng.normal(0, 0.05, size=len(centers_by_peak[k]))
        axB.scatter(xs, centers_by_peak[k], s=12)
        if len(centers_by_peak[k]) > 0:
            med = np.median(centers_by_peak[k])
            axB.plot([k - 0.3, k + 0.3], [med, med], lw=2)
    axB.set_xticks(range(1, 8))
    axB.set_xlabel("Peak index")
    axB.set_ylabel("Fitted center (cm$^{-1}$)")
    axB.set_title("Fitted Peak Centers")
    axB.grid(axis="y", alpha=0.3)

    # (C) Amplitudes
    axC = fig.add_subplot(gs[1, 0])
    data_amp = [amps_by_peak[k] if len(amps_by_peak[k]) > 0 else [np.nan] for k in range(1, 8)]
    axC.boxplot(data_amp, showfliers=False)
    axC.set_xticks(range(1, 8))
    axC.set_xlabel("Peak index")
    axC.set_ylabel("Amplitude (arb. units)")
    axC.set_title("Fitted Amplitudes")
    axC.grid(axis="y", alpha=0.3)

    # (D) Fit success
    axD = fig.add_subplot(gs[1, 1])
    axD.bar([0], [fit_success_pct], width=0.6)
    axD.set_ylim(0, 100)
    axD.set_xticks([0])
    axD.set_xticklabels(["Fit success"])
    axD.set_ylabel("Percent of droplets (%)")
    axD.set_title(f"Fit Success: {fit_success_pct:.1f}%  (n={n_droplets_fit}/{n_droplets_total})")
    for spine in ["top", "right"]:
        axD.spines[spine].set_visible(False)

    fig.suptitle("Hyperspectral Peak-Fit Summary", y=0.98, fontsize=14)
    plt.savefig(out_png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved batch summary figure to: %s", out_png_path)


def visualize_hyperspectral_mask_overlay(cars_image: np.ndarray, lipid_mask: np.ndarray) -> None:
    """Show grayscale CARS, lipid mask (yellow), and an overlay."""
    def create_rgb_mask(mask: np.ndarray, color: list[int]) -> np.ndarray:
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for i in range(3):
            rgb[..., i] = mask.astype(np.uint8) * color[i]
        return rgb

    max_val = cars_image.max() if cars_image.max() > 0 else 1
    grayscale_8bit = (cars_image / max_val * 255).astype(np.uint8)
    lipid_mask_rgb = create_rgb_mask(lipid_mask, [255, 255, 0])
    grayscale_rgb = np.stack([grayscale_8bit] * 3, axis=-1)
    overlay_rgb = np.clip(0.5 * grayscale_rgb + 0.5 * lipid_mask_rgb, 0, 255).astype(np.uint8)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(grayscale_8bit, cmap="gray")
    axs[0].set_title("CARS (9th Hyperspectral Image)")
    axs[0].axis("off")

    axs[1].imshow(lipid_mask_rgb)
    axs[1].set_title("Lipid Mask (Yellow)")
    axs[1].axis("off")

    axs[2].imshow(overlay_rgb)
    axs[2].set_title("Overlay")
    axs[2].axis("off")

    plt.show()
    plt.close(fig)


def infer_hyperspectral_mapping(spectrum_folder: str, cfg: Dict[str, Any]) -> Dict[str, str]:
    """
    Infer {cars_nd2, fluor_nd2, cell_marker} for a hyperspectral folder using
    folder-name tokens and actual files present in cfg['paths']['data_directory'].
    """
    from .filepairing import parse_nd2_filename

    folder_base = os.path.basename(spectrum_folder)

    # Extract sample token from folder name (e.g., AD44, AD33, CTRL/HC)
    sample_token: Optional[str] = None
    try:
        m = re.search(r"\b(AD\d{2}|CTRL|CONTROL|HC)\b", folder_base, flags=re.IGNORECASE)
        if m:
            sample_token = m.group(1).upper()
    except Exception:
        pass

    name_l = folder_base.lower()

    folder_map = cfg.get("hyperspectral_folder_map", {})
    if not folder_map:
        raise ValueError("config['hyperspectral_folder_map'] is missing or empty.")

    mag_kw = cfg["file_keywords"]["magnification_keyword"]

    # 1) Match token
    matched: Optional[Tuple[str, str, List[str]]] = None
    for token, spec in folder_map.items():
        if token in name_l:
            matched = (token, spec["label"], list(spec["markers"]))
            break
    if matched is None:
        raise ValueError(
            f"Cannot infer cell type from folder '{folder_base}'. "
            f"Expected one of: {', '.join(folder_map.keys())}"
        )
    _, target_stacks_label, marker_priority = matched

    # 2) Prefix hint from ND2 inside the spectrum folder
    nd2_inside = [f for f in os.listdir(spectrum_folder) if f.lower().endswith(".nd2")]
    inferred_prefixes: List[str] = []
    for f in nd2_inside:
        try:
            meta = parse_nd2_filename(f, cfg)
            if meta["prefix"]:
                inferred_prefixes.append(meta["prefix"])
        except Exception:
            pass
    prefix_hint: Optional[str] = None
    if inferred_prefixes:
        from collections import Counter

        prefix_hint = Counter(inferred_prefixes).most_common(1)[0][0]

    def stacks_label_matches(meta_label: Optional[str], target_label: str) -> bool:
        """Treat empty/None label as wildcard."""
        return (meta_label is None) or (meta_label == "") or (meta_label == target_label)

    # If prefix_hint conflicts with sample_token, ignore the prefix_hint
    try:
        if prefix_hint and sample_token and sample_token not in str(prefix_hint).upper():
            if VERBOSE:
                logger.info(
                    "[HYPERMAP] Ignoring prefix_hint '%s' (mismatch with sample token '%s').",
                    prefix_hint,
                    sample_token,
                )
            prefix_hint = None
    except Exception:
        pass

    # 3) Scan main data directory for ND2s with the right stacks label
    data_dir = cfg["paths"]["data_directory"]
    cars_candidates: List[Tuple[str, Dict[str, Any]]] = []
    fluor_candidates: List[Tuple[str, Dict[str, Any]]] = []
    for item in os.listdir(data_dir):
        if not item.lower().endswith(".nd2"):
            continue
        if "largearea" in item.lower():
            continue
        meta = parse_nd2_filename(item, cfg)
        if not stacks_label_matches(meta["stacks_label"], target_stacks_label):
            continue
        if meta["contains_cars"]:
            cars_candidates.append((item, meta))
        else:
            fluor_candidates.append((item, meta))

    # 4) Relaxed pass (parity with original)
    if not cars_candidates or not fluor_candidates:
        cars_candidates, fluor_candidates = [], []
        for item in os.listdir(data_dir):
            if not item.lower().endswith(".nd2"):
                continue
            if "largearea" in item.lower():
                continue
            meta = parse_nd2_filename(item, cfg)
            if not stacks_label_matches(meta["stacks_label"], target_stacks_label):
                continue
            if meta["contains_cars"]:
                cars_candidates.append((item, meta))
            else:
                fluor_candidates.append((item, meta))

    if not cars_candidates:
        raise FileNotFoundError(
            f"No CARS file found in '{data_dir}' for stacks '{target_stacks_label}'."
        )
    if not fluor_candidates:
        raise FileNotFoundError(
            f"No fluorescence file found in '{data_dir}' for stacks '{target_stacks_label}'."
        )

    # Prefer candidates whose names/prefixes contain the sample token (e.g., AD44)
    def _filter_by_sample_token(cands: List[Tuple[str, Dict[str, Any]]]) -> List[Tuple[str, Dict[str, Any]]]:
        if not sample_token:
            return cands
        filtered = []
        for n, m in cands:
            name_u = n.upper()
            pref_u = str(m.get("prefix") or "").upper()
            if sample_token in name_u or sample_token in pref_u:
                filtered.append((n, m))
        return filtered or cands  # graceful fallback

    cars_candidates = _filter_by_sample_token(cars_candidates)
    fluor_candidates = _filter_by_sample_token(fluor_candidates)

    # Optional strict mode: require token match if enabled
    if cfg.get("strict_sample_match", False) and sample_token:

        def _has_token(cands: List[Tuple[str, Dict[str, Any]]]) -> bool:
            for n, m in cands:
                name_u = n.upper()
                pref_u = str(m.get("prefix") or "").upper()
                if sample_token in name_u or sample_token in pref_u:
                    return True
            return False

        if not _has_token(cars_candidates) or not _has_token(fluor_candidates):
            raise ValueError(
                f"No ND2 candidates matched sample token '{sample_token}' for folder '{folder_base}'."
            )

    # 5) Score candidates
    def score_cars(name: str, meta: Dict[str, Any]) -> int:
        s = 0
        if sample_token and sample_token in name.upper():
            s += 100
        if mag_kw and mag_kw in name:
            s += 2
        if prefix_hint and meta["prefix"] == prefix_hint:
            s += 1
        return s

    def score_fluor(name: str, meta: Dict[str, Any]) -> int:
        s = 0
        if sample_token and sample_token in name.upper():
            s += 100
        if mag_kw and mag_kw in name:
            s += 2
        if prefix_hint and meta["prefix"] == prefix_hint:
            s += 1
        for idx, mk in enumerate(marker_priority):
            if mk in name:
                s += 10 - idx
                break
        return s

    cars_name, _ = max(cars_candidates, key=lambda nm: score_cars(*nm))
    fluor_name, _ = max(fluor_candidates, key=lambda nm: score_fluor(*nm))

    # 6) Pick the chosen marker
    chosen_marker: Optional[str] = None
    for mk in marker_priority:
        if mk in fluor_name:
            chosen_marker = mk
            break
    if chosen_marker is None:
        chosen_marker = marker_priority[0]
    chosen_marker = resolve_marker_name(chosen_marker, cfg)

    return {"cars_nd2": cars_name, "fluor_nd2": fluor_name, "cell_marker": chosen_marker}


def compute_myelin_average_for_series(
    spectrum_folder: str,
    reference_image: np.ndarray,
    foci_params: dict,
    myelin_params: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Compute a myelin-only average spectrum for a hyperspectral *folder*.

    Uses the 9th wavenumber frame (index 8) to build masks, identical to the droplet
    detection pipeline. Subtracts all droplet pixels from the myelin mask and fits
    peaks on the repaired raw spectrum.
    """
    myelin_params = myelin_params or {}

    # 1) Gather ND2s in this hyperspectral folder and sort by wavenumber index.
    nd2_names = [f for f in os.listdir(spectrum_folder) if f.lower().endswith(".nd2")]
    if not nd2_names:
        return {"Series": os.path.basename(spectrum_folder), "Error": "No ND2 files"}

    def _num_key(p: str) -> int | str:
        m = re.search(r"(\d+)(?=\.nd2$)", os.path.basename(p))
        return int(m.group(1)) if m else p

    nd2_names = sorted(nd2_names, key=_num_key)
    nd2_paths = [os.path.join(spectrum_folder, n) for n in nd2_names]

    # Build wavenumbers exactly like the droplet code path
    def _compute_wavenumber(lambda_nm: float) -> float:
        return 1.0e7 * ((1.0 / lambda_nm) - (1.0 / 1031.0))

    wavelengths_nm = [801.0 - 0.5 * i for i in range(32)]
    wavenumbers = np.array([_compute_wavenumber(wl) for wl in wavelengths_nm], dtype=float)

    # Sanity check: CH stretch should be ~2780–3040 cm^-1
    wmin, wmax = float(np.min(wavenumbers)), float(np.max(wavenumbers))
    if not (2700 <= wmin <= 3045 and 2700 <= wmax <= 3100):
        logger.warning("[MyelinAvg] wavenumbers look off: %.1f–%.1f cm^-1", wmin, wmax)

    # 2) Build corrected hyperspectral stack (N, H, W): East-shadows + reference division.
    stack_corr: List[np.ndarray] = []
    for p in nd2_paths:
        with ND2Reader(p) as nd2:
            nd2.iter_axes = ""  # single plane expected
            raw = np.nan_to_num(nd2.get_frame_2D(c=CARS_CH))
        raw = apply_east_shadows_filter(raw)
        den = np.clip(reference_image, 1e-6, None)
        stack_corr.append(raw / den)
    stack = np.asarray(stack_corr, dtype=float)  # shape: (N, H, W)
    if stack.ndim != 3 or stack.shape[0] < 9:
        return {"Series": os.path.basename(spectrum_folder), "Error": "Unexpected stack shape"}

    # 3) 9th frame => masks
    base9 = stack[8]  # 0-based index for "9th" wavenumber frame

    # 3a) Other features (droplets) mask
    droplet_mask = find_foci(
        base9,
        sigma=float(foci_params.get("sigma", 1.0) or 0.0),
        min_distance=int(foci_params.get("min_distance", 5) or 5),
        min_size=int(foci_params.get("min_size", 20) or 20),
        std_dev_multiplier=float(foci_params.get("std_dev_multiplier", 3.5) or 3.5),
        remove_saturated=bool(foci_params.get("remove_saturated", True)),
        saturation_threshold=float(foci_params.get("saturation_threshold", 3500) or 3500),
        saturated_min_size=int(foci_params.get("saturated_min_size", 50) or 50),
        debug=bool(VERBOSE),
    )

    # 3b) Myelin mask (permissive gate; no watershed)
    my_mask = find_foci(
        base9,
        sigma=float(myelin_params.get("sigma", 1.0) or 1.0),
        min_distance=int(myelin_params.get("min_distance", 8) or 8),
        min_size=int(myelin_params.get("min_size", 300) or 300),
        std_dev_multiplier=float(myelin_params.get("std_dev_multiplier", 0.6) or 0.8),
        remove_saturated=True,
        saturation_threshold=float(foci_params.get("saturation_threshold", 3500) or 3500),
        saturated_min_size=int(foci_params.get("saturated_min_size", 50) or 50),
        separate_objects=False,
        morph_op=str(myelin_params.get("morph_op", "closing") or "closing"),
        morph_radius=int(myelin_params.get("morph_radius", 2) or 2),
        debug=bool(VERBOSE),
    )
    my_frac = float(np.count_nonzero(my_mask)) / float(my_mask.size)

    # 3c) Subtract out all droplet pixels.
    keep_mask = np.logical_and(my_mask, ~droplet_mask)
    n_pix = int(np.count_nonzero(keep_mask))
    if n_pix == 0:
        return {"Series": os.path.basename(spectrum_folder), "Error": "Empty myelin-minus-droplets mask"}

    # 4) Average spectrum over remaining pixels (one mean per wavenumber).
    y = np.array([np.nanmean(frame[keep_mask]) for frame in stack], dtype=float)

    # 5) Repair glitches (fit routine normalizes internally).
    y_repaired = _repair_zero_glitches(y)
    y_norm, scale = _normalize_row_max(y_repaired)  # kept for QA

    # 6) Fit using the same routine as droplets.
    series = os.path.basename(spectrum_folder)
    fit = fit_cars_peaks(wavenumbers, y_repaired, config)  # same call signature

    # 6b) Optional debug plot
    if PEAKFIT_DEBUG:
        _plot_peak_fit_debug(
            wavenumbers,
            y_repaired,
            fit,
            droplet_id=-1,
            category="Avg-Myelin",
            location="N/A",
            marker="",
        )

    row: Dict[str, Any] = dict(fit)
    row.update(
        {
            "Series": series,
            "MyelinMaskFraction": my_frac,
            "PixelsUsed": int(np.count_nonzero(keep_mask)),
            "NormScale": float(scale),
        }
    )
    return row


def process_hyperspectral_series(
    spectrum_folder: str,
    reference_image: np.ndarray,
    output_path: str,
    foci_params: Dict[str, Any],
) -> None:
    """
    Process a hyperspectral series to extract droplet intensities and summary.

    Steps
    -----
    - Read 32 ND2s in the series; correct each via East-shadows + reference image.
    - Build droplet mask from the 9th corrected image.
    - Align to the best position in a separate CARS ND2 via Pearson similarity.
    - Build cell mask from fluorescence ND2.
    - Export Raw/Normalized/Peak Fits sheets and a summary figure + ratio heatmap.
    """
    from skimage.filters import gaussian
    from .analysis import max_project_fluorescence  # local import avoids circular
    from .peakfit import start_debug_capture, chi2_add  # import here once
    from .visualize import debug_display_3way_segmentation

    assert config is not None, "Global 'config' must be set before calling process_hyperspectral_series()."

    folder_base = os.path.basename(spectrum_folder)

    # Peak-fit debug capture (PNG + multi-page PDF, optional PPTX)
    debug_root = os.path.join(config["paths"]["data_directory"], config.get("debug_output_dir", "Debug"))
    series_debug_dir = os.path.join(debug_root, f"peakfits_{folder_base}")
    try:
        if PEAKFIT_DEBUG:
            start_debug_capture(series_debug_dir)
    except Exception as exc:
        if VERBOSE:
            logger.info("[PeakFit DEBUG] start_debug_capture failed: %s", exc)

    cfg_map = config.get("hyperspectral_mapping", {})
    logger.info("Available hyperspectral_mapping keys: %s", list(cfg_map.keys()))
    logger.info("Looking for folder_base: %s", folder_base)

    mapping = cfg_map.get(folder_base)
    if mapping is None:
        mapping = infer_hyperspectral_mapping(spectrum_folder, config)
        logger.info("[HYPERMAP] Inferred mapping: %s", mapping)
    else:
        logger.info("[HYPERMAP] Using config mapping: %s", mapping)

    data_dir = config["paths"]["data_directory"]
    cars_nd2_path = os.path.join(data_dir, mapping["cars_nd2"])
    fluor_nd2_path = os.path.join(data_dir, mapping["fluor_nd2"])
    cell_marker = mapping["cell_marker"]

    # --- Load and correct the 32-image hyperspectral series ---
    def _num_key(p: str) -> int | str:
        m = re.search(r"(\d+)(?=\.nd2$)", os.path.basename(p))
        return int(m.group(1)) if m else p

    nd2_files = sorted(
        [os.path.join(spectrum_folder, f) for f in os.listdir(spectrum_folder) if f.endswith(".nd2")],
        key=_num_key,
    )
    if len(nd2_files) != 32:
        raise ValueError(f"Expected 32 images in the series, but found {len(nd2_files)}.")

    corrected_images: List[np.ndarray] = []
    for nd2_file in nd2_files:
        with ND2Reader(nd2_file) as nd2:
            raw_image = np.nan_to_num(nd2.get_frame_2D(c=CARS_CH))
            correlated_image = apply_east_shadows_filter(raw_image)
            den = np.clip(reference_image, 1e-6, None)
            c_image = correlated_image / den
            corrected_images.append(c_image)

    mask_image = corrected_images[8]  # 9th image (index 8) used for droplet mask

    # --- Align to best position in the CARS ND2 via Pearson r ---
    with ND2Reader(cars_nd2_path) as cars_nd2:
        total_v = cars_nd2.sizes.get("v", 1)
        best_v, best_r = None, -np.inf

        H = mask_image.astype(np.float32)
        Hm, Hs = H.mean(), H.std()

        for v_idx in range(total_v):
            mip_slices = []
            for z in range(cars_nd2.sizes.get("z", 1)):
                raw_sl = np.nan_to_num(cars_nd2.get_frame_2D(v=v_idx, c=CARS_CH, z=z)).astype(np.float32)
                filtered = apply_east_shadows_filter(raw_sl)
                den = np.clip(reference_image, 1e-6, None)
                corrected = filtered / den
                if foci_params.get("sigma", 0) > 0:
                    corrected = gaussian(corrected, sigma=foci_params["sigma"], preserve_range=True)
                mip_slices.append(corrected)
            C = np.max(np.stack(mip_slices, axis=0), axis=0)

            Cm, Cs = C.mean(), C.std()
            r = -np.inf if (Hs == 0 or Cs == 0) else float(((H - Hm) * (C - Cm)).sum() / (Hs * Cs * H.size))
            if r > best_r:
                best_r, best_v = r, v_idx

    logger.info("[HYPERMAP] Folder=%s, best v=%s, r=%.3f", folder_base, str(best_v), best_r)
    if best_v is None:  # safety fallback
        best_v = 0
    cars_img_for_overlay = mask_image

    # --- Build cell/auto/LAMP2 masks from fluorescence ND2 ---
    with ND2Reader(fluor_nd2_path) as fl_nd2:
        ch_idx = config["channel_map"][cell_marker]
        fluoro_mip = max_project_fluorescence(
            fl_nd2,
            ch_index=ch_idx,
            position=best_v,
            fluoro_params=config["morphology_params"]["fluorescence_params"],
        )

        # DEBUG: save alignment triptych (Hyperspec 2850, matched Fluor z, matched CARS z)
        try:
            if config.get("debug_alignment", False) or VERBOSE:
                with ND2Reader(cars_nd2_path) as cars_nd2_dbg:
                    mip_slices_dbg = []
                    for z in range(cars_nd2_dbg.sizes.get("z", 1)):
                        raw_sl = np.nan_to_num(cars_nd2_dbg.get_frame_2D(v=best_v, c=CARS_CH, z=z)).astype(np.float32)
                        filtered = apply_east_shadows_filter(raw_sl)
                        den = np.clip(reference_image, 1e-6, None)
                        corrected = filtered / den
                        if foci_params.get("sigma", 0) > 0:
                            corrected = gaussian(corrected, sigma=foci_params["sigma"], preserve_range=True)
                        mip_slices_dbg.append(corrected)
                    cars_mip_best = np.max(np.stack(mip_slices_dbg, axis=0), axis=0)

                out_dir = os.path.join(config["paths"]["data_directory"], config.get("debug_output_dir", "Debug"))
                out_png = os.path.join(out_dir, f"align_{folder_base}_z{best_v}_r{best_r:.3f}.png")
                save_alignment_triptych(
                    out_png,
                    mask_image,
                    fluoro_mip,
                    cars_mip_best,
                    label=folder_base,
                    chosen_z=best_v,
                    corr_value=best_r,
                    show=config.get("debug_alignment_show_plots", False),
                )
        except Exception as exc:
            if VERBOSE:
                logger.info("[DEBUG] alignment triptych failed: %s", exc)

        auto_ch = config["channel_map"].get("Autofluorescence")
        if auto_ch is not None:
            with ND2Reader(fluor_nd2_path) as fl_nd22:
                auto_mip = max_project_fluorescence(
                    fl_nd22,
                    ch_index=auto_ch,
                    position=best_v,
                    fluoro_params=config["morphology_params"]["autofluorescence_params"],
                )
            auto_mask = find_foci(auto_mip, **config["morphology_params"]["autofluorescence_params"], debug=VERBOSE)
        else:
            auto_mip = None
            auto_mask = np.zeros_like(mask_image, dtype=bool)

        lamp2_mask = None
        lamp2_ch = config["channel_map"].get("LAMP2")
        if lamp2_ch is not None:
            with ND2Reader(fluor_nd2_path) as fl_nd22:
                lamp2_mip = max_project_fluorescence(
                    fl_nd22,
                    ch_index=lamp2_ch,
                    position=best_v,
                    fluoro_params=config["morphology_params"]["fluorescence_params"],
                )
            lamp2_params = config["morphology_params"].get(
                "lamp2_params", config["morphology_params"]["autofluorescence_params"]
            )
            lamp2_mask = find_foci(lamp2_mip, **lamp2_params, debug=VERBOSE)
            lamp2_mask = dilation(lamp2_mask, disk(1))
        lamp2_available = lamp2_mask is not None

    # --- Cell mask thresholding with per-marker overrides ---
    fluorescence_params = config["morphology_params"]["fluorescence_params"]
    marker_thresholds = config.get("marker_thresholds", {}).get(cell_marker, {})
    threshold_method = marker_thresholds.get("threshold_method", fluorescence_params.get("threshold_method", "otsu"))
    offset_val = marker_thresholds.get("offset", fluorescence_params.get("offset", 1.0))

    cell_mask = process_fluorescence_channel(
        fluoro_mip,
        cell_size=fluorescence_params["cell_size"],
        min_size=fluorescence_params["min_size"],
        closing_radius=fluorescence_params["closing_radius"],
        gaussian_sigma=fluorescence_params["gaussian_sigma"],
        fill_holes=fluorescence_params["fill_holes"],
        threshold_method=threshold_method,
        offset=offset_val,
        exclude_dark_regions=fluorescence_params.get("exclude_dark_regions", True),
        dark_threshold=fluorescence_params.get("dark_threshold", 50),
        min_hole_size=fluorescence_params.get("min_hole_size", 20000),
        debug=False,
    )

    # --- Droplet masks and overlays ---
    cars_foci_mask = find_foci(mask_image, **foci_params)
    
    # Use the same rule as analysis.py: overlap threshold = min_size used to detect CARS foci
    min_overlap = int(foci_params.get("min_size", 20))
    
    labeled_pure_lipid, labeled_lipo_lipid, labeled_pure_lipo = _colocalize_objects(
        cars_foci_mask,
        auto_mask if auto_mip is not None else np.zeros_like(cars_foci_mask, dtype=bool),
        min_overlap=min_overlap,
    )
    
    # Boolean views for visualization and downstream logic
    pure_lipid_mask        = (labeled_pure_lipid > 0)
    lipid_lipofuscin_mask  = (labeled_lipo_lipid > 0)
    pure_lipofuscin_mask   = (labeled_pure_lipo > 0)
    
    # “Lipid” = any CARS-positive droplet (pure lipid or lipid+lipofuscin)
    lipid_mask = pure_lipid_mask | lipid_lipofuscin_mask
    
    intracellular_pure_lipid        = pure_lipid_mask & cell_mask
    intracellular_lipid_lipofuscin  = lipid_lipofuscin_mask & cell_mask
    intracellular_pure_lipofuscin   = pure_lipofuscin_mask & cell_mask

    if VERBOSE:
        debug_display_3way_segmentation(
            intracellular_pure_lipid,
            intracellular_lipid_lipofuscin,
            intracellular_pure_lipofuscin,
            cell_mask,
            auto_image=auto_mip,
            cars_image=cars_img_for_overlay,
            pos_index=best_v,
            title_suffix=f"[{cell_marker}]",
        )
        visualize_hyperspectral_mask_overlay(mask_image, lipid_mask)

    # --- Per-droplet intensity table ---
    lipid_labels = measure.label(lipid_mask)
    _save_labeled_mask_images(lipid_labels, mask_image, spectrum_folder)

    lipid_data: List[List[Any]] = []
    cell_marker_report = cell_marker  # report only if intracellular
    for region in measure.regionprops(lipid_labels):
        lipid_id = region.label
        r0, c0 = (int(region.centroid[0]), int(region.centroid[1]))

        if pure_lipid_mask[r0, c0]:
            category = "Lipid"
        elif lipid_lipofuscin_mask[r0, c0]:
            category = "Lipidated Lipofuscin"
        else:
            category = "Lipofuscin"

        intensities = [np.mean(img[region.coords[:, 0], region.coords[:, 1]]) for img in corrected_images]
        is_intra = np.any(cell_mask[region.coords[:, 0], region.coords[:, 1]])
        location = "Intracellular" if is_intra else "Extracellular"
        marker_for_row = cell_marker_report if is_intra else ""
        lamp2_coloc = bool(lamp2_available and np.any(lamp2_mask[region.coords[:, 0], region.coords[:, 1]]))

        lipid_data.append([lipid_id, category, location, marker_for_row, lamp2_coloc] + intensities)

    wnum_cols = [f"Wavenumber {i + 1}" for i in range(32)]
    columns_raw = ["Lipid ID", "Category", "Location", "Cell Marker", "LAMP2_Coloc"] + wnum_cols
    lipid_df_raw = pd.DataFrame(lipid_data, columns=columns_raw)

    # --- Normalized sheet and header rows for raw sheet ---
    def compute_wavenumber(lambda_nm: float) -> float:
        return 1.0e7 * ((1.0 / lambda_nm) - (1.0 / 1031.0))

    wavelengths_nm = [801.0 - 0.5 * i for i in range(32)]
    wavenumbers = [compute_wavenumber(wl) for wl in wavelengths_nm]

    header_row_wavelengths = {k: "" for k in lipid_df_raw.columns}
    header_row_wavenumbers = {k: "" for k in lipid_df_raw.columns}
    for i, col in enumerate([f"Wavenumber {i + 1}" for i in range(32)]):
        header_row_wavelengths[col] = wavelengths_nm[i]
        header_row_wavenumbers[col] = wavenumbers[i]

    raw_with_headers = pd.concat(
        [pd.DataFrame([header_row_wavelengths, header_row_wavenumbers]), lipid_df_raw], ignore_index=True
    )

    lipid_df_norm = lipid_df_raw.copy()
    spectral_cols = [f"Wavenumber {i + 1}" for i in range(32)]
    data_to_normalize = lipid_df_norm[spectral_cols]
    row_maxes = data_to_normalize.max(axis=1).replace({0: 1})
    lipid_df_norm[spectral_cols] = data_to_normalize.div(row_maxes, axis=0)

    rename_map = {f"Wavenumber {i + 1}": f"{wavenumbers[i]:.2f}" for i in range(32)}
    lipid_df_norm = lipid_df_norm.rename(columns=rename_map)

    # --- Peak fitting (optional) ---
    peak_df: Optional[pd.DataFrame] = None
    try:
        x_cm1 = np.array(wavenumbers, dtype=float)
        spectral_cols_raw = [f"Wavenumber {i + 1}" for i in range(32)]
        peak_rows: List[Dict[str, Any]] = []
        for _, r in lipid_df_raw.iterrows():
            # 1) Get raw spectrum
            y_raw = r[spectral_cols_raw].to_numpy(dtype=float)

            # 2) Repair random zero / near-zero spikes
            y_repaired = _repair_zero_glitches(y_raw)

            # 3) Fit on RAW (peakfit.py handles normalization internally)
            fit = fit_cars_peaks(x_cm1, y_repaired, config)

            if PEAKFIT_DEBUG:
                _plot_peak_fit_debug(
                    x_cm1,
                    y_repaired,
                    fit,
                    droplet_id=r["Lipid ID"],
                    category=r["Category"],
                    location=r["Location"],
                    marker=r["Cell Marker"],
                )

            # 4) accumulate χ² for the batch summary
            chi2_add(
                series_label=folder_base,
                droplet_id=int(r["Lipid ID"]),
                redchi=fit.get("redchi", float("nan")),
                success=bool(fit.get("success", False)),
                strategy=str(fit.get("strategy_used", "")),
            )

            # 5) Collect peak rows (include widths and any present peaks)
            peak_indices = sorted(
                int(k[1:]) for k in fit.keys() if isinstance(k, str) and k.startswith("x") and k[1:].isdigit()
            )
            for k in peak_indices:
                peak_rows.append(
                    {
                        "Lipid ID": r["Lipid ID"],
                        "Category": r["Category"],
                        "Location": r["Location"],
                        "Cell Marker": r["Cell Marker"],
                        "LAMP2_Coloc": r["LAMP2_Coloc"],
                        "Peak": k,
                        "Center_cm^-1": fit.get(f"x{k}", np.nan),
                        "Amplitude": fit.get(f"A{k}", np.nan),
                        "Width_cm^-1": fit.get(f"w{k}", np.nan),
                        "FitSuccess": fit.get("success", False),
                    }
                )
        peak_df = pd.DataFrame(peak_rows)
    except Exception as exc:
        logger.info("[PeakFit] Skipping peak fitting: %s", exc)

    # --- Write outputs ---
    with pd.ExcelWriter(output_path) as writer:
        raw_with_headers.to_excel(writer, sheet_name="Raw Data", index=False)
        lipid_df_norm.to_excel(writer, sheet_name="Normalized Data", index=False)
        if peak_df is not None and not peak_df.empty:
            peak_df.to_excel(writer, sheet_name="Peak Fits", index=False)
    logger.info("Hyperspectral lipid intensities saved to %s", output_path)

    # --- Summary figure ---
    summary_png = os.path.join(spectrum_folder, "Hyperspectral_PeakFit_Summary.png")
    try:
        save_batch_peak_summary(peak_df, lipid_df_norm, wavenumbers, summary_png)
    except Exception as exc:
        logger.info("[SummaryPlot] Skipping batch summary: %s", exc)

    # --- Ratio heatmap (2930 / 2850) ---
    ratio_map = np.full_like(lipid_labels, fill_value=-1, dtype=np.float32)
    ratio_values: List[float] = []

    col_2850 = "Wavenumber 9"
    col_2930 = "Wavenumber 19"
    col_to_idx = {c: i for i, c in enumerate(lipid_df_raw.columns)}
    i_2850 = col_to_idx[col_2850]
    i_2930 = col_to_idx[col_2930]

    for row in lipid_df_raw.itertuples(index=False):
        lipid_id = row[col_to_idx["Lipid ID"]]
        intens_2850 = row[i_2850]
        intens_2930 = row[i_2930]
        ratio_val = (intens_2930 / intens_2850) if (intens_2850 > 0) else 0.0
        ratio_map[lipid_labels == lipid_id] = ratio_val
        ratio_values.append(ratio_val)

    if len(ratio_values) == 0:
        logger.info("No droplets found, skipping ratio heatmap.")
        return

    ratio_min = float(np.min(ratio_values))
    ratio_max = float(np.max(ratio_values)) if np.max(ratio_values) > 0 else 1.0
    ratio_norm = (ratio_map - ratio_min) / (ratio_max - ratio_min + 1e-9)
    ratio_norm_clipped = np.clip(ratio_norm, 0.0, 1.0)

    cmap = LinearSegmentedColormap.from_list("yellow_red", [(1.0, 1.0, 0.0), (1.0, 0.0, 0.0)])
    ratio_rgba = cmap(ratio_norm_clipped)
    ratio_rgb = (ratio_rgba[..., :3] * 255).astype(np.uint8)

    bg_mask = ratio_map < 0
    ratio_rgb[bg_mask] = [0, 0, 0]

    plt.figure(figsize=(6, 6))
    plt.imshow(ratio_rgb)
    plt.title("Droplet Ratio Map (2930 / 2850)")
    plt.axis("off")
    plt.show()
    plt.close()

    ratio_bgr = cv2.cvtColor(ratio_rgb, cv2.COLOR_RGB2BGR)
    out_path_ratio = os.path.join(spectrum_folder, "Ratio_2930_over_2850.png")
    cv2.imwrite(out_path_ratio, ratio_bgr)
    logger.info("Ratio heatmap saved to %s", out_path_ratio)