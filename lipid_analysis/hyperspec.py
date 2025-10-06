# lipid_analysis/hyperspec.py

import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from nd2reader import ND2Reader
from skimage import measure
from skimage.morphology import dilation, disk

from .config_utils import resolve_marker_name
from .constants import CARS_CH, PEAKFIT_DEBUG, VERBOSE
from .debug_utils import save_alignment_triptych
from .filters import apply_east_shadows_filter
from .segmentation import find_foci, process_fluorescence_channel

# This is intentionally set from the CLI before calling module functions.
# e.g. in cli.py:  import lipid_analysis.hyperspec as hyperspec; hyperspec.config = config
config = None

# --- Glitch repair helpers ----------------------------------------------------


def _repair_zero_glitches(
    y: np.ndarray, z_abs: float = 1e-9, z_rel: float = 0.02, win: int = 3
) -> np.ndarray:
    """
    Replace random zero/near-zero points by linear interpolation.
    A point is considered a glitch if it is <= z_abs OR <= z_rel * local_median.
    - win: neighborhood half-window for local median (±win).
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
        if neigh.size == 0:
            med[i] = y[i]
        else:
            med[i] = np.median(neigh)

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
    """Normalize by max (ignoring NaNs); returns (y_norm, scale)."""
    y = np.asarray(y, dtype=float)
    y_clean = np.nan_to_num(y, nan=0.0)
    m = float(np.max(y_clean))
    m = m if m > eps else 1.0
    return (y_clean / m), m


def _save_labeled_mask_images(labeled_mask, base_gray, out_dir):
    """Save two PNGs: (1) color-by-label with numeric IDs, (2) overlay on grayscale."""
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
        (tw, th), baseline = cv2.getTextSize(txt, font, 0.5, 1)
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


def save_batch_peak_summary(peak_df, lipid_df_norm, wavenumbers, out_png_path):
    """Create a 2x2 summary figure for one hyperspectral batch."""
    spec_cols = [
        c
        for c in lipid_df_norm.columns
        if c.replace(".", "", 1).replace("-", "", 1).isdigit()
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
        fit_success_pct = (
            (100.0 * n_droplets_fit / n_droplets_total) if n_droplets_total > 0 else 0.0
        )
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
    axA.fill_between(
        wavenumbers, mean_spec - std_spec, mean_spec + std_spec, alpha=0.25
    )
    axA.set_title("Mean ±1 SD (Normalized Spectra)")
    axA.set_xlabel("Raman shift (cm$^{-1}$)")
    axA.set_ylabel("Normalized intensity")
    axA.grid(alpha=0.3)

    # (B) Fitted centers
    axB = fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(0)
    for k in range(1, 8):
        xs = np.full(len(centers_by_peak[k]), k, dtype=float) + rng.normal(
            0, 0.05, size=len(centers_by_peak[k])
        )
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
    data_amp = [
        amps_by_peak[k] if len(amps_by_peak[k]) > 0 else [np.nan] for k in range(1, 8)
    ]
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
    axD.set_title(
        f"Fit Success: {fit_success_pct:.1f}%  (n={n_droplets_fit}/{n_droplets_total})"
    )
    for spine in ["top", "right"]:
        axD.spines[spine].set_visible(False)

    fig.suptitle("Hyperspectral Peak-Fit Summary", y=0.98, fontsize=14)
    plt.savefig(out_png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved batch summary figure to: {out_png_path}")


def visualize_hyperspectral_mask_overlay(cars_image, lipid_mask):
    """Show grayscale CARS, lipid mask (yellow), and an overlay."""

    def create_rgb_mask(mask, color):
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for i in range(3):
            rgb[..., i] = mask * color[i]
        return rgb

    max_val = cars_image.max() if cars_image.max() > 0 else 1
    grayscale_8bit = (cars_image / max_val * 255).astype(np.uint8)
    lipid_mask_rgb = create_rgb_mask(lipid_mask, [255, 255, 0])
    grayscale_rgb = np.stack([grayscale_8bit] * 3, axis=-1)
    overlay_rgb = np.clip(0.5 * grayscale_rgb + 0.5 * lipid_mask_rgb, 0, 255).astype(
        np.uint8
    )

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


def infer_hyperspectral_mapping(spectrum_folder, config):
    """
    Infer {cars_nd2, fluor_nd2, cell_marker} for a hyperspectral folder using
    folder-name tokens and actual files present in config['paths']['data_directory'].
    """
    from .filepairing import parse_nd2_filename

    folder_base = os.path.basename(spectrum_folder)
    # NEW: extract sample token from folder name (e.g., AD44, AD33, CTRL/HC)
    sample_token = None
    try:
        m = re.search(
            r"\b(AD\d{2}|CTRL|CONTROL|HC)\b", folder_base, flags=re.IGNORECASE
        )
        if m:
            sample_token = m.group(1).upper()
    except Exception:
        pass

    name_l = folder_base.lower()

    folder_map = config.get("hyperspectral_folder_map", {})
    if not folder_map:
        raise ValueError("config['hyperspectral_folder_map'] is missing or empty.")

    mag_kw = config["file_keywords"]["magnification_keyword"]

    # 1) Match token
    matched = None
    for token, spec in folder_map.items():
        if token in name_l:
            matched = (token, spec["label"], list(spec["markers"]))
            break
    if matched is None:
        raise ValueError(
            f"Cannot infer cell type from folder '{folder_base}'. "
            f"Expected one of: {', '.join(folder_map.keys())}"
        )
    token, target_stacks_label, marker_priority = matched

    # 2) Prefix hint from ND2 inside the spectrum folder
    nd2_inside = [f for f in os.listdir(spectrum_folder) if f.lower().endswith(".nd2")]
    inferred_prefixes = []
    for f in nd2_inside:
        try:
            meta = parse_nd2_filename(f, config)
            if meta["prefix"]:
                inferred_prefixes.append(meta["prefix"])
        except Exception:
            pass
    prefix_hint = None
    if inferred_prefixes:
        from collections import Counter

        prefix_hint = Counter(inferred_prefixes).most_common(1)[0][0]

    def stacks_label_matches(meta_label, target_label):
        """Treat empty/None label as wildcard."""
        return (
            (meta_label is None) or (meta_label == "") or (meta_label == target_label)
        )

    # NEW: if prefix_hint conflicts with sample_token, ignore the prefix_hint
    try:
        if (
            prefix_hint
            and sample_token
            and sample_token not in str(prefix_hint).upper()
        ):
            if VERBOSE:
                print(
                    f"[HYPERMAP] Ignoring prefix_hint '{prefix_hint}' (mismatch with sample token '{sample_token}')."
                )
            prefix_hint = None
    except Exception:
        pass

    # 3) Scan main data directory for ND2s with the right stacks label
    data_dir = config["paths"]["data_directory"]
    cars_candidates, fluor_candidates = [], []
    for item in os.listdir(data_dir):
        if not item.lower().endswith(".nd2"):
            continue
        if "largearea" in item.lower():
            continue
        meta = parse_nd2_filename(item, config)
        if not stacks_label_matches(meta["stacks_label"], target_stacks_label):
            continue
        if meta["contains_cars"]:
            cars_candidates.append((item, meta))
        else:
            fluor_candidates.append((item, meta))

    # 4) Relaxed pass (same condition, present for parity with original)
    if not cars_candidates or not fluor_candidates:
        cars_candidates, fluor_candidates = [], []
        for item in os.listdir(data_dir):
            if not item.lower().endswith(".nd2"):
                continue
            if "largearea" in item.lower():
                continue
            meta = parse_nd2_filename(item, config)
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

    # NEW: Prefer candidates whose names/prefixes contain the sample token (e.g., AD44)
    def _filter_by_sample_token(cands):
        if not sample_token:
            return cands
        filtered = []
        for n, m in cands:
            name_u = n.upper()
            pref_u = str(m.get("prefix") or "").upper()
            if sample_token in name_u or sample_token in pref_u:
                filtered.append((n, m))
        return filtered or cands  # graceful fallback if none match

    cars_candidates = _filter_by_sample_token(cars_candidates)
    fluor_candidates = _filter_by_sample_token(fluor_candidates)

    # Optional strict mode: require token match if enabled
    if config.get("strict_sample_match", False) and sample_token:

        def _has_token(cands):
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
    def score_cars(name, meta):
        s = 0
        if sample_token and sample_token in name.upper():
            s += 100  # NEW
        if mag_kw and mag_kw in name:
            s += 2
        if prefix_hint and meta["prefix"] == prefix_hint:
            s += 1
        return s

    def score_fluor(name, meta):
        s = 0
        if sample_token and sample_token in name.upper():
            s += 100  # NEW
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
    chosen_marker = None
    for mk in marker_priority:
        if mk in fluor_name:
            chosen_marker = mk
            break
    if chosen_marker is None:
        chosen_marker = marker_priority[0]
    chosen_marker = resolve_marker_name(chosen_marker, config)

    return {
        "cars_nd2": cars_name,
        "fluor_nd2": fluor_name,
        "cell_marker": chosen_marker,
    }


def process_hyperspectral_series(
    spectrum_folder, reference_image, output_path, foci_params
):
    """
    Process a hyperspectral series to extract lipid droplet intensities and summary:
      - Reads 32 ND2s in the series; corrects each via East-shadows + reference image.
      - Builds droplet mask from the 9th corrected image.
      - Aligns to the best position in a separate CARS ND2 via Pearson similarity.
      - Builds cell mask from fluorescence ND2.
      - Exports Raw/Normalized/Peak Fits sheets and a summary figure + ratio heatmap.
    """
    from skimage.filters import gaussian

    from .analysis import max_project_fluorescence  # local import avoids circular
    from .peakfit import (
        _plot_peak_fit_debug,
        fit_cars_peaks,
        start_debug_capture,
        finish_debug_capture,
        chi2_add,
    )

    from .visualize import debug_display_3way_segmentation

    assert (
        config is not None
    ), "Global 'config' must be set before calling process_hyperspectral_series()."

    folder_base = os.path.basename(spectrum_folder)
    
    # --- Peak-fit debug capture (PNG + multi-page PDF, and PPTX if python-pptx is installed)
    debug_root = os.path.join(
        config["paths"]["data_directory"],
        config.get("debug_output_dir", "Debug"),
    )
    series_debug_dir = os.path.join(debug_root, f"peakfits_{folder_base}")
    try:
        # Only start capture if you actually plan to show plots (PEAKFIT_DEBUG is your existing flag)
        if PEAKFIT_DEBUG:
            start_debug_capture(series_debug_dir)  # creates <dir>/fits.pdf and saves each plot as PNG
    except Exception as _e:
        if VERBOSE:
            print("[PeakFit DEBUG] start_debug_capture failed:", _e)

    
    # NEW: extract sample token from folder name (e.g., AD44, AD33, CTRL/HC)
    sample_token = None
    try:
        m = re.search(
            r"\b(AD\d{2}|CTRL|CONTROL|HC)\b", folder_base, flags=re.IGNORECASE
        )
        if m:
            sample_token = m.group(1).upper()
    except Exception:
        sample_token

    cfg_map = config.get("hyperspectral_mapping", {})
    print("Available hyperspectral_mapping keys:", list(cfg_map.keys()))
    print("Looking for folder_base:", folder_base)

    mapping = cfg_map.get(folder_base)
    if mapping is None:
        mapping = infer_hyperspectral_mapping(spectrum_folder, config)
        print("[HYPERMAP] Inferred mapping:", mapping)
    else:
        print("[HYPERMAP] Using config mapping:", mapping)

    data_dir = config["paths"]["data_directory"]
    cars_nd2_path = os.path.join(data_dir, mapping["cars_nd2"])
    fluor_nd2_path = os.path.join(data_dir, mapping["fluor_nd2"])
    cell_marker = mapping["cell_marker"]

    # --- Load and correct the 32-image hyperspectral series ---
    def _num_key(p):
        m = re.search(r"(\d+)(?=\.nd2$)", os.path.basename(p))
        return int(m.group(1)) if m else p

    nd2_files = sorted(
        [
            os.path.join(spectrum_folder, f)
            for f in os.listdir(spectrum_folder)
            if f.endswith(".nd2")
        ],
        key=_num_key,
    )
    if len(nd2_files) != 32:
        raise ValueError(
            f"Expected 32 images in the series, but found {len(nd2_files)}."
        )

    corrected_images = []
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
                raw_sl = np.nan_to_num(
                    cars_nd2.get_frame_2D(v=v_idx, c=CARS_CH, z=z)
                ).astype(np.float32)
                filtered = apply_east_shadows_filter(raw_sl)
                den = np.clip(reference_image, 1e-6, None)
                corrected = filtered / den
                if foci_params.get("sigma", 0) > 0:
                    corrected = gaussian(
                        corrected, sigma=foci_params["sigma"], preserve_range=True
                    )
                mip_slices.append(corrected)
            C = np.max(np.stack(mip_slices, axis=0), axis=0)

            Cm, Cs = C.mean(), C.std()
            if Hs == 0 or Cs == 0:
                r = -np.inf
            else:
                r = float(((H - Hm) * (C - Cm)).sum() / (Hs * Cs * H.size))

            if r > best_r:
                best_r, best_v = r, v_idx

    print(f"[HYPERMAP] Folder={folder_base}, best v={best_v}, r={best_r:.3f}")
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
        # --- DEBUG: save alignment triptych (Hyperspec 2850, matched Fluor z, matched CARS z) ---
        try:
            if config.get("debug_alignment", False) or VERBOSE:
                # Build CARS max-projection for the chosen v (best_v)
                with ND2Reader(cars_nd2_path) as cars_nd2_dbg:
                    mip_slices_dbg = []
                    for z in range(cars_nd2_dbg.sizes.get("z", 1)):
                        raw_sl = np.nan_to_num(
                            cars_nd2_dbg.get_frame_2D(v=best_v, c=CARS_CH, z=z)
                        ).astype(np.float32)
                        filtered = apply_east_shadows_filter(raw_sl)
                        den = np.clip(reference_image, 1e-6, None)
                        corrected = filtered / den
                        if foci_params.get("sigma", 0) > 0:
                            corrected = gaussian(
                                corrected,
                                sigma=foci_params["sigma"],
                                preserve_range=True,
                            )
                        mip_slices_dbg.append(corrected)
                    cars_mip_best = np.max(np.stack(mip_slices_dbg, axis=0), axis=0)

                # Hyperspec 9th image (index 8) already prepared as mask_image
                out_dir = os.path.join(
                    config["paths"]["data_directory"],
                    config.get("debug_output_dir", "Debug"),
                )
                out_png = os.path.join(
                    out_dir, f"align_{folder_base}_z{best_v}_r{best_r:.3f}.png"
                )
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
        except Exception as _e:
            if VERBOSE:
                print("[DEBUG] alignment triptych failed:", _e)

        auto_ch = config["channel_map"].get("Autofluorescence")
        if auto_ch is not None:
            with ND2Reader(fluor_nd2_path) as fl_nd22:
                auto_mip = max_project_fluorescence(
                    fl_nd22,
                    ch_index=auto_ch,
                    position=best_v,
                    fluoro_params=config["morphology_params"][
                        "autofluorescence_params"
                    ],
                )
            auto_mask = find_foci(
                auto_mip,
                **config["morphology_params"]["autofluorescence_params"],
                debug=VERBOSE,
            )
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
    threshold_method = marker_thresholds.get(
        "threshold_method", fluorescence_params.get("threshold_method", "otsu")
    )
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
    lipid_mask = find_foci(mask_image, **foci_params)
    pure_lipid_mask = lipid_mask & ~auto_mask
    lipid_lipofuscin_mask = lipid_mask & auto_mask
    pure_lipofuscin_mask = auto_mask & ~lipid_mask

    intracellular_pure_lipid = pure_lipid_mask & cell_mask
    intracellular_lipid_lipofuscin = lipid_lipofuscin_mask & cell_mask
    intracellular_pure_lipofuscin = pure_lipofuscin_mask & cell_mask

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

    lipid_data = []
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

        intensities = [
            np.mean(img[region.coords[:, 0], region.coords[:, 1]])
            for img in corrected_images
        ]
        is_intra = np.any(cell_mask[region.coords[:, 0], region.coords[:, 1]])
        location = "Intracellular" if is_intra else "Extracellular"
        marker_for_row = cell_marker_report if is_intra else ""
        lamp2_coloc = bool(
            lamp2_available
            and np.any(lamp2_mask[region.coords[:, 0], region.coords[:, 1]])
        )

        lipid_data.append(
            [lipid_id, category, location, marker_for_row, lamp2_coloc] + intensities
        )

    wnum_cols = [f"Wavenumber {i + 1}" for i in range(32)]
    columns_raw = [
        "Lipid ID",
        "Category",
        "Location",
        "Cell Marker",
        "LAMP2_Coloc",
    ] + wnum_cols
    lipid_df_raw = pd.DataFrame(lipid_data, columns=columns_raw)

    # --- Normalized sheet and header rows for raw sheet ---
    def compute_wavenumber(lambda_nm):
        return 1.0e7 * ((1.0 / lambda_nm) - (1.0 / 1031.0))

    wavelengths_nm = [801.0 - 0.5 * i for i in range(32)]
    wavenumbers = [compute_wavenumber(wl) for wl in wavelengths_nm]

    header_row_wavelengths = {k: "" for k in lipid_df_raw.columns}
    header_row_wavenumbers = {k: "" for k in lipid_df_raw.columns}
    for i, col in enumerate([f"Wavenumber {i + 1}" for i in range(32)]):
        header_row_wavelengths[col] = wavelengths_nm[i]
        header_row_wavenumbers[col] = wavenumbers[i]

    raw_with_headers = pd.concat(
        [pd.DataFrame([header_row_wavelengths, header_row_wavenumbers]), lipid_df_raw],
        ignore_index=True,
    )

    lipid_df_norm = lipid_df_raw.copy()
    spectral_cols = [f"Wavenumber {i + 1}" for i in range(32)]
    data_to_normalize = lipid_df_norm[spectral_cols]
    row_maxes = data_to_normalize.max(axis=1).replace({0: 1})
    lipid_df_norm[spectral_cols] = data_to_normalize.div(row_maxes, axis=0)

    rename_map = {f"Wavenumber {i + 1}": f"{wavenumbers[i]:.2f}" for i in range(32)}
    lipid_df_norm = lipid_df_norm.rename(columns=rename_map)

    # --- Peak fitting (optional) ---
    peak_df = None
    try:
        x_cm1 = np.array(wavenumbers, dtype=float)
        spectral_cols_raw = [f"Wavenumber {i + 1}" for i in range(32)]
        peak_rows = []
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
                    y_repaired, # raw spectrum for plotting
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

            # 5) Collect peak rows
            for k in range(1, 8):
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
                        "FitSuccess": fit.get("success", False),
                    }
                )
        peak_df = pd.DataFrame(peak_rows)
    except Exception as e:
        print(f"[PeakFit] Skipping peak fitting: {e}")

    # --- Write outputs ---
    with pd.ExcelWriter(output_path) as writer:
        raw_with_headers.to_excel(writer, sheet_name="Raw Data", index=False)
        lipid_df_norm.to_excel(writer, sheet_name="Normalized Data", index=False)
        if peak_df is not None and not peak_df.empty:
            peak_df.to_excel(writer, sheet_name="Peak Fits", index=False)
    print(f"Hyperspectral lipid intensities saved to {output_path}")

    # --- Summary figure ---
    summary_png = os.path.join(spectrum_folder, "Hyperspectral_PeakFit_Summary.png")
    try:
        save_batch_peak_summary(peak_df, lipid_df_norm, wavenumbers, summary_png)
    except Exception as e:
        print(f"[SummaryPlot] Skipping batch summary: {e}")

    # --- Ratio heatmap (2930 / 2850) ---
    ratio_map = np.full_like(lipid_labels, fill_value=-1, dtype=np.float32)
    ratio_values = []

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
        print("No droplets found, skipping ratio heatmap.")
        return

    ratio_min = float(np.min(ratio_values))
    ratio_max = float(np.max(ratio_values)) if np.max(ratio_values) > 0 else 1.0
    ratio_norm = (ratio_map - ratio_min) / (ratio_max - ratio_min + 1e-9)
    ratio_norm_clipped = np.clip(ratio_norm, 0.0, 1.0)

    cmap = LinearSegmentedColormap.from_list(
        "yellow_red", [(1.0, 1.0, 0.0), (1.0, 0.0, 0.0)]
    )
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
    
    # --- Close the PDF and build PPTX for this series
    try:
        if PEAKFIT_DEBUG:
            finish_debug_capture(make_pptx=True)  # also writes <dir>/fits.pptx if python-pptx is present
    except Exception as _e:
        if VERBOSE:
            print("[PeakFit DEBUG] finish_debug_capture failed:", _e)

    print(f"Ratio heatmap saved to {out_path_ratio}")
