from __future__ import annotations

"""
largearea_layers.py — 40× Large-Area Fluorescence Layer Counts

Purpose
-------
Analyze fluorescence-only 40× large-area ND2 scans (tile/mosaic covering full
cortical depth) to:
  • Build per-marker cell masks via the same segmentation logic used at 100×.
  • Infer the superficial→deep axis by detecting the image border with an
    "empty gap" (background) before tissue begins.
  • Convert pixels→µm from ND2 metadata; bin object centroids into cortical
    layers given a user-provided list of layer widths (µm). Voxels deeper than
    the last layer are labeled White Matter (WM).
  • Output per-image CSV with counts per layer×marker (+WM), and a per-object
    table (centroid, depth, layer, area, marker). Also save debug overlays
    including the inferred superficial edge and layer boundaries.

Run tips
--------
- Place this file alongside the existing package modules (it uses relative imports).
- In Spyder, edit the `if __name__ == "__main__"` section with your config path.
- Requires: nd2reader, numpy, pandas, scikit-image, matplotlib, openpyxl (for xlsx).
"""

import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nd2reader import ND2Reader
from skimage.measure import label, regionprops
from skimage.exposure import rescale_intensity
from skimage.filters import gaussian
from skimage.io import imsave
from skimage import morphology

# --- Reuse existing pipeline modules (relative imports assume same package) ---
from lipid_analysis.config_utils import load_config
from lipid_analysis.segmentation import process_fluorescence_channel
from lipid_analysis.reference import _get_pixel_size_microns
from lipid_analysis.io_utils import ensure_subdirectory
from lipid_analysis.imaging import composite_fluorescence

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ----------------------------
# User-editable constants
# ----------------------------
# Intensity cutoff (composite MIP) for tissue presence used everywhere
TISSUE_INTENSITY_MIN: float = 200.0

# Cortical layer widths in microns: [L1, L2, L3, L4, L5, L6]
LAYER_WIDTHS_UM: List[float] = [200, 350, 750, 350, 850, 750]

# Layer-specific TUJ segmentation overrides (keys: 1..6 for L1..L6)
# These override DEFAULT_SEGMENT_KW ⟶ PER_MARKER_SEG_KW("TUJ") ⟶ per-layer below.
# Tune just L1–L4 where superficial overmerge occurs.
LAYER_SEGMENT_OVERRIDES_TUJ: Dict[int, Dict[str, object]] = {
    1: dict(offset=0.55, closing_radius=1, gaussian_sigma=0.40, fill_holes=False, min_size=180),
    2: dict(offset=0.55, closing_radius=1, gaussian_sigma=0.40, fill_holes=False, min_size=180),
    3: dict(offset=0.45, closing_radius=1, gaussian_sigma=0.45, fill_holes=False, min_size=180),
    4: dict(offset=0.35, closing_radius=2, gaussian_sigma=0.45, fill_holes=False, min_size=170),
    5: dict(offset=0.25, closing_radius=2, gaussian_sigma=0.45, fill_holes=False, min_size=170),
}

# Filename token that identifies large-area scans
LARGEAREA_TOKEN: str = "LargeArea"

# Default segmentation knobs (scaled for 40×; adjust as needed)
DEFAULT_SEGMENT_KW: Dict[str, float | int | bool | str] = dict(
    cell_size=1600,
    min_size=150,
    closing_radius=5,
    gaussian_sigma=1.0,
    fill_holes=True,
    threshold_method="local",
    offset=0.5,
    exclude_dark_regions=True,
    dark_threshold=40,
    min_hole_size=20000,
    debug=False,
)

# Channels to EXCLUDE from “cell markers”
# (comparison is case-insensitive and ignores common punctuation/spacing)
EXCLUDED_NONCELL_NAMES = {
    "dapi",
    "lamp2",
    "autofluorescence",
    "auto-fluorescence",
    "auto fluorescence",
    "af",             # if your configs use a channel literally named "AF"
    "auto",           # catches "AutoFluor" / "Autofluor"
}

def _is_noncell_marker(name: str) -> bool:
    n = name.strip().lower().replace("_"," ").replace("-"," ")
    return any(tok in n.split() for tok in EXCLUDED_NONCELL_NAMES) or \
           any(bad in n for bad in ("autofluor", "autofluo", "autofl"))


# Optional per-marker segmentation overrides (fine-tuning per cell type)
# Any keys here will override DEFAULT_SEGMENT_KW for that specific marker.
PER_MARKER_SEG_KW: Dict[str, Dict[str, object]] = {
    # Examples (tune as you like):
    "IBA1": {"threshold_method": "local", "offset": 0.5, "min_size": 150, "cell_size": 1600},
    "GFAP": {"threshold_method": "local", "offset": 0.5, "min_size": 150, "cell_size": 1600},
    "TUJ": {
        "threshold_method": "local",
        "offset": 0.14,        # ↓ less strict → recovers dim somata
        "cell_size": 800,     # ↓ smaller window → more local contrast
        "gaussian_sigma": 0.45, # less blurring, keeps soma rims
        "closing_radius": 3,
        "exclude_dark_regions": False,
    }
}

# ----------------------------
# Small data helpers
# ----------------------------
@dataclass
class ImageAxis:
    side: str                 # {"top","bottom","left","right"}
    depth_px: np.ndarray      # H×W, raw distance from the superficial edge (px)
    start_px: float           # average background gap before cortex begins (px)
    
    
def _debug_show_mips(sum_img: np.ndarray, imgs: Dict[str, np.ndarray],
                     title_stub: str = "", save_dir: Optional[str] = None, show: bool = True) -> Optional[str]:
    """Show the sum/max MIP, per-channel MIPs, and histogram (Spyder-friendly).
    If save_dir is provided, also save a PNG and return the path.
    """
    import matplotlib.pyplot as plt
    from skimage.exposure import rescale_intensity

    sum_disp = rescale_intensity(sum_img, in_range="image", out_range=(0.0, 1.0))
    keys = sorted(imgs.keys())
    n = len(keys)
    cols = min(3, max(1, n))
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(rows + 1, cols, height_ratios=[1]*rows + [0.6])

    # Per-channel MIPs
    for i, k in enumerate(keys):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        ch_disp = rescale_intensity(imgs[k], in_range="image", out_range=(0.0, 1.0))
        ax.imshow(ch_disp, cmap="gray")
        ax.set_title(f"MIP: {k}")
        ax.axis("off")

    # Sum/max MIP
    ax_sum = fig.add_subplot(gs[rows - 1 if rows > 0 else 0, cols - 1 if n else 0])
    ax_sum.imshow(sum_disp, cmap="gray")
    ax_sum.set_title("MIP: max across channels")
    ax_sum.axis("off")

    # Histogram
    axh = fig.add_subplot(gs[rows, :])
    flat = sum_img.ravel()
    flat = flat[np.isfinite(flat)]
    axh.hist(flat, bins=256, log=True)
    p1, p99 = np.percentile(flat, [1, 99]) if flat.size else (0, 1)
    axh.axvline(p1, color="r", linestyle="--", linewidth=1)
    axh.axvline(p99, color="r", linestyle="--", linewidth=1)
    axh.set_title(f"Histogram (red: 1st/99th pct). {title_stub}")
    axh.set_xlabel("Intensity")
    axh.set_ylabel("Count (log)")
    fig.tight_layout()

    saved_path = None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        saved_path = os.path.join(save_dir, f"{title_stub}_debug_mips.png")
        fig.savefig(saved_path, dpi=160)

    if show:
        plt.show()
    plt.close(fig)
    return saved_path


def _save_fullres_overlay_rgba(
    base_mip: np.ndarray, 
    mask: np.ndarray, 
    out_png: str, 
    out_tif: Optional[str] = None, 
    alpha: float = 0.5
) -> None:
    """
    Write a full-resolution overlay: the grayscale MIP under a partially
    transparent RED mask. No matplotlib; saved at native array size.
    """
    from skimage.exposure import rescale_intensity
    from skimage.io import imsave
    import numpy as np

    # 1) Display-scale the MIP to [0, 255] uint8 (ImageJ-friendly)
    disp = rescale_intensity(base_mip, in_range="image", out_range=(0.0, 1.0)).astype(np.float32)
    base = (np.clip(disp, 0, 1) * 255.0).astype(np.uint8)
    if base.ndim != 2:
        raise ValueError(f"Expected a 2-D MIP for overlay, got shape {base.shape}")

    H, W = base.shape
    rgb = np.stack([base, base, base], axis=-1)  # (H,W,3) uint8

    # 2) Alpha-blend a red layer where mask==True
    m = mask.astype(bool, copy=False)
    if m.shape != base.shape:
        raise ValueError(f"Mask shape {m.shape} != MIP shape {base.shape}")
    a = float(np.clip(alpha, 0.0, 1.0))

    # Existing pixel values as float32 for blending
    rf = rgb.astype(np.float32, copy=True)
    # Blend toward pure red [255,0,0] where mask==1:  new = (1-a)*base + a*target
    rf[m, 0] = (1.0 - a) * rf[m, 0] + a * 255.0  # R
    rf[m, 1] = (1.0 - a) * rf[m, 1] + a * 0.0    # G
    rf[m, 2] = (1.0 - a) * rf[m, 2] + a * 0.0    # B
    over_rgb = np.clip(rf, 0, 255).astype(np.uint8)

    # 3) Save full-res PNG (lossless) and optional TIFF
    imsave(out_png, over_rgb, check_contrast=False)
    if out_tif is not None:
        try:
            from tifffile import imwrite as _tifwrite
            _tifwrite(out_tif, over_rgb, photometric="rgb")
        except Exception:
            # Fallback: also write PNG if TIFF writer missing
            imsave(out_tif, over_rgb, check_contrast=False)


def find_largearea_nd2s(data_dir: str) -> List[str]:
    if not os.path.isdir(data_dir):
        return []
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(".nd2")]
    out = [os.path.join(data_dir, f) for f in files if LARGEAREA_TOKEN.lower() in f.lower()]
    return sorted(out)


def _read_marker_mip(nd2, c_index: int) -> np.ndarray:
    """Return a per-channel MIP; tolerate ND2 files without declared axes."""
    # --- Path A: current ND2Reader route ---
    try:
        Z = int(getattr(nd2, "sizes", {}).get("z", 1) or 1)
        if Z > 1:
            planes = [nd2.get_frame_2D(z=z, c=c_index).astype(np.float32) for z in range(Z)]
            return np.nanmax(np.stack(planes, axis=0), axis=0)
        return nd2.get_frame_2D(c=c_index).astype(np.float32)
    except Exception:
        pass  # fall through to the robust path

    # --- Path B: robust fallback via the 'nd2' library ---
    try:
        import nd2 as nd2lib
        # nd2.ND2File can open even when nd2reader sees "no axes"
        with nd2lib.ND2File(nd2.filename) as f:   # nd2.filename exists on ND2Reader
            arr = f.to_xarray()                   # dims like ('Y','X') or ('Z','Y','X','C',...)
            sel = arr
            # Select channel if present; otherwise take as-is
            for dim in ("c", "C", "channel", "Channel"):
                if dim in sel.dims:
                    sel = sel.isel({dim: int(c_index)})
                    break
            # Max over Z if present
            for dim in ("z", "Z"):
                if dim in sel.dims:
                    sel = sel.max(dim)
                    break
            # Squeeze and return as numpy (keep as float32)
            return sel.squeeze().values.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Axis-less ND2 fallback failed: {e}")


def _build_tissue_mask(sum_image: np.ndarray, _segkw: Dict[str, object]) -> np.ndarray:
    """Tissue mask used for BOTH edge detection and counting:
       mask[y,x] = 1 if composite MIP intensity >= 20, else 0.
    """
    arr = sum_image.astype(np.float32)
    # treat NaNs as background for the purpose of 'is tissue?'
    arr = np.nan_to_num(arr, nan=0.0)
    arr = gaussian(arr, sigma=2.0, preserve_range=True)

    mask = (arr >= TISSUE_INTENSITY_MIN)

    # Minimal cleanup to keep ribbon contiguous without inflating it
    mask = morphology.binary_closing(mask, morphology.disk(3))
    mask = morphology.remove_small_holes(mask, area_threshold=1500)

    return mask.astype(bool, copy=False)


def _infer_superficial_axis(sum_image: np.ndarray) -> ImageAxis:
    """
    Infer superficial edge and the average background gap depth.

    Steps:
      1) Make a simple tissue mask from the MIP: mask[y,x] = 1 if intensity >= 20, else 0.
      2) Consider only the two *shorter* edges of the rectangle:
           - If H <= W → shorter edges are TOP & BOTTOM
           - Else       → shorter edges are LEFT & RIGHT
         Pick the edge whose *half-image* on that side contains the most zeros.
      3) For each pixel along the chosen edge, trace inward along the normal
         until a nonzero pixel is found; record that trace length in pixels.
         The mean of these lengths is start_px (cortex begins there).
      4) depth_px is the raw distance from the chosen edge inward (no offset).
         Downstream we will subtract start_px to get cortical depth.
    """
    H, W = sum_image.shape
    img = np.nan_to_num(sum_image.astype(np.float32), nan=0.0)

    # (1) fixed threshold mask for tissue presence
    mask = (img >= TISSUE_INTENSITY_MIN).astype(np.uint8)

    # (2) pick among the two shorter edges
    if W <= H:  # top vs bottom
        top_half    = mask[:H // 2, :]
        bottom_half = mask[H // 2:, :]
        top_zeros    = np.count_nonzero(top_half == 0)
        bottom_zeros = np.count_nonzero(bottom_half == 0)
        side = "top" if top_zeros > bottom_zeros else "bottom"
    else:        # left vs right
        left_half  = mask[:, :W // 2]
        right_half = mask[:, W // 2:]
        left_zeros  = np.count_nonzero(left_half == 0)
        right_zeros = np.count_nonzero(right_half == 0)
        side = "left" if left_zeros > right_zeros else "right"

    # (3) trace inward from the chosen edge, measuring gap to first nonzero
    if side == "top":
        # for each column, first row index where mask==1
        first = np.argmax(mask > 0, axis=0)
        # columns with all-zero mask should contribute full height
        no_hit = (mask.max(axis=0) == 0)
        first = first.astype(np.float32)
        first[no_hit] = float(H)
        start_px = float(first.mean())

        depth_px = np.tile(np.arange(H, dtype=np.float32)[:, None], (1, W))

    elif side == "bottom":
        # reverse in rows to trace upward
        rev = mask[::-1, :]
        first = np.argmax(rev > 0, axis=0)
        no_hit = (rev.max(axis=0) == 0)
        first = first.astype(np.float32)
        first[no_hit] = float(H)
        start_px = float(first.mean())

        depth_px = np.tile(np.arange(H - 1, -1, -1, dtype=np.float32)[:, None], (1, W))

    elif side == "left":
        first = np.argmax(mask > 0, axis=1)
        no_hit = (mask.max(axis=1) == 0)
        first = first.astype(np.float32)
        first[no_hit] = float(W)
        start_px = float(first.mean())

        depth_px = np.tile(np.arange(W, dtype=np.float32)[None, :], (H, 1))

    else:  # right
        rev = mask[:, ::-1]
        first = np.argmax(rev > 0, axis=1)
        no_hit = (rev.max(axis=1) == 0)
        first = first.astype(np.float32)
        first[no_hit] = float(W)
        start_px = float(first.mean())

        depth_px = np.tile(np.arange(W - 1, -1, -1, dtype=np.float32)[None, :], (H, 1))

    logger.info("[Axis] Superficial=%s | start_px=%.2f (H=%d, W=%d)", side, start_px, H, W)
    return ImageAxis(side=side, depth_px=depth_px, start_px=start_px)


def _segment_cells_2d(img2d: np.ndarray, segkw: Dict[str, object]) -> np.ndarray:
    return process_fluorescence_channel(
        img2d,
        cell_size=int(segkw["cell_size"]),
        min_size=int(segkw["min_size"]),
        closing_radius=int(segkw["closing_radius"]),
        gaussian_sigma=float(segkw["gaussian_sigma"]),
        fill_holes=bool(segkw["fill_holes"]),
        threshold_method=str(segkw["threshold_method"]),
        offset=float(segkw["offset"]),
        exclude_dark_regions=bool(segkw["exclude_dark_regions"]),
        dark_threshold=float(segkw["dark_threshold"]),
        min_hole_size=int(segkw["min_hole_size"]),
        debug=bool(segkw.get("debug", False)),
    ).astype(bool, copy=False)


def _assign_layers_by_centroid(mask: np.ndarray, depth_um: np.ndarray, layer_widths_um: List[float]) -> Tuple[pd.Series, pd.Series]:
    """Return (layer_idx, region_label) per connected component.
    layer_idx: 1..N for cortical layers, N+1 for WM; 0 for undefined (shouldn’t occur).
    region_label: "L1".."L6" or "WM".
    """
    labs = label(mask, connectivity=1)
    boundaries = np.cumsum(np.asarray(layer_widths_um, dtype=float))
    n_layer = len(layer_widths_um)
    lyr_out: Dict[int, int] = {}
    reg_out: Dict[int, str] = {}
    for r in regionprops(labs):
        cy, cx = map(int, np.round(r.centroid))
        cy = np.clip(cy, 0, depth_um.shape[0]-1)
        cx = np.clip(cx, 0, depth_um.shape[1]-1)
        d = float(depth_um[cy, cx])
        lyr = int(np.searchsorted(boundaries, d, side="right")) + 1  # 1..N+1
        if lyr <= n_layer:
            reg = f"L{lyr}"
        else:
            lyr = n_layer + 1
            reg = "WM"
        lyr_out[r.label] = lyr
        reg_out[r.label] = reg
    return pd.Series(lyr_out, name="Layer"), pd.Series(reg_out, name="Region")


def _layer_index_map(depth_um: np.ndarray, layer_widths_um: List[float]) -> np.ndarray:
    boundaries = np.cumsum(np.asarray(layer_widths_um, dtype=float))
    return (np.searchsorted(boundaries, depth_um, side="right") + 1).astype(np.int16, copy=False)


def _save_debug_overlays(
    out_dir: str,
    file_stub: str,
    composite_rgb: np.ndarray,
    tissue_mask: np.ndarray,
    axis: ImageAxis,
    px_um: float,
    layer_widths_um: List[float],
) -> Tuple[str, str]:
    """Save two overlays:
    (1) Tissue & superficial side arrow; (2) Layer boundary lines on composite.
    Returns (path1, path2).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Normalize composite for display
    comp = np.clip(composite_rgb.astype(np.float32), 0, 1)
    if comp.max() > 1.0:
        comp = (comp / comp.max())
    H, W = tissue_mask.shape
    # Downscale very large images for overlays to avoid huge RGBA buffers
    MAX_DEBUG_PIXELS = 12_000_000  # ~12 MP cap for display
    scale = min(1.0, (MAX_DEBUG_PIXELS / max(1, H * W)) ** 0.5)
    
    comp_disp = comp
    tm_disp = tissue_mask
    if scale < 1.0:
        from skimage.transform import rescale
        comp_disp = rescale(comp, scale, channel_axis=2, anti_aliasing=True, preserve_range=True).astype(np.float32)
        tm_disp = rescale(tissue_mask.astype(np.float32), scale, anti_aliasing=False, preserve_range=True) > 0.5
        H, W = tm_disp.shape
    # IMPORTANT: scale layer offsets to match the downscaled display geometry
    start_px_disp = float(axis.start_px) * float(scale)
    px_um_disp = float(px_um) / float(scale)  # µm/px on the display canvas

    # Figure 1: B/W tissue mask only (no composite), with superficial arrow
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.imshow(tm_disp, cmap="gray", vmin=0, vmax=1)
    ax1.set_title(f"Tissue mask (threshold ≥ {int(TISSUE_INTENSITY_MIN)}) — superficial: {axis.side}")
    # draw arrow along the superficial edge pointing inward
    if axis.side == "top":
        ax1.arrow(W*0.5, 10, 0, H*0.1, color="yellow", width=2, head_width=40, length_includes_head=True)
    elif axis.side == "bottom":
        ax1.arrow(W*0.5, H-10, 0, -H*0.1, color="yellow", width=2, head_width=40, length_includes_head=True)
    elif axis.side == "left":
        ax1.arrow(10, H*0.5, W*0.1, 0, color="yellow", width=2, head_width=40, length_includes_head=True)
    else:  # right
        ax1.arrow(W-10, H*0.5, -W*0.1, 0, color="yellow", width=2, head_width=40, length_includes_head=True)
    ax1.axis("off")
    p1 = os.path.join(out_dir, f"{file_stub}_overlay_superficial.png")
    fig1.savefig(p1, dpi=180, bbox_inches="tight")
    plt.close(fig1)

    # Figure 2: layer boundaries
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.imshow(comp_disp)
    # rows/cols computations keep using H, W from above (already rescaled if needed)
    cum = np.cumsum(np.asarray(layer_widths_um, dtype=float))
    # convert layer cumulative depths (µm) to display pixels and add scaled start offset
    lines_px_disp = (cum / max(px_um_disp, 1e-9)) + start_px_disp
    if axis.side in ("top", "bottom"):
        # horizontal lines
        rows = lines_px_disp if axis.side == "top" else (H - lines_px_disp)
        for r in rows:
            rr = float(np.clip(r, 0, H-1))
            ax2.plot([0, W-1], [rr, rr], linestyle="--", linewidth=2)
    else:
        # vertical lines
        cols = lines_px_disp if axis.side == "left" else (W - lines_px_disp)
        for c in cols:
            cc = float(np.clip(c, 0, W-1))
            ax2.plot([cc, cc], [0, H-1], linestyle="--", linewidth=2)

    ax2.set_title("Layer boundaries (dashed)")
    ax2.axis("off")
    p2 = os.path.join(out_dir, f"{file_stub}_overlay_layers.png")
    fig2.savefig(p2, dpi=180, bbox_inches="tight")
    plt.close(fig2)

    return p1, p2


# ----------------------------
# Core analysis
# ----------------------------

def analyze_largearea_nd2(nd2_path: str, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze one ND2 large-area scan.

    Returns
    -------
    counts_df : DataFrame with columns [File, Layer (1..6,7=WM), Region, <marker columns...>, ALL]
    objects_df: Per-object table: [File, Marker, y, x, Depth_um, Layer, Region, Area_px]
    """
    # Base segmentation knobs come only from DEFAULT_SEGMENT_KW.
    # Per-marker tweaks are applied via PER_MARKER_SEG_KW below.
    base_segkw = dict(DEFAULT_SEGMENT_KW)

    # Resolve channel_map: use all fluorescence channels with non-None indices
    ch_map: Dict[str, Optional[int]] = dict(config.get("channel_map", {}))
    usable = {m: int(ci) for m, ci in ch_map.items() if ci is not None}
    
    # NEW: choose all usable channels EXCEPT the excluded/non-cell ones
    cell_markers = [m for m in usable.keys() if not _is_noncell_marker(m)]
    
    # Sensible fallback: if everything was excluded, use all channels but warn.
    if not cell_markers:
        cell_markers = list(usable.keys())
        logger.warning(
            "[LargeArea] All channels matched the non-cell exclusion; "
            "falling back to all usable channels: %s", cell_markers
        )

    if not usable:
        raise RuntimeError(f"No usable fluorescence channels in channel_map for {os.path.basename(nd2_path)}")

    with ND2Reader(nd2_path) as nd2:
        px_um = _get_pixel_size_microns(nd2)
        # Build per-marker MIPs and composite
        imgs: Dict[str, np.ndarray] = {}
        for marker, ci in usable.items():
            try:
                imgs[marker] = _read_marker_mip(nd2, ci)
            except Exception as exc:
                logger.warning("[Load] %s (c=%s) failed: %s", marker, ci, exc)
        if not imgs:
            raise RuntimeError(f"No channels could be read from {os.path.basename(nd2_path)}")

    # Tissue extent and axis inference use a bright composite (max across markers)
    sum_img = None
    for _k, _img in imgs.items():
        cur = _img.astype(np.float32, copy=False)
        if sum_img is None:
            sum_img = cur.copy()
        else:
            # nan-aware, in-place maximum (avoid 3-D stack allocation)
            np.maximum(sum_img, cur, out=sum_img, where=np.isfinite(cur))
            # If sum_img still has NaNs where cur is finite, fill from cur
            nan_locs = ~np.isfinite(sum_img)
            if nan_locs.any():
                sum_img[nan_locs] = cur[nan_locs]
    sum_img = np.nan_to_num(sum_img, nan=0.0)
    
    # -- SAVE the composite MIP used for tissue mask so we can inspect it in ImageJ --
    from tifffile import imwrite
    
    out_img_dir = ensure_subdirectory(os.path.dirname(nd2_path), "LargeArea/Images")
    stub = os.path.splitext(os.path.basename(nd2_path))[0]
    
    # 1) Raw float32 (no scaling)
    raw_tif = os.path.join(out_img_dir, f"{stub}_compositeMIP_raw32.tif")
    imwrite(raw_tif, sum_img.astype(np.float32))
    
    # 2) Percentile-scaled to uint16 (easy to view)
    finite = np.isfinite(sum_img)
    p1, p999 = (np.percentile(sum_img[finite], [1.0, 99.9]) if finite.any() else (0.0, 1.0))
    den = max(p999 - p1, 1e-6)
    scaled16 = np.clip((sum_img - p1) / den, 0, 1)
    scaled_tif = os.path.join(out_img_dir, f"{stub}_compositeMIP_scaled16.tif")
    imwrite(scaled_tif, (scaled16 * 65535).astype(np.uint16))
    
    # 3) “<20 set to zero” diagnostic (same percentile scaling for display)
    thr20 = sum_img.copy()
    thr20[~finite] = 0.0
    thr20[thr20 < TISSUE_INTENSITY_MIN] = 0.0
    thr20_disp = np.clip((thr20 - p1) / den, 0, 1)
    thr20_tif = os.path.join(out_img_dir, f"{stub}_compositeMIP_thr20zero16.tif")
    imwrite(thr20_tif, (thr20_disp * 65535).astype(np.uint16))
    
    # Helpful logging
    logger.info("[MIP] Saved composite MIPs: raw32=%s | scaled16=%s | thr20zero16=%s | p1=%.2f p99.9=%.2f",
                raw_tif, scaled_tif, thr20_tif, p1, p999)
    
    # keep a light reference for composite overlays later
    imgs_for_comp = dict(imgs)

    _debug_show_mips(sum_img, imgs, title_stub=stub, save_dir=out_img_dir, show=False)

    tissue_mask = _build_tissue_mask(sum_img, base_segkw)
    axis = _infer_superficial_axis(sum_img)
    depth_um = np.clip(axis.depth_px - axis.start_px, 0, None) * float(px_um)
    
    # Per-pixel layer indices 1..6, 7=WM
    layer_idx_map = _layer_index_map(depth_um, LAYER_WIDTHS_UM)

    # Per-marker segmentation & object extraction restricted to tissue
    rows: List[dict] = []
    for marker in cell_markers:
        img = imgs[marker]
        
        # Scale intensities to 12-bit (0–4096) for display-like brightness
        disp = rescale_intensity(img, in_range='image', out_range=(0.0, 1.0)).astype(np.float32)
        # Display-only NaN fill to avoid black tiles in PNGs
        if not np.isfinite(disp).all():
            med = np.nanmedian(disp)
            # If the whole image is NaN, default to 0
            if not np.isfinite(med):
                med = 0.0
            disp[~np.isfinite(disp)] = med
        scaled = (disp * 4096).astype(np.uint16)
        mip_path = os.path.join(out_img_dir, f"{stub}_{marker}_mip.png")
        imsave(mip_path, scaled, check_contrast=False)
    
        # --- Build per-marker seg kwargs (base + specific overrides) ---
        mk_segkw = dict(base_segkw, **PER_MARKER_SEG_KW.get(marker, {}))
        
        # --- TUJ-only: layer-specific segmentation to prevent superficial overmerge ---
        if str(marker).strip().upper() == "TUJ":
            H, W = img.shape
            union_mask = np.zeros((H, W), dtype=bool)
        
            # Dynamically handle whatever layers have overrides (e.g., 1..4, or 1..5)
            override_layers = sorted(int(k) for k in LAYER_SEGMENT_OVERRIDES_TUJ.keys())
            handled = np.zeros((H, W), dtype=bool)
        
            for lyr in override_layers:
                m = (layer_idx_map == lyr)
                if tissue_mask is not None:
                    m &= tissue_mask
                if not m.any():
                    continue
        
                # TUJ defaults + per-layer override
                segkw_layer = dict(mk_segkw)
                segkw_layer.update(LAYER_SEGMENT_OVERRIDES_TUJ.get(lyr, {}))
        
                # Log and segment full frame (stable local neighborhoods), then clamp to layer
                logger.info("[TUJ] Layer %d overrides: %s", lyr,
                            {k: segkw_layer[k] for k in ("offset","closing_radius","gaussian_sigma","fill_holes","min_size")
                             if k in segkw_layer})
        
                mask_layer = _segment_cells_2d(img, segkw_layer) & m
                union_mask |= mask_layer
                handled |= m
        
            # Fallback: any tissue pixels in layers NOT overridden use default TUJ params
            m_fallback = tissue_mask & (~handled) if tissue_mask is not None else (~handled)
            if m_fallback.any():
                mask_fb = _segment_cells_2d(img, mk_segkw) & m_fallback
                union_mask |= mask_fb
        
            cell_mask = union_mask
        else:
            # Non-TUJ markers: original path
            cell_mask = _segment_cells_2d(img, mk_segkw)

        mask_path = os.path.join(out_img_dir, f"{stub}_{marker}_mask.png")
        imsave(mask_path, (cell_mask.astype(np.uint16) * 65535), check_contrast=False)
        
        # --- NEW: save full-res alpha overlay (red) on top of the MIP ---
        overlay_png = os.path.join(out_img_dir, f"{stub}_{marker}_overlay.png")
        overlay_tif = os.path.join(out_img_dir, f"{stub}_{marker}_overlay.tif")
        _save_fullres_overlay_rgba(disp, cell_mask, overlay_png, out_tif=overlay_tif, alpha=0.5)
    
        # --- Apply tissue mask before downstream counting ---
        cell_mask &= tissue_mask
        labs = label(cell_mask, connectivity=1)
        layer_idx, region_lbl = _assign_layers_by_centroid(cell_mask, depth_um, LAYER_WIDTHS_UM)
        for r in regionprops(labs):
            cy, cx = map(int, np.round(r.centroid))
            cy = np.clip(cy, 0, depth_um.shape[0]-1)
            cx = np.clip(cx, 0, depth_um.shape[1]-1)
            rows.append(dict(
                File=os.path.basename(nd2_path),
                Marker=marker,
                y=int(cy), x=int(cx),
                Depth_um=float(depth_um[cy, cx]),
                Layer=int(layer_idx.get(r.label, 0)),
                Region=str(region_lbl.get(r.label, "")),
                Area_px=int(r.area),
            ))
            
        # --- free large per-marker arrays before next marker ---
        try:
            del cell_mask, labs, disp, scaled
        except NameError:
            pass
        if marker in imgs:
            del imgs[marker]

    objects_df = pd.DataFrame(rows)

    # Build counts per Layer (1..N, WM=N+1) × marker (cell markers only)
    n_layer = len(LAYER_WIDTHS_UM)
    layer_levels = list(range(1, n_layer + 2))  # +1 for WM
    region_levels = [f"L{i}" for i in range(1, n_layer+1)] + ["WM"]
    
    if not rows:
        # No objects at all → empty counts with cell marker columns
        counts_df = pd.DataFrame({
            "File": [os.path.basename(nd2_path)] * len(layer_levels),
            "Layer": layer_levels,
            "Region": region_levels,
        })
        for m in cell_markers:
            counts_df[m] = 0
        counts_df["ALL"] = 0
    else:
        df = pd.DataFrame(rows)
        df = df[df["Layer"].isin(layer_levels)].copy()
        mapping = {i: f"L{i}" for i in range(1, n_layer + 1)}
        mapping[n_layer + 1] = "WM"
    
        pivot = (df.pivot_table(index=["Layer", "Region"], columns="Marker",
                                values="Area_px", aggfunc="count")
                   .fillna(0).astype(int))
    
        aligned = [(i, mapping[i]) for i in (list(range(1, n_layer + 1)) + [n_layer + 1])]
        idx = pd.MultiIndex.from_tuples(aligned, names=["Layer", "Region"])
        pivot = pivot.reindex(idx, fill_value=0)
    
        # Ensure all declared cell markers are present as columns, even if 0
        for m in cell_markers:
            if m not in pivot.columns:
                pivot[m] = 0
    
        pivot["ALL"] = pivot.sum(axis=1)
        counts_df = pivot.reset_index().sort_values("Layer")
        counts_df.insert(0, "File", os.path.basename(nd2_path))

    # Save overlays (composite + boundaries)
    out_img_dir = ensure_subdirectory(os.path.dirname(nd2_path), "LargeArea/Images")
    # Build display composite; avoid large RGB on huge mosaics
    try:
        if sum_img.size > 20_000_000:  # ~20 MP threshold
            gray = rescale_intensity(sum_img, in_range="image", out_range=(0.0, 1.0)).astype(np.float32)
            composite = np.dstack([gray, gray, gray])
        else:
            # Use cached copy even if imgs was partially freed
            composite = composite_fluorescence(imgs_for_comp, config)  # float RGB in [0, 1]
    except Exception as exc:
        logger.warning("[LargeArea] Composite RGB fallback due to %s", exc)
        gray = rescale_intensity(sum_img, in_range="image", out_range=(0.0, 1.0)).astype(np.float32)
        composite = np.dstack([gray, gray, gray])

    stub = os.path.splitext(os.path.basename(nd2_path))[0]
    _save_debug_overlays(out_img_dir, stub, composite, tissue_mask, axis, float(px_um), LAYER_WIDTHS_UM)

    return counts_df, objects_df


# ----------------------------
# Batch/CLI wrapper (Spyder-friendly)
# ----------------------------

def run_batch(
    config_py: str,
    data_dir: Optional[str] = None,
    out_name: str = "LargeArea_Counts.xlsx",
) -> tuple[str, int, int]:
    """
    Returns:
      (xlsx_path, n_ok, n_fail)
      Exit policy (handled in __main__):
        - if n_ok == 0: exit code 1 (all failed)
        - elif n_fail > 0: exit code 2 (partial failure)
        - else: 0 (success)
    """
    cfg = load_config(config_py)
    base_dir = data_dir or cfg["paths"]["data_directory"]

    out_dir = ensure_subdirectory(base_dir, "LargeArea")
    files = find_largearea_nd2s(base_dir)
    if not files:
        raise FileNotFoundError(f"No ND2 files with token '{LARGEAREA_TOKEN}' in {base_dir}")

    all_counts: List[pd.DataFrame] = []
    all_objects: List[pd.DataFrame] = []
    failed: List[tuple[str, str]] = []   # (filename, error)
    succeeded: List[str] = []

    import gc
    for nd2 in files:
        try:
            counts, objs = analyze_largearea_nd2(nd2, cfg)
            all_counts.append(counts)
            all_objects.append(objs)
            succeeded.append(os.path.basename(nd2))
        except Exception as exc:
            logger.error("[LargeArea] Failed on %s: %s", os.path.basename(nd2), exc)
            failed.append((os.path.basename(nd2), str(exc)))
        finally:
            # Aggressively drop GUI/display buffers & Python objects between files
            import matplotlib.pyplot as _plt
            _plt.close('all')
            gc.collect()

    # Concatenate (may be empty)
    counts_df = pd.concat(all_counts, ignore_index=True) if all_counts else pd.DataFrame()
    objects_df = pd.concat(all_objects, ignore_index=True) if all_objects else pd.DataFrame()

    # Write Excel (even if empty) so there's always an artifact to inspect
    xlsx_path = os.path.join(out_dir, out_name)
    with pd.ExcelWriter(xlsx_path) as writer:
        (counts_df if not counts_df.empty else pd.DataFrame(columns=["File","Layer","Region","ALL"])) \
            .to_excel(writer, sheet_name="Counts", index=False)
        (objects_df if not objects_df.empty else pd.DataFrame(columns=["File","Marker","y","x","Depth_um","Layer","Region","Area_px"])) \
            .to_excel(writer, sheet_name="Objects", index=False)

    # Write a human-readable status file next to the Excel
    status_path = os.path.join(out_dir, "_STATUS.txt")
    with open(status_path, "w", encoding="utf-8") as fh:
        fh.write(f"Folder: {os.path.basename(base_dir)}\n")
        fh.write(f"Output: {xlsx_path}\n")
        fh.write(f"Succeeded: {len(succeeded)} | Failed: {len(failed)} | Total: {len(files)}\n\n")
        if succeeded:
            fh.write("[Succeeded ND2s]\n")
            for s in succeeded:
                fh.write(f"  - {s}\n")
            fh.write("\n")
        if failed:
            fh.write("[Failed ND2s]\n")
            for nm, err in failed:
                fh.write(f"  - {nm}: {err}\n")

    n_ok = len(succeeded)
    n_fail = len(failed)
    logger.info("[LargeArea] Wrote %s (ok=%d, fail=%d)", xlsx_path, n_ok, n_fail)
    return xlsx_path, n_ok, n_fail


if __name__ == "__main__":
    import sys

    # Local defaults for manual/Spyder runs (safe fallback)
    DEFAULT_CONFIG_PY = r"D:\OneDrive - Stanford\Research Documents\Python Scripts\config_files\config_AD4e.py"
    DEFAULT_DATA_DIR  = r"D:\OneDrive - Stanford\Research Documents\AD Project\2025\AD4e"

    # Prefer CLI args passed by the batch driver:  sys.argv[1]=config, sys.argv[2]=data_dir
    if len(sys.argv) >= 3:
        config_py = sys.argv[1]
        data_dir  = sys.argv[2]
    else:
        print("[LargeArea] No CLI args detected; using local defaults.")
        config_py = DEFAULT_CONFIG_PY
        data_dir  = DEFAULT_DATA_DIR

    try:
        # Use folder name in the output filename when running via CLI; plain name on manual runs.
        base = os.path.basename(os.path.normpath(data_dir))
        outname = f"LargeArea_Counts_{base}.xlsx" if len(sys.argv) >= 3 else "LargeArea_Counts.xlsx"

        out_path, n_ok, n_fail = run_batch(config_py, data_dir, out_name=outname)
        print(f"[LargeArea] Done → {out_path} (ok={n_ok}, fail={n_fail})")

        # Exit codes: 0=success, 2=partial failure, 1=all failed
        if n_ok == 0:
            sys.exit(1)
        elif n_fail > 0:
            sys.exit(2)
        else:
            sys.exit(0)
    except Exception as e:
        print(f"[LargeArea] ERROR: {e}")
        sys.exit(1)
