"""
Spyder-friendly TUJ parameter sweep helper
==========================================

This script runs a local-thresholding sweep on the TUJ channel for a single
ND2 dataset, using the existing TUJ MIP saved by your main pipeline.

Usage
-----
Simply open this file in Spyder and press **Run (F5)**.
The script will:
    • Locate the TUJ MIP and composite tissue mask in the
      <folder>/LargeArea/Images directory.
    • Sweep across the specified `offsets` and `cell_sizes`.
    • Save full-resolution overlays and masks in:
          <folder>/LargeArea/TUJ_sweep/

Default dataset and parameter grid can be edited near the bottom of this file
under the `if __name__ == "__main__":` block.

Requires:
    - segmentation.py   (in lipid_analysis/)
    - largearea_layers.py (in the same folder as this file)
"""

import os
import sys
import glob
import numpy as np
from typing import Iterable, Tuple, Optional

from skimage.io import imread, imsave
from skimage.exposure import rescale_intensity

# Add script dir AND its parent (so sibling package 'lipid_analysis' is importable)
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.abspath(os.path.join(_HERE, ".."))
    if _HERE not in sys.path:
        sys.path.append(_HERE)
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
except Exception:
    pass

try:
    import largearea_layers as la  # same folder as this helper
except Exception as e:
    la = None
    print(f"[WARN] Could not import largearea_layers: {e}")

# Prefer the package path (your segmentation.py lives in /lipid_analysis)
try:
    from lipid_analysis.segmentation import process_fluorescence_channel
except Exception:
    # Fallback if someone runs this with a flat layout
    try:
        from segmentation import process_fluorescence_channel
    except Exception as e:
        print(f"[ERROR] Could not import process_fluorescence_channel: {e}")
        raise

def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def _find_mip(folder: str, stub: str) -> Optional[str]:
    """Try to find an existing TUJ MIP saved by the pipeline."""
    images_dir = os.path.join(folder, "LargeArea", "Images")
    patterns = [
        os.path.join(images_dir, f"{stub}_TUJ_mip.png"),
        os.path.join(images_dir, f"{stub}_TUJ_mip.tif"),
        os.path.join(images_dir, f"{stub}_TUJ_mip.tiff"),
    ]
    for p in patterns:
        if os.path.exists(p):
            return p
    globs = glob.glob(os.path.join(images_dir, "*TUJ_mip.*"))
    return globs[0] if globs else None

def _overlay_rgba(base_mip: np.ndarray, mask: np.ndarray,
                  color: Tuple[int,int,int]=(0,255,255), alpha: float=0.65) -> np.ndarray:
    """Create a full-res RGB overlay (no matplotlib)."""
    if base_mip.ndim != 2:
        raise ValueError("base_mip must be 2D")
    if mask.shape != base_mip.shape:
        raise ValueError(f"mask shape {mask.shape} != base {base_mip.shape}")

    disp = rescale_intensity(base_mip, in_range="image", out_range=(0, 1)).astype(np.float32)
    base8 = (disp * 255.0).astype(np.uint8)
    rgb = np.stack([base8, base8, base8], axis=-1).astype(np.float32)

    r,g,b = color
    a = float(np.clip(alpha, 0.0, 1.0))
    m = mask.astype(bool)
    rgb[m, 0] = (1-a)*rgb[m, 0] + a*r
    rgb[m, 1] = (1-a)*rgb[m, 1] + a*g
    rgb[m, 2] = (1-a)*rgb[m, 2] + a*b
    return np.clip(rgb, 0, 255).astype(np.uint8)

def _load_tuj_mip(folder: str, filename: str):
    """Load TUJ MIP from Images; if missing and helpers exist, try to build it."""
    stub = os.path.splitext(os.path.basename(filename))[0]
    mip_path = _find_mip(folder, stub)
    if mip_path and os.path.exists(mip_path):
        arr = imread(mip_path)
        if arr.ndim == 3:
            arr = (0.2126*arr[...,0] + 0.7152*arr[...,1] + 0.0722*arr[...,2]).astype(np.float32)
        else:
            arr = arr.astype(np.float32)
        return arr, stub, mip_path

    if la is not None and hasattr(la, "load_marker_mip"):
        mip = la.load_marker_mip(os.path.join(folder, filename), marker="TUJ")
        return mip.astype(np.float32), stub, None

    raise FileNotFoundError("Could not locate TUJ MIP and no helper available to build it.")

def preview_defaults(folder: str, filename: str) -> None:
    """Print which files will be used (MIP & tissue diagnostic) for sanity checks."""
    try:
        _, stub, mip_found = _load_tuj_mip(folder, filename)
    except Exception as e:
        stub = os.path.splitext(os.path.basename(filename))[0]
        mip_found = None
        print(f"[preview] TUJ MIP not found: {e}")
    images_dir = os.path.join(folder, "LargeArea", "Images")
    thr_path = os.path.join(images_dir, f"{stub}_compositeMIP_thr20zero16.tif")
    print(f"[preview] folder: {folder}")
    print(f"[preview] file: {filename}")
    print(f"[preview] stub: {stub}")
    print(f"[preview] TUJ MIP exists: {bool(mip_found)} → {mip_found}")
    print(f"[preview] tissue diagnostic: {os.path.exists(thr_path)} → {thr_path}")

def run_sweep(
    folder: str,
    filename: str,
    offsets: Iterable[float],
    cell_sizes: Iterable[int],
    gaussian_sigmas: Iterable[float],
    closing_radii: Iterable[int],
    alpha: float=0.65,
    color: Tuple[int,int,int]=(0, 255, 255),
    outdir: Optional[str]=None,
    force_local: bool=True,
    fill_holes: bool=True,
    min_size: int=140,
    apply_tissue_mask: bool=True,
) -> str:
    """
    Run a TUJ thresholding parameter sweep inside Spyder/IPython.

    Returns the output directory path.
    """
    outdir = outdir or os.path.join(folder, "LargeArea", "TUJ_sweep")
    _ensure_dir(outdir)

    tuj_mip, stub, mip_path = _load_tuj_mip(folder, filename)
    if mip_path:
        print(f"[INFO] Using existing TUJ MIP: {mip_path}")
    else:
        print("[INFO] TUJ MIP built on-the-fly")

    # Start with TUJ defaults if available
    seg_defaults = {}
    if la is not None and hasattr(la, "PER_MARKER_SEG_KW"):
        seg_defaults = dict(la.PER_MARKER_SEG_KW.get("TUJ", {}))

    if force_local:
        seg_defaults["threshold_method"] = "local"

    seg_defaults["min_size"] = min_size
    seg_defaults["fill_holes"] = fill_holes
    seg_defaults["exclude_dark_regions"] = seg_defaults.get("exclude_dark_regions", False)

    # Optional tissue mask
    tissue_mask = None
    if apply_tissue_mask:
        try:
            images_dir = os.path.join(folder, "LargeArea", "Images")
            thr_path = os.path.join(images_dir, f"{stub}_compositeMIP_thr20zero16.tif")
            if os.path.exists(thr_path):
                thr = imread(thr_path)
                if thr.ndim == 3:
                    thr = (0.2126*thr[...,0] + 0.7152*thr[...,1] + 0.0722*thr[...,2]).astype(np.float32)
                tissue_mask = thr > 0
                print(f"[INFO] Found tissue mask diagnostic: {thr_path}")
        except Exception as e:
            print(f"[WARN] Could not load tissue mask diagnostic: {e}")

    for off in offsets:
        for cs in cell_sizes:
            for gs in gaussian_sigmas:
                for cr in closing_radii:
                    segkw = dict(seg_defaults)
                    segkw["offset"] = float(off)
                    segkw["cell_size"] = int(cs)
                    segkw["gaussian_sigma"] = float(gs)
                    segkw["closing_radius"] = int(cr)
    
                    mask = process_fluorescence_channel(tuj_mip, **segkw)
    
                    if tissue_mask is not None and tissue_mask.shape == mask.shape:
                        mask = mask & tissue_mask
    
                    tag = f"off{off:.2f}_cell{cs}_gs{gs:.2f}_cr{cr}"
                    over = _overlay_rgba(tuj_mip, mask, color=color, alpha=alpha)
    
                    out_png = os.path.join(outdir, f"{stub}_TUJ_overlay_{tag}.png")
                    out_tif = os.path.join(outdir, f"{stub}_TUJ_overlay_{tag}.tif")
                    out_mask = os.path.join(outdir, f"{stub}_TUJ_mask_{tag}.png")
                    imsave(out_png, over, check_contrast=False)
                    try:
                        imsave(out_tif, over, check_contrast=False)
                    except Exception as e:
                        print(f"[WARN] TIFF save failed for {tag}: {e}")
                    imsave(out_mask, (mask.astype(np.uint8) * 255), check_contrast=False)
                    print(f"[OK] Saved {tag} → {out_png}")

    print(f"[DONE] Sweep results → {outdir}")
    return outdir

if __name__ == "__main__":
    # ---- Edit these for your next dataset ----
    folder = r"D:\OneDrive - Stanford\Research Documents\AD Project\2025\AD4e"
    filename = "Control-S2218-DAPI-TUJ-LAMP2-40X-LargeArea.nd2"

    # parameter sweep grid
    offsets = (0.14,)
    cell_sizes = (800,)
    gaussian_sigmas = (0.3, 0.45, 0.6, 0.75, 0.9)
    closing_radii = (3,)

    # run directly
    print("\n[RUNNING TUJ SWEEP TEST]\n")
    run_sweep(
        folder=folder,
        filename=filename,
        offsets=offsets,
        cell_sizes=cell_sizes,
        alpha=0.65,
        color=(0, 255, 255),
        force_local=True,
        fill_holes=True,
        min_size=140,
        gaussian_sigmas=gaussian_sigmas,
        closing_radii=closing_radii,
        apply_tissue_mask=False,
    )
