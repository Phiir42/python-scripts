from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _normalize_img(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).any():
        return np.zeros_like(x, dtype=float)
    lo, hi = np.percentile(x[np.isfinite(x)], [1, 99])
    if hi <= lo:
        hi = lo + 1e-9
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0, 1)


def save_alignment_triptych(
    out_path: str | Path,
    hs_img,
    fluor_img,
    cars_img,
    label: str = "",
    chosen_z: int | None = None,
    corr_value: float | None = None,
    show: bool = False,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    hs = _normalize_img(hs_img)
    fl = _normalize_img(fluor_img)
    ca = _normalize_img(cars_img)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(hs, cmap="gray")
    axs[0].set_title("Hyperspec CARS @ ~2850 cm⁻¹")
    axs[1].imshow(fl, cmap="gray")
    axs[1].set_title(f"Matched Fluorescence z={chosen_z}")
    axs[2].imshow(ca, cmap="gray")
    axs[2].set_title(f"Matched CARS z={chosen_z}")
    for ax in axs:
        ax.axis("off")

    title = "Alignment debug"
    if label:
        title += f" — {label}"
    if corr_value is not None:
        title += f"  |  corr={corr_value:.3f}"
    fig.suptitle(title, y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"[DEBUG] alignment triptych saved → {out_path}")
