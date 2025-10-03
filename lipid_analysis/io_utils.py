import os

import cv2
import pandas as pd

from .imaging import (
    blend_fluorescence_cars,
    composite_fluorescence,
    grayscale_autoscale,
)


def save_results_to_excel(results, summary, output_file):
    """(Unchanged docstring/logic from original.)"""
    results_df = pd.DataFrame(results)
    summary_df = pd.DataFrame(summary)

    if not results_df.empty and {"file_name", "cell_marker", "z_stack"}.issubset(
        results_df.columns
    ):
        results_df = results_df.sort_values(by=["file_name", "cell_marker", "z_stack"])
    if not summary_df.empty and {"file_name", "cell_marker", "z_stack"}.issubset(
        summary_df.columns
    ):
        summary_df = summary_df.sort_values(by=["file_name", "cell_marker", "z_stack"])

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with pd.ExcelWriter(output_file) as writer:
        results_df.to_excel(writer, sheet_name="Detailed Results", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)


def ensure_subdirectory(main_dir, sub_name="Images"):
    out_dir = os.path.join(main_dir, sub_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_composite_images(fluor_images, cars_image, config_dict, main_dir, file_stub):
    """(Unchanged docstring/logic from original.)"""
    out_dir = ensure_subdirectory(main_dir, "Images")
    composite_fluor = composite_fluorescence(fluor_images, config_dict)
    cars_gray_8bit = grayscale_autoscale(cars_image)
    fluor_cars_blended = blend_fluorescence_cars(
        composite_fluor, cars_gray_8bit, alpha=0.5
    )

    fluor_bgr = cv2.cvtColor(composite_fluor, cv2.COLOR_RGB2BGR)
    blend_bgr = cv2.cvtColor(fluor_cars_blended, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(out_dir, f"{file_stub}_fluor.png"), fluor_bgr)
    cv2.imwrite(os.path.join(out_dir, f"{file_stub}_cars.png"), cars_gray_8bit)
    cv2.imwrite(os.path.join(out_dir, f"{file_stub}_fluor_cars.png"), blend_bgr)
    print(f"Saved composites to {out_dir}")
