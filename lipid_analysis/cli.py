# lipid_analysis/cli.py
"""
Command-line entry point for the lipid_analysis pipeline.

Responsibilities
---------------
1) Parse CLI arguments.
2) Load and normalize the config.
3) Set VERBOSE-driven logging across the repo.
4) Generate the reference image.
5) Process paired ND2 files (fluorescence + CARS).
6) Process hyperspectral series (including optional myelin average fits).
7) Post-classify hyperspectral outputs (optional).
8) Save consolidated Excel outputs and report χ² summary.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

from . import constants
from .analysis import process_nd2_pair
from .config_utils import load_config
from .filepairing import find_nd2_files
from .hyperspec import compute_myelin_average_for_series, process_hyperspectral_series
from .io_utils import save_results_to_excel
from .peakfit import chi2_reset, finish_debug_capture, report_chi2_summary
from .reference import generate_reference_image
from .runtime import capture_logs_on_failure

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


def _sync_logging_with_verbose(verbose: bool) -> None:
    """
    Synchronize repository logging with the VERBOSE flag:
    - Update constants.VERBOSE and constants.LOG_LEVEL
    - Set root logger level
    - Keep 'nd2reader' quiet
    """
    constants.VERBOSE = bool(verbose)
    level = logging.DEBUG if constants.VERBOSE else logging.WARNING
    constants.LOG_LEVEL = level

    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    else:
        root.setLevel(level)
        for h in root.handlers:
            try:
                h.setLevel(level)
            except Exception:
                pass

    logging.getLogger("nd2reader").setLevel(logging.ERROR)


def _resolve_config_path(args: argparse.Namespace) -> str:
    """Choose config path from CLI flags, enforcing required options."""
    debug_config_path = (
        r"D:\OneDrive - Stanford\Research Documents\Python Scripts\config_files\config_AD3d.py"
    )
    if args.debug_single:
        logger.info("[DEBUG MODE] Using debug config: %s", debug_config_path)
        return debug_config_path
    if args.config:
        return args.config
    raise SystemExit("Either --config must be provided or --debug_single must be set.")


def _inject_global_config(config: dict) -> None:
    """
    Inject a shared config object into modules that expect a module-level `config`
    to be set by the runner (maintains original design).
    """
    import lipid_analysis
    import lipid_analysis.analysis as _ana
    import lipid_analysis.hyperspec as _hs

    lipid_analysis.config = config
    _ana.config = config
    _hs.config = config


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(prog="lipid-analysis")
    parser.add_argument(
        "--config",
        help=(
            "Path to a .py file containing `config` dict. "
            "If not provided and --debug_single is set, uses the debug config."
        ),
    )
    parser.add_argument(
        "--debug_single",
        action="store_true",
        help="Run a hardcoded single config file for debugging (ignores --config).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show debug logs and interactive figures; otherwise capture/minimize logs.",
    )
    parser.add_argument(
        "--no_classify",
        action="store_true",
        help="Skip rule-based classification of hyperspectral results.",
    )
    parser.add_argument(
        "--classify_rules",
        default=None,
        help="Optional JSON to override classification thresholds.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Override root path for all absolute paths in config.",
    )
    args = parser.parse_args()

    # Align logging with --verbose
    _sync_logging_with_verbose(args.verbose)

    cfg_path = _resolve_config_path(args)
    config = load_config(cfg_path)

    # If --root provided, re-run path normalization with CLI override
    if args.root:
        from .path_utils import resolve_all_paths
        config = resolve_all_paths(config, args.root)

    # Optional: verbose implies alignment debug
    if args.verbose:
        config.setdefault("debug_alignment", True)
        config.setdefault("debug_output_dir", "Debug")
        config.setdefault("debug_alignment_show_plots", False)

    _inject_global_config(config)

    # Let user config do relative imports (if it references siblings)
    sys.path.insert(0, os.path.dirname(cfg_path))

    # --- Key paths
    data_dir: str = config["paths"]["data_directory"]
    reference_file = os.path.join(data_dir, "Reference.nd2")
    reference_output_path = os.path.join(data_dir, "Reference.tif")
    output_file = os.path.join(data_dir, "analysis_results.xlsx")

    if not os.path.isfile(reference_file):
        raise FileNotFoundError(f"Reference ND2 not found: {reference_file}")

    # 1) Reference
    logger.info("Generating reference image from %s", reference_file)
    reference_image = generate_reference_image(
        reference_file, reference_output_path, blur_radius_microns=2
    )
    logger.info("Saved reference image to %s", reference_output_path)

    # 2) Paired ND2 processing
    all_results, all_summary = [], []
    paired_files, hyperspectral_folders = find_nd2_files(data_dir, config)
    for key_val, paths_dict in paired_files.items():
        label = f"paired run: {key_val}"
        with capture_logs_on_failure(label, enabled=(not constants.VERBOSE)):
            logger.info("Processing fluorescence/CARS pair: %s", key_val)
            pair_res, pair_sum = process_nd2_pair(
                paths_dict["fluorescence"], paths_dict["CARS"], reference_image, config
            )
            all_results.extend(pair_res)
            all_summary.extend(pair_sum)

    # 3) Hyperspectral processing
    hyperspec_params = config["morphology_params"]["foci_params_hyperspectral"]
    chi2_reset()
    myelin_rows = []

    for folder in hyperspectral_folders:
        folder_name = os.path.basename(folder)
        label = f"hyperspectral run: {folder_name}"
        with capture_logs_on_failure(label, enabled=(not constants.VERBOSE)):
            out_xlsx = os.path.join(data_dir, f"Hyperspectral_Results_{folder_name}.xlsx")
            logger.info("Processing hyperspectral series: %s", folder_name)
            process_hyperspectral_series(folder, reference_image, out_xlsx, hyperspec_params)

            # NEW: compute myelin-minus-droplets average spectrum & fit it
            myelin_row = compute_myelin_average_for_series(
                folder,
                reference_image,
                hyperspec_params,
                config.get("myelin_params", {}),
            )
            if myelin_row:
                myelin_row["Folder"] = folder_name
                myelin_rows.append(myelin_row)

            # Close the peak-fit debug capture AFTER myelin
            try:
                if constants.PEAKFIT_DEBUG:
                    finish_debug_capture(make_pptx=True)
            except Exception as exc:
                if constants.VERBOSE:
                    logger.warning("[PeakFit DEBUG] finish_debug_capture (CLI) failed: %s", exc)

    report_chi2_summary()

    if myelin_rows:
        out_xlsx = os.path.join(data_dir, "Hyperspectral_Myelin_AverageFits.xlsx")
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            pd.DataFrame(myelin_rows).to_excel(
                writer, sheet_name="Myelin_Average_Fits", index=False
            )
        logger.info("[myelin] Wrote per-series myelin average fits → %s", out_xlsx)

    # 3b) Post-classification
    if not args.no_classify:
        from .postclassify import classify_hyperspectral_dir

        try:
            classify_hyperspectral_dir(
                data_dir,
                rules_json=args.classify_rules,
                write_back=True,
                consolidate=True,
            )
        except Exception as exc:
            logger.warning("[WARN] Post-classification failed: %s", exc)

    # 4) Excel outputs
    save_results_to_excel(all_results, all_summary, output_file)
    logger.info("Results saved to %s", output_file)


if __name__ == "__main__":
    main()
