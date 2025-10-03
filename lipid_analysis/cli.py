import argparse
import os
import sys

from . import constants
from .analysis import process_nd2_pair
from .config_utils import load_config
from .filepairing import find_nd2_files
from .hyperspec import process_hyperspectral_series
from .io_utils import save_results_to_excel
from .reference import generate_reference_image
from .runtime import capture_logs_on_failure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="Path to a .py file containing `config` dict. If not provided and --debug_single is set, will run the debug config.",
    )
    parser.add_argument(
        "--debug_single",
        action="store_true",
        help="Run a hardcoded single config file for debugging (ignores --config).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show all print/figure output during each batch run instead of capturing.",
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

    # Make VERBOSE dynamic
    constants.VERBOSE = bool(args.verbose)

    debug_config_path = r"D:\OneDrive - Stanford\Research Documents\Python Scripts\config_files\config_AD3d.py"
    if args.debug_single:
        cfg_path = debug_config_path
        print(f"[DEBUG MODE] Running lipid_analysis with config: {cfg_path}")
    elif args.config:
        cfg_path = args.config
    else:
        parser.error("Either --config must be provided or --debug_single must be set.")

    config = load_config(cfg_path)

    # if --root provided, run path normalization again using the CLI value
    if args.root:
        from .path_utils import resolve_all_paths

        config = resolve_all_paths(config, args.root)

    # Optional: make verbose imply alignment debug
    if getattr(args, "verbose", False):
        config.setdefault("debug_alignment", True)
        config.setdefault("debug_output_dir", "Debug")
        config.setdefault("debug_alignment_show_plots", False)

    # Inject global config into all modules that reference a module-level `config`
    import lipid_analysis
    import lipid_analysis.analysis as _ana
    import lipid_analysis.hyperspec as _hs

    lipid_analysis.config = config
    _ana.config = config
    _hs.config = config

    # Let user config imports work (if they reference sibling files)
    sys.path.insert(0, os.path.dirname(cfg_path))

    DIRECTORY = config["paths"]["data_directory"]
    reference_file = os.path.join(DIRECTORY, "Reference.nd2")
    reference_output_path = os.path.join(DIRECTORY, "Reference.tif")
    output_file = os.path.join(DIRECTORY, "analysis_results.xlsx")

    if not os.path.isfile(reference_file):
        raise FileNotFoundError(f"Reference ND2 not found: {reference_file}")

    # 1) Reference
    reference_image = generate_reference_image(
        reference_file, reference_output_path, blur_radius_microns=2
    )

    # 2) Paired ND2 processing
    all_results_list, all_summary_list = [], []
    paired_files, hyperspectral_folders = find_nd2_files(DIRECTORY, config)
    for key_val, paths_dict in paired_files.items():
        label = f"paired run: {key_val}"
        with capture_logs_on_failure(label, enabled=(not constants.VERBOSE)):
            print(f"Processing pair: {key_val}")
            pair_res, pair_sum = process_nd2_pair(
                paths_dict["fluorescence"], paths_dict["CARS"], reference_image
            )
            all_results_list.extend(pair_res)
            all_summary_list.extend(pair_sum)

    # 3) Hyperspectral
    hyperspectral_foci_params = config["morphology_params"]["foci_params_hyperspectral"]
    for folder in hyperspectral_folders:
        folder_name = os.path.basename(folder)
        label = f"hyperspectral run: {folder_name}"
        with capture_logs_on_failure(label, enabled=(not constants.VERBOSE)):
            hyperspectral_output = os.path.join(
                DIRECTORY, f"Hyperspectral_Results_{folder_name}.xlsx"
            )
            process_hyperspectral_series(
                folder, reference_image, hyperspectral_output, hyperspectral_foci_params
            )

    # 3b) Post-classify hyperspectral outputs (CH-stretch rules only)
    if not args.no_classify:
        from .postclassify import classify_hyperspectral_dir

        try:
            classify_hyperspectral_dir(
                DIRECTORY,
                rules_json=args.classify_rules,
                write_back=True,
                consolidate=True,
            )
        except Exception as e:
            print(f"[WARN] Post-classification failed: {e}")

    # 4) Excel outputs
    save_results_to_excel(all_results_list, all_summary_list, output_file)
    print(f"Results saved to {output_file}")
