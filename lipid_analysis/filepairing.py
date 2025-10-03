# lipid_analysis/filepairing.py
import os
import re
from typing import Dict


def parse_nd2_filename(filename, config):
    """Parse ND2 filename into prefix/markers/CARS/mag/Stacks label/number (unchanged semantics)."""
    base_no_ext = filename.replace(".nd2", "")
    cars_keyword = config["file_keywords"]["cars_keyword"]
    contains_cars = cars_keyword in base_no_ext
    mag_keyword = config["file_keywords"]["magnification_keyword"]
    magnification = mag_keyword if (mag_keyword in base_no_ext) else None

    found_markers = set()
    for mk in config["file_keywords"]["fluorescence_markers"]:
        if mk in base_no_ext:
            found_markers.add(mk)

    match = re.search(r"(Stacks([A-Za-z]*)(\d*)$)", base_no_ext)
    stacks_label = stacks_number = None
    if match:
        label_part, digit_part = match.group(2), match.group(3)
        stacks_label = label_part
        if digit_part:
            stacks_number = int(digit_part)

    prefix_candidate = base_no_ext
    if match:
        prefix_candidate = prefix_candidate[: match.start(1)]
    if magnification is not None:
        prefix_candidate = prefix_candidate.replace(mag_keyword, "")
    if contains_cars:
        prefix_candidate = prefix_candidate.replace(cars_keyword, "")
    for mk in found_markers:
        prefix_candidate = prefix_candidate.replace(mk, "")
    prefix_candidate = re.sub(r"[-_]+$", "", prefix_candidate).strip()

    return {
        "base_no_ext": base_no_ext,
        "prefix": prefix_candidate,
        "markers_found": found_markers,
        "contains_cars": contains_cars,
        "magnification": magnification,
        "stacks_label": stacks_label,
        "stacks_number": stacks_number,
    }


def get_file_key(filename, config):
    """Construct a pairing key from filename (unchanged semantics)."""
    base = filename.replace(".nd2", "")
    is_cars = config["file_keywords"]["cars_keyword"] in base
    match = re.search(r"(Stacks[A-Za-z]*\d*)$", base)
    if not match:
        raise ValueError(f"No valid 'Stacks...' found in filename: {filename}")

    stacks_part = match.group(1)
    prefix_candidate = base[: match.start(1)]
    removal_candidates = config["file_keywords"]["fluorescence_markers"] + [
        config["file_keywords"]["cars_keyword"],
        config["file_keywords"]["magnification_keyword"],
    ]
    for kw in removal_candidates:
        prefix_candidate = prefix_candidate.replace(kw, "")
    prefix_candidate = re.sub(r"-+", "-", prefix_candidate).strip("-")

    match_stacks = re.search(r"Stacks([A-Za-z]*)(\d*)", stacks_part)
    if not match_stacks:
        raise ValueError(f"Could not parse the 'Stacks' suffix: {stacks_part}")

    label_part, digit_part = match_stacks.group(1), match_stacks.group(2)

    if digit_part:
        stack_num = int(digit_part)
        if not is_cars:
            offset_total = 0
            for marker in config["file_keywords"]["fluorescence_markers"]:
                if marker in base:
                    offset_total += config["stack_offset"].get(marker, 0)
            stack_num += offset_total
        final_stacks_part = f"Stacks{label_part}{stack_num}"
    else:
        final_stacks_part = f"Stacks{label_part}"

    return (
        f"{prefix_candidate}-{final_stacks_part}"
        if prefix_candidate
        else final_stacks_part
    )


def match_fluoro_and_cars(fluoro_list, cars_list, config):
    """Pair fluorescence and CARS files by prefix/label/stack number (+ optional marker overlap)."""
    paired_files: Dict[str, Dict[str, str]] = {}
    stack_offset_dict = config.get("stack_offset", {})

    for f_meta, f_path in fluoro_list:
        offset_total = sum(
            stack_offset_dict.get(mk, 0) for mk in f_meta["markers_found"]
        )
        f_stacks_num = (
            f_meta["stacks_number"] + offset_total
            if f_meta["stacks_number"] is not None
            else None
        )

        for c_meta, c_path in cars_list:
            if f_meta["prefix"] != c_meta["prefix"]:
                continue
            if f_meta["stacks_label"] != c_meta["stacks_label"]:
                continue
            if f_stacks_num != c_meta["stacks_number"]:
                continue

            c_markers = c_meta["markers_found"]
            if c_markers and not c_markers.issubset(f_meta["markers_found"]):
                continue

            c_markers_sorted = "-".join(sorted(list(c_markers))) or "NoMarkers"
            key = f"{f_meta['prefix']}-Stacks{f_meta['stacks_label']}{f_stacks_num or ''}-{c_markers_sorted}"
            paired_files[key] = {"fluorescence": f_path, "CARS": c_path}
            break  # assume 1:1

    return paired_files


def find_nd2_files(directory, config):
    """Scan a directory for ND2s, returning (paired_files, hyperspectral_folders)."""
    hyperspectral_folders, cars_list, fluorescence_list = [], [], []
    file_kw = config["file_keywords"]
    hyperspec_kw = file_kw["hyperspectral_keyword"].lower()

    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        name_l = item.lower()

        if os.path.isdir(full_path) and hyperspec_kw in name_l:
            hyperspectral_folders.append(full_path)
            continue

        if not (
            name_l.endswith(".nd2")
            and file_kw["magnification_keyword"].lower() in name_l
        ):
            continue

        meta = parse_nd2_filename(item, config)
        print(f"File: {item} => meta: {meta}")
        if meta["contains_cars"]:
            cars_list.append((meta, full_path))
        else:
            fluorescence_list.append((meta, full_path))

    paired_files = match_fluoro_and_cars(fluorescence_list, cars_list, config)
    return paired_files, hyperspectral_folders
