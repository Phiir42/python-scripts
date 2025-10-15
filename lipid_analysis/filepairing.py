"""ND2 filename parsing and fluorescence↔CARS file pairing."""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from .constants import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

# Types
ND2Meta = MutableMapping[str, object]
ParsedList = List[Tuple[ND2Meta, str]]  # (metadata, full_path)


# ----------------------------- helpers -------------------------------------


def _strip_ext_case_insensitive(filename: str) -> str:
    """Return filename without its final extension (case-insensitive)."""
    base, ext = os.path.splitext(filename)
    return base if ext.lower() == ".nd2" else filename


def _ci_contains(haystack: str, needle: str) -> bool:
    """Case-insensitive substring test."""
    return needle.lower() in haystack.lower()


def _ci_remove_all(text: str, tokens: Iterable[str]) -> str:
    """Remove all occurrences of tokens from text, case-insensitively."""
    out = text
    for tok in tokens:
        if not tok:
            continue
        pattern = re.compile(re.escape(tok), flags=re.IGNORECASE)
        out = pattern.sub("", out)
    return out


def _compact_separators(s: str) -> str:
    """
    Normalize and trim separators:
    - Collapse multiple dashes/underscores to a single dash.
    - Strip leading/trailing dashes/underscores/whitespace.
    """
    s = re.sub(r"[-_]+", "-", s)
    return s.strip("-_ ").strip()


def _parse_stacks_suffix(base_no_ext: str) -> Tuple[str, Optional[int]]:
    """
    Extract the trailing 'Stacks<Label><Digits>' components from the end of the string.

    Returns
    -------
    label : str
        The alphabetic label (possibly empty string if absent).
    number : Optional[int]
        The numeric suffix if present, else None.
    """
    m = re.search(r"(Stacks([A-Za-z]*)(\d*)$)", base_no_ext)
    if not m:
        return "", None
    label = m.group(2) or ""
    digits = m.group(3) or ""
    return label, (int(digits) if digits else None)


# ----------------------------- core API ------------------------------------


def parse_nd2_filename(filename: str, config: Mapping[str, object]) -> ND2Meta:
    """
    Parse an ND2 filename into structured metadata.

    The parser looks for:
    - presence of the CARS keyword,
    - a magnification keyword (treated as a tag, not a value),
    - a set of fluorescence markers contained in the name,
    - a trailing 'Stacks<Label><Digits>' suffix,
    - a cleaned 'prefix' with those tokens removed for pairing.

    Parameters
    ----------
    filename
        ND2 filename (not necessarily a full path).
    config
        Configuration mapping with keys:
            file_keywords: {
                "cars_keyword": str,
                "magnification_keyword": str,
                "hyperspectral_keyword": str,      # used elsewhere
                "fluorescence_markers": List[str],
            }

    Returns
    -------
    dict
        {
          "base_no_ext": str,
          "prefix": str,
          "markers_found": Set[str],
          "contains_cars": bool,
          "magnification": Optional[str],    # the magnification keyword if present
          "stacks_label": str,               # '' if absent
          "stacks_number": Optional[int],
        }
    """
    file_kw = config["file_keywords"]  # type: ignore[index]
    cars_kw: str = file_kw["cars_keyword"]  # type: ignore[index]
    mag_kw: str = file_kw["magnification_keyword"]  # type: ignore[index]
    markers: Sequence[str] = file_kw["fluorescence_markers"]  # type: ignore[index]

    base_no_ext = _strip_ext_case_insensitive(filename)
    base_lower = base_no_ext.lower()

    contains_cars = _ci_contains(base_lower, cars_kw)
    magnification = mag_kw if _ci_contains(base_lower, mag_kw) else None

    # Collect markers present (store canonical strings as given in config)
    found_markers: Set[str] = set()
    for mk in markers:
        if _ci_contains(base_lower, mk):
            found_markers.add(mk)

    # Parse trailing Stacks label/number
    stacks_label, stacks_number = _parse_stacks_suffix(base_no_ext)

    # Build prefix by removing: Stacks suffix (only the trailing piece), mag, cars, and markers
    prefix_candidate = base_no_ext
    m = re.search(r"(Stacks[A-Za-z]*\d*)$", prefix_candidate)
    if m:
        prefix_candidate = prefix_candidate[: m.start(1)]

    removal_tokens = list(found_markers) + [mag_kw, cars_kw]
    prefix_candidate = _ci_remove_all(prefix_candidate, removal_tokens)
    prefix_candidate = _compact_separators(prefix_candidate)

    return {
        "base_no_ext": base_no_ext,
        "prefix": prefix_candidate,
        "markers_found": found_markers,
        "contains_cars": contains_cars,
        "magnification": magnification,
        "stacks_label": stacks_label,       # '' if absent
        "stacks_number": stacks_number,     # int or None
    }


def get_file_key(filename: str, config: Mapping[str, object]) -> str:
    """
    Construct a pairing key from a filename.

    The key is "<prefix>-Stacks<Label><Number>" or "Stacks<Label><Number>" if the
    prefix is empty. The prefix is built by removing all configured tokens from
    the base name (magnification, CARS, fluorescence markers) and normalizing
    separators. Stack numbers for non-CARS files will later be offset in pairing.

    Parameters
    ----------
    filename
        ND2 filename (not necessarily a full path).
    config
        Same structure as in `parse_nd2_filename`.

    Returns
    -------
    str
        A normalized key suitable for matching across fluorescence and CARS files.

    Raises
    ------
    ValueError
        If no 'Stacks...' suffix can be parsed from the filename.
    """
    file_kw = config["file_keywords"]  # type: ignore[index]
    cars_kw: str = file_kw["cars_keyword"]  # type: ignore[index]
    mag_kw: str = file_kw["magnification_keyword"]  # type: ignore[index]
    markers: Sequence[str] = file_kw["fluorescence_markers"]  # type: ignore[index]

    base = _strip_ext_case_insensitive(filename)
    base_lower = base.lower()

    is_cars = _ci_contains(base_lower, cars_kw)

    match = re.search(r"(Stacks[A-Za-z]*\d*)$", base)
    if not match:
        raise ValueError(f"No valid 'Stacks...' suffix found in filename: {filename}")
    stacks_part = match.group(1)

    prefix_candidate = base[: match.start(1)]
    # Remove tokens regardless of case, then normalize separators
    removal_candidates = list(markers) + [cars_kw, mag_kw]
    prefix_candidate = _ci_remove_all(prefix_candidate, removal_candidates)
    prefix_candidate = _compact_separators(prefix_candidate)

    match_stacks = re.search(r"Stacks([A-Za-z]*)(\d*)$", stacks_part)
    if not match_stacks:
        raise ValueError(f"Could not parse the 'Stacks' suffix: {stacks_part}")

    label_part = match_stacks.group(1) or ""
    digit_part = match_stacks.group(2) or ""

    if digit_part:
        stack_num = int(digit_part)
        if not is_cars:
            # Offsets are applied later in pairing too; here we only normalize the key’s stacks-part.
            # We keep the original number here for the key; pairing uses metadata numbers + offsets.
            pass
        final_stacks_part = f"Stacks{label_part}{stack_num}"
    else:
        final_stacks_part = f"Stacks{label_part}"

    return f"{prefix_candidate}-{final_stacks_part}" if prefix_candidate else final_stacks_part


def match_fluoro_and_cars(
    fluoro_list: ParsedList,
    cars_list: ParsedList,
    config: Mapping[str, object],
) -> Dict[str, Dict[str, str]]:
    """
    Pair fluorescence (.nd2) files with CARS (.nd2) files by prefix/label/stack number.

    Matching rules
    --------------
    - Prefix equality (after token removal/normalization).
    - Stacks label equality (empty string allowed).
    - Stacks number equality, after adding marker-specific offsets to the
      fluorescence file's stack number (if present).
    - If the CARS filename lists any markers, require that they are a subset
      of the fluorescence markers.

    The first match per fluorescence file is used (assumed 1:1 pairing).

    Parameters
    ----------
    fluoro_list
        List of (metadata, full_path) for fluorescence files.
    cars_list
        List of (metadata, full_path) for CARS files.
    config
        Same structure as in `parse_nd2_filename`.

    Returns
    -------
    Dict[str, Dict[str, str]]
        Mapping from a composite key to {"fluorescence": path, "CARS": path}.
        The key is: "<prefix>-Stacks<Label><Number>-<CARSMarkers|NoMarkers>".
    """
    paired_files: Dict[str, Dict[str, str]] = {}
    stack_offset_dict: Mapping[str, int] = config.get("stack_offset", {})  # type: ignore[assignment]

    for f_meta, f_path in fluoro_list:
        # Compute total offset from all markers found in the fluorescence filename
        f_markers = f_meta["markers_found"]  # type: ignore[assignment]
        offset_total = sum(stack_offset_dict.get(mk, 0) for mk in f_markers)  # type: ignore[arg-type]
        f_stacks_num_raw = f_meta["stacks_number"]  # type: ignore[assignment]
        f_stacks_num = (f_stacks_num_raw + offset_total) if f_stacks_num_raw is not None else None

        for c_meta, c_path in cars_list:
            if f_meta["prefix"] != c_meta["prefix"]:
                continue
            if f_meta["stacks_label"] != c_meta["stacks_label"]:
                continue
            if f_stacks_num != c_meta["stacks_number"]:
                continue

            c_markers: Set[str] = c_meta["markers_found"]  # type: ignore[assignment]
            if c_markers and not c_markers.issubset(f_markers):
                continue

            c_markers_sorted = "-".join(sorted(c_markers)) if c_markers else "NoMarkers"
            label_str: str = f_meta["stacks_label"]  # '' allowed
            num_str = "" if f_stacks_num is None else str(f_stacks_num)
            key = f"{f_meta['prefix']}-Stacks{label_str}{num_str}-{c_markers_sorted}" if f_meta["prefix"] else f"Stacks{label_str}{num_str}-{c_markers_sorted}"

            paired_files[key] = {"fluorescence": f_path, "CARS": c_path}
            break  # assume 1:1 pairing; stop at the first match

    return paired_files


def find_nd2_files(
    directory: str, config: Mapping[str, object]
) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """
    Scan a directory for ND2 files and hyperspectral folders, and produce pairs.

    Selection
    ---------
    - Hyperspectral folders are any subdirectories whose names contain the
      configured `hyperspectral_keyword` (case-insensitive).
    - Candidate ND2 files must:
        * have extension .nd2 (any casing), and
        * contain the `magnification_keyword` in the filename (case-insensitive).

    Parameters
    ----------
    directory
        Directory to scan (non-recursive).
    config
        Configuration mapping with keys:
            file_keywords: {
                "cars_keyword": str,
                "magnification_keyword": str,
                "hyperspectral_keyword": str,
                "fluorescence_markers": List[str],
            }
            stack_offset: {marker: int, ...}  # optional

    Returns
    -------
    paired_files : Dict[str, Dict[str, str]]
        Pairing results (see `match_fluoro_and_cars`).
    hyperspectral_folders : List[str]
        Full paths of folders detected as hyperspectral.
    """
    hyperspectral_folders: List[str] = []
    cars_list: ParsedList = []
    fluorescence_list: ParsedList = []

    file_kw = config["file_keywords"]  # type: ignore[index]
    hyperspec_kw: str = file_kw["hyperspectral_keyword"]  # type: ignore[index]
    mag_kw: str = file_kw["magnification_keyword"]  # type: ignore[index]

    try:
        entries = os.listdir(directory)
    except FileNotFoundError as exc:  # noqa: BLE001
        raise FileNotFoundError(f"Directory not found: {directory}") from exc

    for item in entries:
        full_path = os.path.join(directory, item)
        name_lower = item.lower()

        if os.path.isdir(full_path) and _ci_contains(name_lower, hyperspec_kw):
            hyperspectral_folders.append(full_path)
            continue

        # Keep .nd2 only, must contain magnification keyword (case-insensitive)
        base, ext = os.path.splitext(item)
        if ext.lower() != ".nd2" or not _ci_contains(name_lower, mag_kw):
            continue

        meta = parse_nd2_filename(item, config)
        logger.info("Found ND2: %s -> %s", item, meta)

        if meta["contains_cars"]:
            cars_list.append((meta, full_path))
        else:
            fluorescence_list.append((meta, full_path))

    paired_files = match_fluoro_and_cars(fluorescence_list, cars_list, config)
    return paired_files, hyperspectral_folders
