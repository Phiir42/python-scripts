# lipid_analysis/config_utils.py
from __future__ import annotations  # <-- add this line

import importlib.util
import os
from typing import Any, Dict, TYPE_CHECKING

from .path_utils import resolve_all_paths

if TYPE_CHECKING:
    from types import ModuleType

def load_config(py_file_path: str) -> Dict[str, Any]:
    """
    Dynamically load a Python config file as a module and return its `config` dict.

    Raises
    ------
    FileNotFoundError : if the file does not exist
    ImportError       : if the module cannot be loaded
    AttributeError    : if the module does not define `config`
    TypeError         : if `config` is not a dict
    """
    if not os.path.isfile(py_file_path):
        raise FileNotFoundError(f"Config file not found: {py_file_path}")

    spec = importlib.util.spec_from_file_location("cfg_module", py_file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for: {py_file_path}")

    mod: ModuleType = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception as e:
        raise ImportError(
            f"Failed to import config module '{py_file_path}': {e}"
        ) from e

    if not hasattr(mod, "config"):
        raise AttributeError(f"'config' not found in module: {py_file_path}")

    cfg = getattr(mod, "config")
    if not isinstance(cfg, dict):
        raise TypeError(
            f"'config' must be a dict, got {type(cfg).__name__} in {py_file_path}"
        )

    # allow a temporary root override via env (used if CLI doesn't pass one)
    root_override = os.environ.get("LIPID_ROOT", None)

    # normalize/relocate all paths in the config
    cfg = resolve_all_paths(cfg, root_override)

    return cfg


def resolve_marker_name(name: str, config: Dict[str, Any]) -> str:
    """
    Map a logical marker name (e.g., 'TUJ') to an actual key in channel_map
    (e.g., 'TUJ_Ck'). Uses aliases, then case-insensitive and substring matches.

    Raises
    ------
    KeyError : if no suitable mapping is found
    """
    key = (name or "").strip()
    cm = config["channel_map"]

    # direct hit
    if key in cm and cm[key] is not None:
        return key

    # explicit alias mapping
    alias = config.get("marker_aliases", {}).get(key)
    if alias and alias in cm:
        return alias

    # case-insensitive exact match
    for k in cm:
        if k.lower() == key.lower():
            return k

    # substring match (e.g., 'TUJ' in 'TUJ_Ck')
    low_key = key.lower()
    for k in cm:
        if low_key in k.lower():
            return k

    raise KeyError(
        f"Marker '{name}' not found in channel_map. "
        "Add it to channel_map or add an alias in config['marker_aliases']."
    )
