"""Utilities for loading the pipeline config and resolving marker names."""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import Any, Dict, TYPE_CHECKING

from .constants import LOG_LEVEL
from .path_utils import resolve_all_paths

if TYPE_CHECKING:
    from types import ModuleType

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


def load_config(py_file_path: str) -> Dict[str, Any]:
    """
    Load a Python-based config module and return its `config` dict with resolved paths.

    The module at `py_file_path` must define a top-level variable named `config`
    that is a dictionary. All paths inside the config are normalized/relocated by
    `resolve_all_paths`, optionally using the environment variable `LIPID_ROOT`
    as a temporary project-root override.

    Parameters
    ----------
    py_file_path
        Path to a Python file that defines a `config` dictionary.

    Returns
    -------
    Dict[str, Any]
        The loaded and path-resolved configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ImportError
        If the module cannot be imported.
    AttributeError
        If the module does not define `config`.
    TypeError
        If `config` is not a dict.
    """
    if not os.path.isfile(py_file_path):
        raise FileNotFoundError(f"Config file not found: {py_file_path}")

    spec = importlib.util.spec_from_file_location("cfg_module", py_file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for: {py_file_path}")

    mod: ModuleType = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            f"Failed to import config module '{py_file_path}': {exc}"
        ) from exc

    if not hasattr(mod, "config"):
        raise AttributeError(f"'config' not found in module: {py_file_path}")

    cfg = getattr(mod, "config")
    if not isinstance(cfg, dict):
        raise TypeError(
            f"'config' must be a dict, got {type(cfg).__name__} in {py_file_path}"
        )

    # Allow a temporary root override via env (used if CLI doesn't pass one)
    root_override = os.environ.get("LIPID_ROOT", None)
    if root_override:
        logger.info("Applying LIPID_ROOT override: %s", root_override)

    # Normalize/relocate all paths in the config
    cfg = resolve_all_paths(cfg, root_override)

    logger.info("Loaded configuration from %s", py_file_path)
    return cfg


def resolve_marker_name(name: str, config: Dict[str, Any]) -> str:
    """
    Map a logical marker name (e.g., 'TUJ') to an actual key in `channel_map`
    (e.g., 'TUJ_Ck'). Resolution priority:

      1) direct exact key match with non-None channel index
      2) explicit alias mapping in `config['marker_aliases']`
      3) case-insensitive exact match
      4) case-insensitive substring match

    Parameters
    ----------
    name
        Logical marker name to resolve (case-insensitive).
    config
        Configuration dictionary that contains:
          - channel_map: Dict[str, Optional[int]]
          - marker_aliases: Optional[Dict[str, str]]

    Returns
    -------
    str
        The resolved key present in `channel_map`.

    Raises
    ------
    KeyError
        If no suitable mapping is found.
    """
    key = (name or "").strip()
    cm: Dict[str, Any] = config["channel_map"]  # type: ignore[assignment]

    # 1) Direct exact hit with a usable (non-None) channel index
    if key in cm and cm[key] is not None:
        return key

    # 2) Explicit alias mapping
    alias_map: Dict[str, str] = config.get("marker_aliases", {})  # type: ignore[assignment]
    alias = alias_map.get(key)
    if alias and alias in cm and cm[alias] is not None:
        return alias

    # 3) Case-insensitive exact match
    lower_key = key.lower()
    for k, v in cm.items():
        if k.lower() == lower_key and v is not None:
            return k

    # 4) Substring match (e.g., 'TUJ' in 'TUJ_Ck'), prefer the shortest match
    candidates = [k for k, v in cm.items() if lower_key in k.lower() and v is not None]
    if candidates:
        # Choose the shortest key to avoid overly specific variants trumping base names
        return sorted(candidates, key=len)[0]

    raise KeyError(
        f"Marker '{name}' not found in channel_map. "
        "Add it to channel_map or add an alias in config['marker_aliases']."
    )
