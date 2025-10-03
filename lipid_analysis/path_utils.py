# lipid_analysis/path_utils.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

KNOWN_PREFIXES = [
    r"D:\OneDrive - Stanford",
    r"C:\Users\clchr\OneDrive - Stanford",
]


def _swap_prefix(s: str, prefixes: Iterable[str], new_root: str) -> str:
    # Replace any known absolute prefix with new_root, preserving the relative tail.
    for pref in prefixes:
        if s.startswith(pref):
            tail = s[len(pref) :].lstrip("\\/")  # keep relative part
            return str(Path(new_root) / tail)
    return s


def _expand_placeholders(s: str, new_root: str | None) -> str:
    # Support {ONEDRIVE} placeholder and ~ expansion
    s2 = s.replace("{ONEDRIVE}", os.environ.get("ONEDRIVE", new_root or ""))
    return str(Path(s2).expanduser())


def resolve_all_paths(obj: Any, new_root: str | None = None) -> Any:
    """
    Recursively walk dict/list structures and:
      - swap known absolute OneDrive prefixes to `new_root` (if provided)
      - expand {ONEDRIVE} placeholder and ~
    """
    if isinstance(obj, dict):
        return {k: resolve_all_paths(v, new_root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_all_paths(v, new_root) for v in obj]
    if isinstance(obj, str):
        s = _expand_placeholders(obj, new_root)
        return _swap_prefix(s, KNOWN_PREFIXES, new_root) if new_root else s
    return obj
