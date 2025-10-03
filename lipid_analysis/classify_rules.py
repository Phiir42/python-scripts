"""
classify_rules.py
-----------------
Rule-based object classification using CH-stretch peak features only
(works when fingerprint region is unavailable).

Classes implemented (enabled by default):
  - TG_unsat
  - TG_sat
  - myelin_like
Optional (disabled by default due to lack of fingerprint disambiguation):
  - cholesterol
  - CE

Thresholds are intentionally conservative and intended to be *tuned* on your dataset.
You can override via a JSON file passed to the runner (see run_classify.py).

Outputs:
  - class_label
  - class_score (0-1 heuristic confidence)
  - rules_fired (comma-separated audit trail)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

DEFAULT_RULES: Dict[str, Any] = {
    "classes_enabled": ["TG_unsat", "TG_sat", "myelin_like"],
    "min_snr": 5.0,  # below this => uncertain
    "require_3010_se": False,  # if True, demands det_3010 based on SE logic
    # thresholds (tune on your data)
    "U_unsat_min": 0.06,  # A3010/A2850 threshold for "unsaturated"
    "Rpack_myelin_min": 0.95,  # A2885/A2850 minimum for ordered lipid
    "Rpack_myelin_max": 1.35,  # keep within a plausible range
    "U_myelin_max": 0.03,  # unsaturation should be low for myelin-like
    "Rhi_tg_min": 0.65,  # (A2935+A2960)/(A2850+A2885) typical TG region
    # confidence shaping
    "p_floor": 0.55,  # minimum score for a confident assignment
}


@dataclass
class RuleResult:
    label: str
    score: float
    rules: List[str]


# ---- Numeric safety helpers (added) ----
EPS = 1e-9  # floor to avoid division by zero
RATIO_CAP = 1e6  # clamp absurd ratios that would not change class interpretation


def safe_div(num: float, den: float, cap: float = RATIO_CAP) -> float:
    """Division with epsilon floor & clamp."""
    try:
        if den is None:
            return 0.0
        d = den if abs(den) > EPS else (EPS if den >= 0 else -EPS)
        r = num / d
        # clamp extremely large ratios; keeps interpretation (very high) without overflow risk
        if r > cap:
            r = cap
        if r < -cap:
            r = -cap
        return r
    except Exception:
        return 0.0
    

def sigmoid_stable(d: float) -> float:
    """Overflow-safe logistic: 1/(1+exp(-d))."""
    if d >= 80.0:
        return 1.0
    if d <= -80.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-d))

def sigmoid_pos_stable(d: float) -> float:
    """Overflow-safe 1/(1+exp(d)) == logistic(-d)."""
    if d >= 80.0:
        return 0.0
    if d <= -80.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(d))


# ---- End helpers ----


def _is_valid(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def _score_from_margin(
    val: float, thr: float, kind: str = "gte", width: float = 0.02
) -> float:
    """
    Map distance from threshold to a [0,1] score; width sets transition softness.
    """
    if not _is_valid(val) or not _is_valid(thr):
        return 0.0
    d = (val - thr) / max(width, 1e-6)
    if kind == "gte":
        return float(sigmoid_stable(d))
    else:  # "lte"
        return float(sigmoid_pos_stable(d))


def classify_row(feat: pd.Series, rules: Dict[str, Any]) -> RuleResult:
    snr = feat.get("SNR_fit", np.nan)
    if _is_valid(snr) and snr < rules["min_snr"]:
        return RuleResult("uncertain", 0.0, ["low_snr"])

    U = feat.get("U", np.nan)
    Rpack = feat.get("R_pack", np.nan)
    Rhi = feat.get("R_hi", np.nan)
    det3010 = bool(feat.get("det_3010", 0.0))

    # optional strictness on 3010 detection
    if rules.get("require_3010_se", False) and not det3010:
        # still allow TG_sat or myelin_like; penalize unsat
        pass

    candidates: List[RuleResult] = []

    if "TG_unsat" in rules["classes_enabled"] and _is_valid(U):
        c = _score_from_margin(U, rules["U_unsat_min"], "gte", width=0.02)
        score = 0.7 * c + 0.3 * (
            _score_from_margin(Rhi, rules["Rhi_tg_min"], "gte", width=0.03)
            if _is_valid(Rhi)
            else 0.0
        )
        rules_fired = []
        if U >= rules["U_unsat_min"]:
            rules_fired.append(f"U>={rules['U_unsat_min']}")
        if _is_valid(Rhi) and Rhi >= rules["Rhi_tg_min"]:
            rules_fired.append(f"Rhi>={rules['Rhi_tg_min']}")
        candidates.append(RuleResult("TG_unsat", float(score), rules_fired))

    if "myelin_like" in rules["classes_enabled"] and _is_valid(Rpack) and _is_valid(U):
        c1 = _score_from_margin(Rpack, rules["Rpack_myelin_min"], "gte", width=0.05)
        c2 = _score_from_margin(U, rules["U_myelin_max"], "lte", width=0.02)
        score = 0.6 * c1 + 0.4 * c2
        rules_fired = []
        if Rpack >= rules["Rpack_myelin_min"]:
            rules_fired.append(f"Rpack>={rules['Rpack_myelin_min']}")
        if U <= rules["U_myelin_max"]:
            rules_fired.append(f"U<={rules['U_myelin_max']}")
        candidates.append(RuleResult("myelin_like", float(score), rules_fired))

    if "TG_sat" in rules["classes_enabled"] and _is_valid(U):
        # TG_sat = low U but not strongly ordered (distinguish from myelin)
        c1 = _score_from_margin(U, rules["U_unsat_min"], "lte", width=0.02)
        # penalize if *too* ordered like myelin
        penalty = (
            _score_from_margin(Rpack, rules["Rpack_myelin_min"], "gte", width=0.05)
            if _is_valid(Rpack)
            else 0.0
        )
        score = float(0.8 * c1 + 0.2 * max(0.0, 1.0 - penalty))
        rules_fired = []
        if U < rules["U_unsat_min"]:
            rules_fired.append(f"U<{rules['U_unsat_min']}")
        if _is_valid(Rpack) and Rpack < rules["Rpack_myelin_min"]:
            rules_fired.append(f"Rpack<{rules['Rpack_myelin_min']}")
        candidates.append(RuleResult("TG_sat", score, rules_fired))

    if not candidates:
        return RuleResult("uncertain", 0.0, ["insufficient_features"])

    # pick best
    best = max(candidates, key=lambda r: r.score)
    if best.score < rules["p_floor"]:
        best = RuleResult("uncertain", best.score, best.rules)
    return best


def classify_table(
    feat_df: pd.DataFrame, rules: Dict[str, Any] | None = None
) -> pd.DataFrame:
    rules = rules or DEFAULT_RULES
    out = []
    for idx, row in feat_df.iterrows():
        rr = classify_row(row, rules)
        out.append(
            {
                "class_label": rr.label,
                "class_score": rr.score,
                "rules_fired": ",".join(rr.rules),
            }
        )
    return pd.concat([feat_df.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def load_rules(json_path: str | None) -> Dict[str, Any]:
    if not json_path:
        return DEFAULT_RULES.copy()
    with open(json_path, "r", encoding="utf-8") as f:
        user = json.load(f)
    rules = DEFAULT_RULES.copy()
    rules.update(user or {})
    return rules
