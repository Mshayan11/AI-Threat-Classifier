"""
unsw_ml.py

Helper functions to use the UNSW-NB15 RandomForest model trained by
train_unsw_model.py inside the AI Cyber Threat Radar.

- Loads models/unsw_rf_model.joblib lazily (first use).
- Builds a numeric feature vector for a given log entry.
- Returns P(attack) as a float in [0, 1], or None if ML can't be used.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

# Must match MODEL_BUNDLE_PATH in train_unsw_model.py
UNSW_MODEL_BUNDLE_PATH = os.path.join("models", "unsw_rf_model.joblib")


@lru_cache(maxsize=1)
def _load_unsw_model_bundle() -> Optional[dict]:
    
    try:
        bundle = joblib.load(UNSW_MODEL_BUNDLE_PATH)
        if "pipeline" not in bundle or "feature_names" not in bundle:
            print("[UNSW-ML] Invalid model bundle structure, disabling ML path.")
            return None
        print("[UNSW-ML] UNSW model bundle loaded.")
        return bundle
    except FileNotFoundError:
        print(
            f"[UNSW-ML] No model bundle found at {UNSW_MODEL_BUNDLE_PATH}; "
            f"run train_unsw_model.py if you want ML support."
        )
    except Exception as exc:  # defensive
        print(f"[UNSW-ML] Failed to load model bundle: {exc!r}")
    return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_feature_vector_from_log(log: Any, feature_names: List[str]) -> Optional[np.ndarray]:
    
    # Example of optional richer mapping based on typical log fields
    derived: Dict[str, float] = {}

    # Try using length of message/details as a crude proxy for "payload size".
    msg = getattr(log, "raw", None) or getattr(log, "message", None) or getattr(log, "details", None)
    if isinstance(msg, str):
        msg_len = float(len(msg))
    else:
        msg_len = 0.0

    derived.setdefault("sbytes", msg_len)
    derived.setdefault("dbytes", 0.0)
    derived.setdefault("spkts", 1.0)
    derived.setdefault("dpkts", 1.0)
    derived.setdefault("dur", 0.0)
    derived.setdefault("rate", 0.0)
    derived.setdefault("sttl", 64.0)
    derived.setdefault("dttl", 64.0)
    derived.setdefault("sload", msg_len)
    derived.setdefault("dload", 0.0)

    # Build vector in the exact order of feature_names
    vector: List[float] = []
    for name in feature_names:
        # 1) exact attribute on log
        if hasattr(log, name):
            value = getattr(log, name)
            vector.append(_safe_float(value))
            continue

        # 2) derived mapping if present
        if name in derived:
            vector.append(derived[name])
            continue

        # 3) default fallback
        vector.append(0.0)

    return np.array(vector, dtype=float).reshape(1, -1)


def ml_attack_probability_from_log(log: Any) -> Optional[float]:
    

    bundle = _load_unsw_model_bundle()
    if bundle is None:
        return None

    pipe = bundle["pipeline"]
    feature_names: List[str] = bundle["feature_names"]

    try:
        vector = _build_feature_vector_from_log(log, feature_names)
        if vector is None:
            return None
        prob_attack = pipe.predict_proba(vector)[0][1]
        return float(prob_attack)
    except Exception as exc:  # defensive
        print(f"[UNSW-ML] Prediction failed for log {log}: {exc!r}")
        return None
