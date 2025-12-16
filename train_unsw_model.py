
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


UNSW_CSV_PATH = os.path.join("data", "UNSW-NB15.csv")
LABEL_COLUMN = "label"
MODEL_BUNDLE_PATH = os.path.join("models", "unsw_rf_model.joblib")

@dataclass
class DatasetSplits:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]

def load_unsw_dataset() -> DatasetSplits:
    """
    Load the UNSW dataset from CSV, handle slightly broken encodings/lines,
    automatically choose numeric feature columns, and split into train/test.
    """
    if not os.path.exists(UNSW_CSV_PATH):
        raise FileNotFoundError(
            f"UNSW CSV not found at {UNSW_CSV_PATH}.\n"
            f"Put your UNSW_NB15_training-set.csv there or update UNSW_CSV_PATH."
        )

    print(f"[UNSW] Loading dataset from: {UNSW_CSV_PATH}")

    df = pd.read_csv(
    UNSW_CSV_PATH,
    encoding="latin1",
    on_bad_lines="skip",  # skip malformed lines instead of crashing
    engine="python",      # more tolerant than the default C engine
)

    print(f"[UNSW] Loaded {len(df):,} rows.")

    if LABEL_COLUMN not in df.columns:
        raise KeyError(
            f"Label column {LABEL_COLUMN!r} not found in dataset.\n"
            f"Available columns include: {list(df.columns)[:20]} ..."
        )

    # Select numeric feature columns automatically (except the label itself)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if LABEL_COLUMN in numeric_cols:
        numeric_cols.remove(LABEL_COLUMN)

    if len(numeric_cols) == 0:
        raise ValueError(
            "No numeric columns found for features. "
            "Check that your UNSW CSV has numeric traffic columns."
        )

    # To keep things manageable, use at most 30 numeric columns
    feature_names = numeric_cols[:30]

    print(f"[UNSW] Using {len(feature_names)} numeric feature columns:")
    print("       " + ", ".join(feature_names))

    X = df[feature_names].values

    # Build binary labels from LABEL_COLUMN
    raw_labels = df[LABEL_COLUMN]
    y = build_binary_labels(raw_labels)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"[UNSW] Train size: {len(X_train):,} rows")
    print(f"[UNSW] Test  size: {len(X_test):,} rows")

    return DatasetSplits(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
    )


def build_binary_labels(raw_labels: pd.Series) -> np.ndarray:
    """
    Convert the label column into a binary 0/1 attack label.

    - If numeric: 0 = benign, >0 = attack.
    - If non-numeric: create a mapping; first unique value is 0, others are 1.
    """
    if pd.api.types.is_numeric_dtype(raw_labels):
        # Many UNSW variants use 0 for normal, 1 for attack
        arr = raw_labels.to_numpy()
        y = (arr > 0).astype(int)
        print("[UNSW] Interpreting numeric labels: 0 = normal, >0 = attack")
        return y

    # Non-numeric label column (e.g., "normal", "attack")
    uniques = list(raw_labels.dropna().unique())
    print(f"[UNSW] Non-numeric labels detected, uniques: {uniques}")

    if len(uniques) == 1:
        # Degenerate case: only one class -> all zeros
        y = np.zeros(len(raw_labels), dtype=int)
        print(f"[UNSW] Only one label value present ({uniques[0]!r}), treating all as 0.")
        return y

    primary_normal = uniques[0]
    mapping: Dict[Any, int] = {primary_normal: 0}
    for v in uniques[1:]:
        mapping[v] = 1

    print(f"[UNSW] Label mapping applied: {mapping}")
    y = raw_labels.map(mapping).fillna(0).astype(int).to_numpy()
    return y


# MODEL TRAINING

def build_pipeline() -> Pipeline:
    """
    Build a simple but solid baseline pipeline:
    - StandardScaler
    - RandomForestClassifier
    """
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )
    return pipe


def train_and_evaluate() -> None:
    data = load_unsw_dataset()
    pipe = build_pipeline()

    print("[UNSW] Training RandomForest pipeline...")
    pipe.fit(data.X_train, data.y_train)

    print("[UNSW] Evaluating on test split...")
    y_pred = pipe.predict(data.X_test)

    print("\n=== Classification report (binary attack vs normal) ===")
    print(classification_report(data.y_test, y_pred, digits=4))

    cm = confusion_matrix(data.y_test, y_pred)
    print("=== Confusion matrix ===")
    print(cm)

    # Make sure the models/ directory exists
    os.makedirs(os.path.dirname(MODEL_BUNDLE_PATH), exist_ok=True)

    bundle = {
        "pipeline": pipe,
        "feature_names": data.feature_names,
        "label_column": LABEL_COLUMN,
    }
    joblib.dump(bundle, MODEL_BUNDLE_PATH)

    print(f"\n[UNSW] Model bundle saved to: {MODEL_BUNDLE_PATH}")


if __name__ == "__main__":
    train_and_evaluate()
