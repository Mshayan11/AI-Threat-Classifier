

from __future__ import annotations

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# Paths must match train_unsw_model.py
UNSW_CSV_PATH = os.path.join("data", "UNSW-NB15.csv")
MODEL_BUNDLE_PATH = os.path.join("models", "unsw_rf_model.joblib")
OUTPUT_DIR = os.path.join("outputs", "shap")


def main() -> None:
    if not os.path.exists(MODEL_BUNDLE_PATH):
        raise FileNotFoundError(
            f"Model bundle not found at {MODEL_BUNDLE_PATH}. "
            f"Run train_unsw_model.py first."
        )

    if not os.path.exists(UNSW_CSV_PATH):
        raise FileNotFoundError(
            f"UNSW CSV not found at {UNSW_CSV_PATH}. "
            f"Put your UNSW-NB15.csv there or update UNSW_CSV_PATH."
        )

    # Load trained model bundle
    bundle = joblib.load(MODEL_BUNDLE_PATH)
    pipe = bundle["pipeline"]
    feature_names = bundle["feature_names"]

    # Load the dataset using the SAME options as train_unsw_model.py
    print(f"[SHAP] Loading dataset from: {UNSW_CSV_PATH}")
    df = pd.read_csv(
        UNSW_CSV_PATH,
        encoding="latin1",
        on_bad_lines="skip",
        engine="python",
    )

    # Sanity check: all features must be present
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise KeyError(
            f"The following feature columns are missing from {UNSW_CSV_PATH}: {missing}\n"
            f"Available columns include: {list(df.columns)[:20]} ..."
        )

    X = df[feature_names]

    # Use a manageable subset for SHAP background + summary
    n_sample = min(1000, len(X))
    X_sample = X.sample(n=n_sample, random_state=42)
    print(f"[SHAP] Using a sample of {n_sample} rows for explanations.")

    # Get the underlying RandomForest (after preprocessing)
    rf = pipe.named_steps["rf"]
    scaler = pipe.named_steps["scaler"]

    # SHAP for tree-based models
    explainer = shap.TreeExplainer(rf)

    # Scale the sample as the model sees it
    X_scaled = scaler.transform(X_sample.values)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Global summary for the "attack" class (1)
    shap_values = explainer.shap_values(X_scaled)

    # In binary classification, shap_values is a list [class0, class1]
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_attack = shap_values[1]
    else:
        shap_attack = shap_values

    plt.figure()
    shap.summary_plot(
        shap_attack,
        X_sample,
        feature_names=feature_names,
        show=False,
    )
    plt.title("SHAP summary plot for attack class (UNSW RandomForest)")
    plt.tight_layout()
    summary_path = os.path.join(OUTPUT_DIR, "shap_summary_attack.png")
    plt.savefig(summary_path, dpi=200)
    plt.close()
    print(f"[SHAP] Saved SHAP summary plot to {summary_path}")

    # Single-example local explanation for a high-probability attack
    probs = pipe.predict_proba(X_sample.values)[:, 1]
    idx = int(np.argmax(probs))
    x_example = X_sample.iloc[[idx]]
    x_scaled = scaler.transform(x_example.values)

    shap_vals_example = explainer.shap_values(x_scaled)

    if isinstance(shap_vals_example, list) and len(shap_vals_example) > 1:
        shap_example_attack = shap_vals_example[1][0]
    else:
        shap_example_attack = shap_vals_example[0]

    plt.figure()
    shap.bar_plot(
        shap_example_attack,
        feature_names=feature_names,
        show=False,
    )
    plt.title("Local SHAP explanation for high-probability attack sample")
    plt.tight_layout()
    local_path = os.path.join(OUTPUT_DIR, "shap_local_attack_bar.png")
    plt.savefig(local_path, dpi=200)
    plt.close()
    print(f"[SHAP] Saved local SHAP bar plot to {local_path}")


if __name__ == "__main__":
    main()
