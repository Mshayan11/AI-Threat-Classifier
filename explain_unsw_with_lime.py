from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
UNSW_CSV_PATH = os.path.join("data", "UNSW-NB15.csv")
MODEL_BUNDLE_PATH = os.path.join("models", "unsw_rf_model.joblib")
OUTPUT_DIR = os.path.join("outputs", "lime")
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
    bundle = joblib.load(MODEL_BUNDLE_PATH)
    pipe = bundle["pipeline"]
    feature_names = bundle["feature_names"]
    print(f"[LIME] Loading dataset from: {UNSW_CSV_PATH}")
    df = pd.read_csv(
        UNSW_CSV_PATH,
        encoding="latin1",
        on_bad_lines="skip",
        engine="python",
    )
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise KeyError(
            f"The following feature columns are missing from {UNSW_CSV_PATH}: {missing}\n"
            f"Available columns include: {list(df.columns)[:20]} ..."
        )

    X = df[feature_names].values

    os.makedirs(OUTPUT_DIR, exist_ok=True)

   explainer = LimeTabularExplainer(
        training_data=X,
        feature_names=feature_names,
        class_names=["benign", "attack"],
        discretize_continuous=True,
        mode="classification",
        verbose=False,
    )
    print("[LIME] Computing model probabilities to pick a strong attack example...")
    probs = pipe.predict_proba(X)[:, 1]  
    idx = int(np.argmax(probs))
    x_example = X[idx]

    print(f"[LIME] Selected row index {idx} with P_attack={probs[idx]:.3f}")

    exp = explainer.explain_instance(
        data_row=x_example,
        predict_fn=pipe.predict_proba,
        num_features=min(10, len(feature_names)),
        top_labels=1,
    )

    html_path = os.path.join(OUTPUT_DIR, "lime_example_attack.html")
    exp.save_to_file(html_path)
    print(f"[LIME] Saved LIME explanation HTML to {html_path}")


if __name__ == "__main__":
    main()

