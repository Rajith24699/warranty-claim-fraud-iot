"""
SHAP explainability for the trained XGBoost fraud model.

Produces reports/figures/shap_summary.png, a beeswarm plot showing which
engineered features push individual claims toward / away from a fraud
prediction. Intended for a fraud-investigator audience: "why did the model
flag this claim?" rather than just "what is globally important?".
"""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from train import load_data, split_xy, ID_COLS, GROUND_TRUTH_COLS, LABEL_COL, CATEGORICAL_COLS
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
FIG_DIR = BASE_DIR / "reports" / "figures"
RANDOM_STATE = 42


def main():
    df = load_data()
    X, y = split_xy(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    pipe = joblib.load(MODEL_DIR / "fraud_model.joblib")

    # recompute anomaly_score for this X_test the same way train.py did, using the saved IsolationForest
    iso = joblib.load(MODEL_DIR / "isolation_forest.joblib")
    from sklearn.impute import SimpleImputer
    import json
    with open(MODEL_DIR / "feature_columns.json") as f:
        cols = json.load(f)
    numeric_cols_no_anom = [c for c in cols["numeric"] if c != "anomaly_score"]
    imputer = SimpleImputer(strategy="median")
    test_num = imputer.fit_transform(X_test[numeric_cols_no_anom])
    X_test = X_test.copy()
    X_test["anomaly_score"] = -iso.score_samples(test_num)

    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]
    X_test_t = prep.transform(X_test)
    feature_names = prep.get_feature_names_out()

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test_t)

    # RandomForestClassifier (multi-output) returns either a list [class0, class1]
    # or a 3D array (n_samples, n_features, n_classes) depending on the shap version.
    # XGBClassifier (single-output margin) returns a plain 2D array. Normalize to 2D,
    # class-1 (fraud) SHAP values in all cases.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test_t, feature_names=feature_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot -> {FIG_DIR / 'shap_summary.png'}")


if __name__ == "__main__":
    main()
