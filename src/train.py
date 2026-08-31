"""
Train and evaluate warranty-claim fraud models.

Pipeline:
  1. Load data/processed/claims_features.csv (run features.py first if missing).
  2. Split train/test (stratified on is_fraud).
  3. Fit an IsolationForest on the *presumed-legitimate* training claims to
     produce an `anomaly_score` -- a classic fraud-analytics trick: score how
     unusual a claim's telemetry looks relative to normal claims, then feed
     that score into the supervised model as an extra feature.
  4. Fit three supervised classifiers (Logistic Regression baseline, Random
     Forest, XGBoost) with class-imbalance handling, compare them, and keep
     XGBoost as the final model.
  5. Evaluate with ROC-AUC, PR-AUC, precision/recall/F1, and a
     precision-at-top-k% curve (the investigator-capacity framing: "if the
     fraud team can only review the top 10% of claims, how much fraud do we
     catch?").
  6. Save plots to reports/figures/, metrics to reports/metrics.json, and the
     trained model + feature list to models/.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, classification_report, confusion_matrix,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "reports" / "figures"
MODEL_DIR = BASE_DIR / "models"
RANDOM_STATE = 42

ID_COLS = ["claim_id", "device_id", "customer_id", "claim_date"]
GROUND_TRUTH_COLS = ["true_scenario"]
LABEL_COL = "is_fraud"
CATEGORICAL_COLS = ["model", "claim_reason"]


def load_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "claims_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run src/generate_data.py then src/features.py first.")
    return pd.read_csv(path, parse_dates=["claim_date"])


def split_xy(df: pd.DataFrame):
    drop_cols = ID_COLS + GROUND_TRUTH_COLS + [LABEL_COL]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[LABEL_COL].astype(int)
    return X, y


def add_anomaly_score(X_train, X_test, numeric_cols):
    """Fit IsolationForest on presumed-legit *train* rows only; score both splits."""
    imputer = SimpleImputer(strategy="median")
    train_num = imputer.fit_transform(X_train[numeric_cols])
    test_num = imputer.transform(X_test[numeric_cols])

    iso = IsolationForest(n_estimators=300, contamination=0.08, random_state=RANDOM_STATE)
    iso.fit(train_num)  # trained on the full train split (unsupervised -- doesn't see labels)

    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["anomaly_score"] = -iso.score_samples(train_num)  # higher = more anomalous
    X_test["anomaly_score"] = -iso.score_samples(test_num)
    return X_train, X_test, iso


def build_preprocessor(numeric_cols, categorical_cols):
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", __import__("sklearn").preprocessing.OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])


def precision_at_k(y_true, scores, k_pct):
    n = len(y_true)
    k = max(1, int(np.ceil(n * k_pct)))
    order = np.argsort(scores)[::-1][:k]
    return float(np.mean(np.asarray(y_true)[order]))


def evaluate_model(name, y_true, y_proba):
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    p_at_5 = precision_at_k(y_true, y_proba, 0.05)
    p_at_10 = precision_at_k(y_true, y_proba, 0.10)
    p_at_20 = precision_at_k(y_true, y_proba, 0.20)
    return {
        "model": name,
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision_at_top5pct": round(p_at_5, 4),
        "precision_at_top10pct": round(p_at_10, 4),
        "precision_at_top20pct": round(p_at_20, 4),
    }


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X, y = split_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    numeric_cols = [c for c in X_train.columns if c not in CATEGORICAL_COLS]
    X_train, X_test, iso_model = add_anomaly_score(X_train, X_test, numeric_cols)
    numeric_cols = numeric_cols + ["anomaly_score"]

    preprocessor = build_preprocessor(numeric_cols, CATEGORICAL_COLS)

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    models = {
        "logistic_regression": Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=400, max_depth=8, class_weight="balanced_subsample",
                random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ]),
        "xgboost": Pipeline([
            ("prep", preprocessor),
            ("clf", XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
                random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ]),
    }

    results = []
    proba_by_model = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        proba_by_model[name] = proba
        results.append(evaluate_model(name, y_test, proba))
        print(f"[{name}] fitted.")

    results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    print("\n=== Model comparison (test set) ===")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["model"]
    best_pipe = models[best_name]
    best_proba = proba_by_model[best_name]

    # ---- classification report at a chosen operating threshold ----
    threshold = 0.5
    y_pred = (best_proba >= threshold).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics_out = {
        "best_model": best_name,
        "comparison": results_df.to_dict(orient="records"),
        "classification_report_at_0.5": report,
        "confusion_matrix_at_0.5": cm,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "fraud_rate_overall": round(float(y.mean()), 4),
    }
    with open(BASE_DIR / "reports" / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    # ---- plots ----
    plt.figure(figsize=(6, 5))
    for name, proba in proba_by_model.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, proba):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Warranty Claim Fraud Models")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_curve.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 5))
    for name, proba in proba_by_model.items():
        prec, rec, _ = precision_recall_curve(y_test, proba)
        plt.plot(rec, prec, label=f"{name} (AP={average_precision_score(y_test, proba):.3f})")
    baseline = y_test.mean()
    plt.axhline(baseline, color="k", linestyle="--", alpha=0.4, label=f"random baseline ({baseline:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Warranty Claim Fraud Models")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / "pr_curve.png", dpi=150); plt.close()

    ks = np.linspace(0.02, 0.5, 25)
    plt.figure(figsize=(6, 5))
    for name, proba in proba_by_model.items():
        p_at_k = [precision_at_k(y_test, proba, k) for k in ks]
        plt.plot(ks * 100, p_at_k, label=name)
    plt.xlabel("% of claims reviewed (ranked by predicted fraud risk)")
    plt.ylabel("Precision among reviewed claims")
    plt.title("Precision at Top-K% — Investigator Capacity View")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / "precision_at_k.png", dpi=150); plt.close()

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["pred legit", "pred fraud"], yticklabels=["actual legit", "actual fraud"])
    plt.title(f"Confusion Matrix — {best_name} @ threshold {threshold}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=150); plt.close()

    # feature importance (best model, if tree-based) or |coef| (logistic)
    try:
        feature_names = best_pipe.named_steps["prep"].get_feature_names_out()
        clf = best_pipe.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        else:
            importances = np.abs(clf.coef_).ravel()
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False).head(20)
        plt.figure(figsize=(7, 7))
        sns.barplot(data=imp_df, x="importance", y="feature", color="#4C72B0")
        plt.title(f"Top 20 Feature Importances — {best_name}")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "feature_importance.png", dpi=150); plt.close()
    except Exception as e:
        print(f"skipped feature importance plot: {e}")

    joblib.dump(best_pipe, MODEL_DIR / "fraud_model.joblib")
    joblib.dump(iso_model, MODEL_DIR / "isolation_forest.joblib")
    with open(MODEL_DIR / "feature_columns.json", "w") as f:
        json.dump({"numeric": numeric_cols, "categorical": CATEGORICAL_COLS}, f, indent=2)

    print(f"\nBest model: {best_name}")
    print(f"Saved model -> {MODEL_DIR / 'fraud_model.joblib'}")
    print(f"Saved metrics -> {BASE_DIR / 'reports' / 'metrics.json'}")
    print(f"Saved figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
