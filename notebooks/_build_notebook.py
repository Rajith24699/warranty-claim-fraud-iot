"""One-off script that assembles 01_eda_and_modeling.ipynb via nbformat.
Run once (or whenever the notebook needs regenerating), then execute with:
    jupyter nbconvert --to notebook --execute --inplace 01_eda_and_modeling.ipynb
This file is not part of the analysis itself -- it just builds the notebook.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

md = lambda src: cells.append(nbf.v4.new_markdown_cell(src))
code = lambda src: cells.append(nbf.v4.new_code_cell(src))

md(r"""# Warranty Claim Fraud Detection via IoT Sensor Logs

**Goal:** flag warranty claims on smart-appliance IoT devices that are likely fraudulent
(misuse concealed as a manufacturing defect, pre-existing damage, telemetry tampering, or
"just-in-time" claims filed right before warranty expiry) using the device's own sensor
telemetry plus claim metadata.

**Why this matters:** warranty fraud costs manufacturers real margin, and the same
device-generated telemetry that improves customer experience (predictive maintenance,
remote diagnostics) doubles as an audit trail investigators rarely exploit systematically.
This project builds that audit trail into a ranked worklist for a fraud-review team.

**Data:** synthetic but structurally realistic. See [`src/generate_data.py`](../src/generate_data.py)
for the exact generative process (usage scenarios, telemetry patterns, and the label-noise
model used to keep the task learnable-but-not-trivial, like real fraud data). Ground-truth
fields used only to *build* the simulation (`true_scenario`) are never used as model inputs
-- see [`src/features.py`](../src/features.py).

**Pipeline:** `generate_data.py` &rarr; `features.py` &rarr; `train.py` &rarr; `explain.py`.
This notebook re-loads the already-computed feature table and trained-model artifacts so it
stays fast to read; regenerate everything from scratch with:
```bash
python src/generate_data.py && python src/features.py && python src/train.py && python src/explain.py
```
""")

code(r"""import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display, Markdown

sns.set_theme(style="whitegrid", palette="deep")
BASE_DIR = Path.cwd().parent
FIG_DIR = BASE_DIR / "reports" / "figures"

claims_features = pd.read_csv(BASE_DIR / "data" / "processed" / "claims_features.csv", parse_dates=["claim_date"])
print(claims_features.shape)
claims_features.head()
""")

md("## 1. Exploratory data analysis")

code(r"""fraud_rate = claims_features["is_fraud"].mean()
print(f"Claims: {len(claims_features):,}")
print(f"Fraud rate: {fraud_rate:.2%}")

ax = claims_features["is_fraud"].value_counts().sort_index().plot(
    kind="bar", figsize=(4, 4), color=["#4C72B0", "#C44E52"]
)
ax.set_xticklabels(["legitimate", "fraud"], rotation=0)
ax.set_ylabel("claims")
ax.set_title("Class balance")
plt.tight_layout()
plt.show()
""")

code(r"""fig, axes = plt.subplots(1, 2, figsize=(12, 4))

reason_rate = claims_features.groupby("claim_reason")["is_fraud"].mean().sort_values(ascending=False)
sns.barplot(x=reason_rate.values, y=reason_rate.index, ax=axes[0], color="#4C72B0")
axes[0].set_title("Fraud rate by stated claim reason")
axes[0].set_xlabel("fraud rate")

model_rate = claims_features.groupby("model")["is_fraud"].mean().sort_values(ascending=False)
sns.barplot(x=model_rate.values, y=model_rate.index, ax=axes[1], color="#55A868")
axes[1].set_title("Fraud rate by device model")
axes[1].set_xlabel("fraud rate")

plt.tight_layout()
plt.show()
""")

code(r"""fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, ["vibration_max", "pct_days_over_rated_temp", "telemetry_uptime_pct_recent14"]):
    sns.boxplot(data=claims_features, x="is_fraud", y=col, ax=ax, palette=["#4C72B0", "#C44E52"])
    ax.set_xticklabels(["legit", "fraud"])
    ax.set_title(col)
plt.tight_layout()
plt.show()
""")

md(r"""### A closer look at raw telemetry

`data/samples/` ships a handful of full 90-day telemetry windows (one full run of
`generate_data.py` produces ~26MB of sensor logs across all 4,200 claims, which is why the
full `sensor_logs.csv` isn't committed -- regenerate it locally if you want the whole thing).
Below: a legitimate `normal_defect` claim next to a `tamper_before_failure` claim. Watch the
telemetry-uptime line in the last ~10 days before the claim date (`day_offset = 0`).""")

code(r"""samples = pd.read_csv(BASE_DIR / "data" / "samples" / "sensor_logs_sample.csv", parse_dates=["date"])
sample_claims = pd.read_csv(BASE_DIR / "data" / "samples" / "claims_sample.csv")

example_ids = {
    "normal_defect": sample_claims.loc[sample_claims["true_scenario"] == "normal_defect", "claim_id"].iloc[0],
    "tamper_before_failure": sample_claims.loc[sample_claims["true_scenario"] == "tamper_before_failure", "claim_id"].iloc[0],
}

fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
for col_idx, (scenario, claim_id) in enumerate(example_ids.items()):
    g = samples[samples["claim_id"] == claim_id].sort_values("day_offset")
    axes[0, col_idx].plot(g["day_offset"], g["avg_temp_c"], color="#C44E52")
    axes[0, col_idx].set_title(f"{scenario}\n({claim_id})")
    axes[0, col_idx].set_ylabel("avg_temp_c")

    axes[1, col_idx].plot(g["day_offset"], g["telemetry_received"].astype(int), color="#4C72B0", drawstyle="steps-post")
    axes[1, col_idx].set_ylabel("telemetry_received")
    axes[1, col_idx].set_xlabel("day_offset (0 = claim date)")
    axes[1, col_idx].set_yticks([0, 1])

plt.tight_layout()
plt.show()
""")

md("## 2. Modeling recap")

md(r"""Full pipeline lives in [`src/train.py`](../src/train.py):

1. Stratified 75/25 train/test split.
2. An `IsolationForest` is fit on the training claims' sensor-derived features (unsupervised,
   no label leakage) to produce an `anomaly_score` -- how unusual a claim's telemetry looks
   relative to the training distribution. That score is added as one more feature.
3. Three supervised classifiers are trained with class-imbalance handling and compared:
   Logistic Regression (baseline), Random Forest, and XGBoost.
4. The best model by PR-AUC is kept and evaluated with ROC-AUC, PR-AUC, and
   **precision-at-top-k%** -- the operational framing: *"if the fraud team can only review
   the top k% of claims ranked by risk, what fraction of that review queue is actually
   fraud?"*
""")

code(r"""with open(BASE_DIR / "reports" / "metrics.json") as f:
    metrics = json.load(f)

comparison = pd.DataFrame(metrics["comparison"]).set_index("model")
display(Markdown(f"**Best model:** `{metrics['best_model']}`  |  **Test claims:** {metrics['n_test']:,}  |  **Fraud rate:** {metrics['fraud_rate_overall']:.2%}"))
comparison
""")

code(r"""display(Image(filename=str(FIG_DIR / "pr_curve.png")))""")
code(r"""display(Image(filename=str(FIG_DIR / "roc_curve.png")))""")
code(r"""display(Image(filename=str(FIG_DIR / "precision_at_k.png")))""")
code(r"""display(Image(filename=str(FIG_DIR / "confusion_matrix.png")))""")
code(r"""display(Image(filename=str(FIG_DIR / "feature_importance.png")))""")

md(r"""### Explainability

`src/explain.py` runs SHAP's `TreeExplainer` on the trained model so an investigator can see
*why* a specific claim was flagged, not just that it was. The top drivers line up with the
scenarios baked into the simulation: unusual vibration/temperature relative to the device's
rated spec, a high anomaly score, and a recent usage spike versus the claim's earlier
telemetry.""")

code(r"""display(Image(filename=str(FIG_DIR / "shap_summary.png")))""")

md(r"""## 3. Takeaways

- **Telemetry beats claim metadata alone.** Sensor-derived features (`anomaly_score`,
  `vibration_max`, `pct_days_over_rated_temp`, the recent-vs-prior usage spike ratio) dominate
  the top of the feature-importance ranking -- claim-form fields like `claim_reason` carry
  comparatively little signal on their own.
- **Precision-at-top-k is the metric that matters operationally.** A fraud-review team with
  capacity for ~10% of incoming claims can expect roughly 3-4x the precision of blind
  sampling by working the model's ranked queue instead (see the precision-at-top-k plot).
- **Telemetry gaps are a real tell.** The `tamper_before_failure` example above shows the
  mechanism directly: normal-looking telemetry, then a blackout window right before the
  claim, then a spike. `telemetry_uptime_pct_recent14` and `max_consecutive_missing_days`
  exist specifically to capture that.

### Limitations & next steps

- Labels are simulated; a production system would need investigator feedback (confirmed
  fraud/not-fraud outcomes) to validate and recalibrate against.
- The fraud-generating scenarios here are hand-authored; real fraud patterns drift over time
  and a deployed model would need periodic retraining and drift monitoring.
- No cost-sensitive threshold tuning was done beyond the top-k framing -- a real deployment
  should weigh the cost of a false accusation (customer trust, legal exposure) against a
  missed fraud (payout cost) explicitly.
""")

nb["cells"] = cells
with open("01_eda_and_modeling.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote 01_eda_and_modeling.ipynb")
