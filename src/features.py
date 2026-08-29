"""
Feature engineering: turn raw (devices, claims, sensor_logs) tables into one
row per claim with model-ready features + the is_fraud label.

Every feature here is something a warranty-fraud investigator could plausibly
compute from telemetry and claim metadata alone -- no ground-truth /
ground-truth-adjacent columns (true_scenario, is_fraud_prone_segment, etc.)
are used as inputs, only as the label or for post-hoc evaluation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

RECENT_WINDOW_DAYS = 14


def _max_consecutive_false(mask: np.ndarray) -> int:
    """Longest run of False (i.e. missing telemetry) in a boolean array."""
    best = cur = 0
    for v in mask:
        if not v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _claim_features(claim_id: str, g: pd.DataFrame, rated_temp: float, rated_pressure: float, rated_duty: float) -> dict:
    g = g.sort_values("day_offset")
    received = g["telemetry_received"].astype(bool).values
    recent = g[g["day_offset"] >= -RECENT_WINDOW_DAYS]
    prior = g[g["day_offset"] < -RECENT_WINDOW_DAYS]

    def safe_mean(s):
        return float(s.mean()) if len(s) and s.notna().any() else np.nan

    feat = {
        "claim_id": claim_id,
        "n_days_observed": len(g),
        "telemetry_uptime_pct": float(received.mean()) if len(received) else np.nan,
        "telemetry_uptime_pct_recent14": float(recent["telemetry_received"].mean()) if len(recent) else np.nan,
        "max_consecutive_missing_days": _max_consecutive_false(received),

        "temp_mean": safe_mean(g["avg_temp_c"]),
        "temp_max": float(g["max_temp_c"].max()) if g["max_temp_c"].notna().any() else np.nan,
        "temp_std": float(g["avg_temp_c"].std()) if g["avg_temp_c"].notna().sum() > 1 else 0.0,
        "pct_days_over_rated_temp": float((g["avg_temp_c"] > rated_temp).mean(skipna=True)),

        "pressure_mean": safe_mean(g["avg_pressure_kpa"]),
        "pressure_max": float(g["max_pressure_kpa"].max()) if g["max_pressure_kpa"].notna().any() else np.nan,
        "pct_days_over_rated_pressure": float((g["avg_pressure_kpa"] > rated_pressure).mean(skipna=True)),

        "vibration_mean": safe_mean(g["vibration_rms_mm_s"]),
        "vibration_max": float(g["vibration_rms_mm_s"].max()) if g["vibration_rms_mm_s"].notna().any() else np.nan,

        "runtime_hours_mean": safe_mean(g["runtime_hours"]),
        "runtime_hours_mean_recent14": safe_mean(recent["runtime_hours"]),
        "runtime_hours_mean_prior": safe_mean(prior["runtime_hours"]),
        "pct_days_over_rated_duty": float((g["runtime_hours"] > rated_duty).mean(skipna=True)),

        "error_code_total": float(g["error_code_count"].sum(skipna=True)),
        "error_code_recent14": float(recent["error_code_count"].sum(skipna=True)),

        "power_draw_mean": safe_mean(g["power_draw_kwh"]),
    }

    prior_runtime = feat["runtime_hours_mean_prior"]
    recent_runtime = feat["runtime_hours_mean_recent14"]
    if prior_runtime and not np.isnan(prior_runtime) and prior_runtime > 0.05:
        feat["usage_spike_ratio_recent_vs_prior"] = recent_runtime / prior_runtime if not np.isnan(recent_runtime) else np.nan
    else:
        feat["usage_spike_ratio_recent_vs_prior"] = np.nan

    return feat


def build_feature_table() -> pd.DataFrame:
    devices = pd.read_csv(RAW_DIR / "devices.csv")
    claims = pd.read_csv(RAW_DIR / "claims.csv", parse_dates=["claim_date"])
    logs = pd.read_csv(RAW_DIR / "sensor_logs.csv", parse_dates=["date"])

    devices_idx = devices.set_index("device_id")

    rows = []
    for claim_id, g in logs.groupby("claim_id", sort=False):
        device_id = claims.loc[claims["claim_id"] == claim_id, "device_id"].iloc[0]
        dev = devices_idx.loc[device_id]
        rows.append(
            _claim_features(
                claim_id, g,
                rated_temp=dev["rated_max_temp_c"],
                rated_pressure=dev["rated_max_pressure_kpa"],
                rated_duty=dev["rated_duty_cycle_hrs_day"],
            )
        )
    sensor_feats = pd.DataFrame(rows)

    # customer claim-history features computed causally (claims before this claim's date only)
    claims = claims.sort_values("claim_date").reset_index(drop=True)
    prior_claim_counts = []
    seen: dict[str, int] = {}
    for cust in claims["customer_id"]:
        prior_claim_counts.append(seen.get(cust, 0))
        seen[cust] = seen.get(cust, 0) + 1
    claims["customer_prior_claims_count"] = prior_claim_counts

    device_counts = devices.groupby("customer_id").size().rename("customer_devices_owned")
    claims = claims.merge(device_counts, on="customer_id", how="left")

    claims["pct_of_warranty_elapsed"] = (
        claims["days_since_install"] / (claims["days_since_install"] + claims["days_to_warranty_expiry"])
    ).clip(0, 2)

    df = claims.merge(sensor_feats, on="claim_id", how="left")
    df = df.merge(
        devices[["device_id", "rated_max_temp_c", "rated_max_pressure_kpa", "rated_duty_cycle_hrs_day", "unit_price_usd"]],
        on="device_id", how="left",
    )

    # columns kept for reference/EDA but excluded from the model matrix in train.py
    df.attrs["ground_truth_cols"] = ["true_scenario"]
    df.attrs["id_cols"] = ["claim_id", "device_id", "customer_id", "claim_date"]
    df.attrs["label_col"] = "is_fraud"

    return df


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = build_feature_table()
    out_path = PROCESSED_DIR / "claims_features.csv"
    df.to_csv(out_path, index=False)
    print(f"claims_features: {len(df):,} rows, {df.shape[1]} cols -> {out_path}")
    print(f"fraud rate: {df['is_fraud'].mean():.2%}")


if __name__ == "__main__":
    main()
