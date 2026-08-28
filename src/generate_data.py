"""
Synthetic data generator for the warranty-claim-fraud-iot project.

Scenario
--------
A manufacturer sells IoT-connected smart water heaters. Each unit streams a
rolling 90-day buffer of daily telemetry (temperature, pressure, vibration,
runtime, power draw, error codes, and a telemetry-uptime flag) to the cloud.
When a customer files a warranty claim, the last 90 days of telemetry for
that device are pulled for investigation.

A small fraction of claims are fraudulent: the claimed failure was actually
caused by misuse that voids the warranty, damage that pre-dated the claim,
or usage patterns that were altered/hidden (e.g. disabling the gateway)
right before filing. This script simulates that generative process end to
end, including realistic noise and a small amount of *label noise* (a
fraction of legitimate claims get investigated/flagged incorrectly, and a
fraction of illegitimate claims slip through), so the resulting task is
learnable but not trivial -- similar to real fraud data.

Ground-truth fields such as `true_scenario` and `true_cause` are written to
data/raw/ for transparency and EDA only. They are NEVER used as model
features (see src/features.py) -- a real fraud team would not have them
either.

Output
------
data/raw/devices.csv        one row per device
data/raw/claims.csv         one row per warranty claim (includes is_fraud)
data/raw/sensor_logs.csv    one row per (claim_id, day_offset) telemetry reading
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

RNG_SEED = 42
N_CUSTOMERS = 3200
FRAUD_PRONE_CUSTOMER_SHARE = 0.045  # small serial-filer segment
N_CLAIMS = 4200
WINDOW_DAYS = 90  # rolling telemetry buffer pulled per claim
WARRANTY_MONTHS = 24

MODELS = {
    # model_name: (rated_max_temp_c, rated_max_pressure_kpa, rated_duty_cycle_hrs_day, base_price)
    "AquaHeat SH-200": (65, 600, 4, 420),
    "AquaHeat SH-400 Pro": (75, 750, 6, 640),
    "AquaHeat SH-Compact": (60, 550, 3, 310),
    "AquaHeat SH-Commercial X": (80, 900, 10, 980),
}
REGIONS = ["North", "South", "East", "West", "Central"]
CLAIM_REASONS = ["no_heat", "leak", "unusual_noise", "error_code_fault", "inconsistent_temp", "other"]

SCENARIOS = [
    # name, weight, fraud_probability
    ("normal_defect", 0.82, 0.015),
    ("misuse_claimed_as_defect", 0.05, 0.75),
    ("pre_existing_damage", 0.03, 0.75),
    ("tamper_before_failure", 0.025, 0.85),
    ("timing_gaming", 0.025, 0.45),
    ("heavy_but_legit_use", 0.05, 0.03),
]


def _rng():
    return np.random.default_rng(RNG_SEED)


def generate_customers(rng: np.random.Generator) -> pd.DataFrame:
    customer_ids = [f"CUST-{i:06d}" for i in range(N_CUSTOMERS)]
    is_fraud_prone = rng.random(N_CUSTOMERS) < FRAUD_PRONE_CUSTOMER_SHARE
    n_devices = rng.choice([1, 2, 3], size=N_CUSTOMERS, p=[0.72, 0.22, 0.06])
    return pd.DataFrame(
        {
            "customer_id": customer_ids,
            "is_fraud_prone_segment": is_fraud_prone,  # ground truth only, not a feature
            "n_devices_owned": n_devices,
        }
    )


def generate_devices(rng: np.random.Generator, customers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    device_counter = 0
    model_names = list(MODELS.keys())
    for _, cust in customers.iterrows():
        for _ in range(cust["n_devices_owned"]):
            model = rng.choice(model_names, p=[0.42, 0.28, 0.22, 0.08])
            rated_temp, rated_pressure, rated_duty, price = MODELS[model]
            install_offset_days = int(rng.integers(30, 900))  # relative "days ago" anchor
            rows.append(
                {
                    "device_id": f"DEV-{device_counter:07d}",
                    "customer_id": cust["customer_id"],
                    "model": model,
                    "region": rng.choice(REGIONS),
                    "rated_max_temp_c": rated_temp,
                    "rated_max_pressure_kpa": rated_pressure,
                    "rated_duty_cycle_hrs_day": rated_duty,
                    "unit_price_usd": price,
                    "install_days_ago": install_offset_days,
                    "warranty_months": WARRANTY_MONTHS,
                }
            )
            device_counter += 1
    return pd.DataFrame(rows)


def _pick_scenario(rng: np.random.Generator, fraud_prone: bool) -> tuple[str, float]:
    names = [s[0] for s in SCENARIOS]
    weights = np.array([s[1] for s in SCENARIOS], dtype=float)
    if fraud_prone:
        # fraud-prone customers are more likely to end up in a fraud-associated scenario
        boost = np.array([0.75, 1.5, 1.4, 1.5, 1.4, 0.85])
        weights = weights * boost
    weights = weights / weights.sum()
    idx = rng.choice(len(names), p=weights)
    return SCENARIOS[idx][0], SCENARIOS[idx][2]


def generate_claims(rng: np.random.Generator, customers: pd.DataFrame, devices: pd.DataFrame) -> pd.DataFrame:
    fraud_prone_lookup = customers.set_index("customer_id")["is_fraud_prone_segment"].to_dict()

    # sample which devices file a claim; fraud-prone customers' devices are oversampled
    device_weight = devices["customer_id"].map(fraud_prone_lookup).map({True: 1.8, False: 1.0}).fillna(1.0)
    device_weight = device_weight / device_weight.sum()
    claim_device_idx = rng.choice(devices.index, size=N_CLAIMS, replace=True, p=device_weight.values)

    rows = []
    for i, dev_idx in enumerate(claim_device_idx):
        dev = devices.loc[dev_idx]
        fraud_prone = bool(fraud_prone_lookup.get(dev["customer_id"], False))
        scenario, base_fraud_p = _pick_scenario(rng, fraud_prone)

        # days_since_install: pre_existing_damage scenario fails early; timing_gaming fails near warranty end
        warranty_days = dev["warranty_months"] * 30
        if scenario == "pre_existing_damage":
            days_since_install = int(rng.integers(3, 45))
        elif scenario == "timing_gaming":
            days_since_install = int(rng.integers(int(warranty_days * 0.88), warranty_days - 2))
        else:
            days_since_install = int(rng.integers(20, warranty_days + 30))  # a few claims arrive just past expiry

        days_to_warranty_expiry = warranty_days - days_since_install
        claim_reason = rng.choice(
            CLAIM_REASONS,
            p={
                "misuse_claimed_as_defect": [0.10, 0.20, 0.15, 0.10, 0.10, 0.35],
                "pre_existing_damage": [0.30, 0.30, 0.10, 0.10, 0.15, 0.05],
                "tamper_before_failure": [0.35, 0.15, 0.10, 0.20, 0.10, 0.10],
                "timing_gaming": [0.25, 0.10, 0.15, 0.20, 0.20, 0.10],
                "heavy_but_legit_use": [0.20, 0.05, 0.20, 0.15, 0.30, 0.10],
                "normal_defect": [0.22, 0.18, 0.15, 0.20, 0.15, 0.10],
            }[scenario],
        )

        # resolve fraud with label noise: ~4% of "should be fraud" cases slip through undetected label as 0,
        # and ~3% of legit cases get an incorrect fraud flag from an overzealous/incorrect investigation.
        raw_fraud = rng.random() < base_fraud_p
        if raw_fraud:
            is_fraud = 0 if rng.random() < 0.04 else 1
        else:
            is_fraud = 1 if rng.random() < 0.03 else 0

        rows.append(
            {
                "claim_id": f"CLM-{i:07d}",
                "device_id": dev["device_id"],
                "customer_id": dev["customer_id"],
                "model": dev["model"],
                "days_since_install": days_since_install,
                "days_to_warranty_expiry": days_to_warranty_expiry,
                "claim_reason": claim_reason,
                "true_scenario": scenario,  # ground truth, NOT a model feature
                "is_fraud": int(is_fraud),
            }
        )

    claims = pd.DataFrame(rows)

    # anchor claim_date so the dataset "ends" today and claims are spread over the last ~2 years
    today = pd.Timestamp.today().normalize()
    claim_offsets = rng.integers(1, 730, size=len(claims))
    claims["claim_date"] = [today - timedelta(days=int(d)) for d in claim_offsets]
    claims = claims.sort_values("claim_date").reset_index(drop=True)
    return claims


def _simulate_sensor_window(rng: np.random.Generator, dev: pd.Series, claim: pd.Series) -> pd.DataFrame:
    scenario = claim["true_scenario"]
    rated_temp = dev["rated_max_temp_c"]
    rated_pressure = dev["rated_max_pressure_kpa"]
    rated_duty = dev["rated_duty_cycle_hrs_day"]

    days = np.arange(-WINDOW_DAYS + 1, 1)  # day_offset: -89 .. 0 (0 = claim date)

    # baseline "normal" usage curves with gentle noise
    temp = rated_temp * 0.75 + rng.normal(0, 2.0, size=WINDOW_DAYS)
    pressure = rated_pressure * 0.70 + rng.normal(0, 15.0, size=WINDOW_DAYS)
    vibration = 1.2 + rng.normal(0, 0.15, size=WINDOW_DAYS)
    runtime = rated_duty * 0.6 + rng.normal(0, 0.4, size=WINDOW_DAYS)
    power = runtime * 1.5 + rng.normal(0, 0.3, size=WINDOW_DAYS)
    error_codes = rng.poisson(0.02, size=WINDOW_DAYS)
    telemetry_received = np.ones(WINDOW_DAYS, dtype=bool)

    if scenario == "misuse_claimed_as_defect":
        # sustained overuse beyond rated spec for most of the window
        temp += rated_temp * 0.28
        pressure += rated_pressure * 0.25
        runtime += rated_duty * 0.9
        vibration += 0.6
        power = runtime * 1.5 + rng.normal(0, 0.4, size=WINDOW_DAYS)
        error_codes = rng.poisson(0.12, size=WINDOW_DAYS)

    elif scenario == "heavy_but_legit_use":
        # elevated but still under/near rated spec -- looks similar but stays within bounds
        temp += rated_temp * 0.12
        runtime += rated_duty * 0.35
        power = runtime * 1.5 + rng.normal(0, 0.3, size=WINDOW_DAYS)

    elif scenario == "pre_existing_damage":
        # anomalous readings from the very start of the (short) observation window
        temp += rng.normal(8, 3, size=WINDOW_DAYS)
        vibration += rng.normal(1.8, 0.4, size=WINDOW_DAYS).clip(min=0)
        error_codes = rng.poisson(0.35, size=WINDOW_DAYS)

    elif scenario == "tamper_before_failure":
        # normal-looking for most of window, then a telemetry blackout + spike right before claim
        blackout_len = int(rng.integers(4, 14))
        spike_start = WINDOW_DAYS - blackout_len - int(rng.integers(1, 4))
        temp[spike_start:] += rated_temp * 0.35
        vibration[spike_start:] += 1.1
        runtime[spike_start:] += rated_duty * 0.8
        error_codes[spike_start:] = rng.poisson(0.2, size=len(error_codes[spike_start:]))
        telemetry_received[-blackout_len:] = False

    elif scenario == "timing_gaming":
        # quiet usage, then a sudden spike in the final week right before warranty expiry
        spike_len = int(rng.integers(4, 9))
        temp[-spike_len:] += rated_temp * 0.22
        runtime[-spike_len:] += rated_duty * 1.1
        error_codes[-spike_len:] = rng.poisson(0.18, size=spike_len)

    # small amount of "innocent" telemetry dropout for everyone (wifi hiccups etc.)
    innocent_drop = rng.random(WINDOW_DAYS) < 0.015
    telemetry_received = telemetry_received & ~innocent_drop

    max_temp = temp + rng.normal(3, 1, size=WINDOW_DAYS)
    max_pressure = pressure + rng.normal(25, 8, size=WINDOW_DAYS)

    df = pd.DataFrame(
        {
            "claim_id": claim["claim_id"],
            "day_offset": days,
            "date": [claim["claim_date"] + timedelta(days=int(d)) for d in days],
            "avg_temp_c": temp.round(2),
            "max_temp_c": max_temp.round(2),
            "avg_pressure_kpa": pressure.round(1),
            "max_pressure_kpa": max_pressure.round(1),
            "vibration_rms_mm_s": vibration.clip(min=0).round(3),
            "runtime_hours": runtime.clip(min=0).round(2),
            "power_draw_kwh": power.clip(min=0).round(2),
            "error_code_count": error_codes,
            "telemetry_received": telemetry_received,
        }
    )
    # for devices younger than the window, there's no telemetry before install
    df = df[df["day_offset"] >= -claim["days_since_install"]]
    return df


def generate_sensor_logs(rng: np.random.Generator, devices: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    devices_idx = devices.set_index("device_id")
    chunks = []
    for _, claim in claims.iterrows():
        dev = devices_idx.loc[claim["device_id"]]
        chunks.append(_simulate_sensor_window(rng, dev, claim))
    logs = pd.concat(chunks, ignore_index=True)
    # non-reported days: sensor values are simply absent (NaN), only telemetry_received=False is known
    missing_mask = ~logs["telemetry_received"]
    sensor_cols = [
        "avg_temp_c", "max_temp_c", "avg_pressure_kpa", "max_pressure_kpa",
        "vibration_rms_mm_s", "runtime_hours", "power_draw_kwh", "error_code_count",
    ]
    logs.loc[missing_mask, sensor_cols] = np.nan
    return logs


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rng = _rng()
    customers = generate_customers(rng)
    devices = generate_devices(rng, customers)
    claims = generate_claims(rng, customers, devices)
    sensor_logs = generate_sensor_logs(rng, devices, claims)

    customers.to_csv(raw_dir / "customers.csv", index=False)
    devices.to_csv(raw_dir / "devices.csv", index=False)
    claims.to_csv(raw_dir / "claims.csv", index=False)
    sensor_logs.to_csv(raw_dir / "sensor_logs.csv", index=False)

    print(f"customers:   {len(customers):>7,} rows -> {raw_dir / 'customers.csv'}")
    print(f"devices:     {len(devices):>7,} rows -> {raw_dir / 'devices.csv'}")
    print(f"claims:      {len(claims):>7,} rows -> {raw_dir / 'claims.csv'}")
    print(f"sensor_logs: {len(sensor_logs):>7,} rows -> {raw_dir / 'sensor_logs.csv'}")
    print(f"fraud rate:  {claims['is_fraud'].mean():.2%}")


if __name__ == "__main__":
    main()
