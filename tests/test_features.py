import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features import _max_consecutive_false, _claim_features  # noqa: E402


def test_max_consecutive_false_basic():
    assert _max_consecutive_false(np.array([True, True, True])) == 0
    assert _max_consecutive_false(np.array([False, False, True, False])) == 2
    assert _max_consecutive_false(np.array([False, False, False])) == 3


def test_max_consecutive_false_empty():
    assert _max_consecutive_false(np.array([])) == 0


def test_claim_features_flags_telemetry_gap_and_over_spec_usage():
    days = np.arange(-9, 1)
    telemetry_received = np.array([True] * 7 + [False] * 3)
    g = pd.DataFrame(
        {
            "day_offset": days,
            "avg_temp_c": [80.0] * 10,      # rated_temp is 60 -> every day over spec
            "max_temp_c": [85.0] * 10,
            "avg_pressure_kpa": [500.0] * 10,
            "max_pressure_kpa": [520.0] * 10,
            "vibration_rms_mm_s": [2.0] * 10,
            "runtime_hours": [5.0] * 10,
            "power_draw_kwh": [7.5] * 10,
            "error_code_count": [0] * 10,
            "telemetry_received": telemetry_received,
        }
    )

    feat = _claim_features("CLM-TEST", g, rated_temp=60, rated_pressure=550, rated_duty=4)

    assert feat["n_days_observed"] == 10
    assert feat["max_consecutive_missing_days"] == 3
    assert feat["telemetry_uptime_pct"] == 0.7
    assert feat["pct_days_over_rated_temp"] == 1.0
    assert feat["pct_days_over_rated_duty"] == 1.0
