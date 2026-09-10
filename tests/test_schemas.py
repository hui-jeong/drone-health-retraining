from __future__ import annotations

import pandas as pd

from src.common.schemas import required_window_columns, validate_window_features


def _valid_frame() -> pd.DataFrame:
    row = {
        "window_id": "s::r0001::000000",
        "source_key": "s",
        "run_id": "000",
        "scenario_id": "A1",
        "phase": "HOVER",
        "data_class": "normal",
        "anomaly_type": "none",
        "window_start": 0.0,
        "window_end": 4.9,
        "anomaly_fraction": 0.0,
        "y_true": 0,
        "ae_score": 0.01,
        "ae_percentile_score": 0.2,
        "residual_probability": 0.1,
        "hybrid_score": 0.15,
        "context_threshold": 0.5,
        "raw_prediction": 0,
        "final_prediction": 0,
    }
    for i in range(32):
        row[f"latent_{i:02d}"] = 0.0
    return pd.DataFrame([row])


def test_schema_valid() -> None:
    result = validate_window_features(_valid_frame())
    assert result.valid
    assert result.duplicate_window_ids == 0


def test_duplicate_detected() -> None:
    df = pd.concat([_valid_frame(), _valid_frame()], ignore_index=True)
    result = validate_window_features(df)
    assert not result.valid
    assert result.duplicate_window_ids == 1
