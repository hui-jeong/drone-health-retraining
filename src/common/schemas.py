from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import pandas as pd


BASE_WINDOW_COLUMNS = [
    "window_id",
    "source_key",
    "run_id",
    "scenario_id",
    "phase",
    "data_class",
    "anomaly_type",
    "window_start",
    "window_end",
    "anomaly_fraction",
    "y_true",
    "ae_score",
    "ae_percentile_score",
    "residual_probability",
    "hybrid_score",
    "context_threshold",
    "raw_prediction",
    "final_prediction",
]


def latent_columns(latent_size: int = 32) -> list[str]:
    return [f"latent_{i:02d}" for i in range(latent_size)]


def required_window_columns(latent_size: int = 32) -> list[str]:
    return BASE_WINDOW_COLUMNS + latent_columns(latent_size)


@dataclass
class SchemaValidationResult:
    row_count: int
    duplicate_window_ids: int
    missing_required_values: int
    latent_size: int
    valid: bool
    errors: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _all_in(values: Iterable, allowed: set) -> bool:
    return set(pd.Series(values).dropna().unique()).issubset(allowed)


def validate_window_features(df: pd.DataFrame, latent_size: int = 32) -> SchemaValidationResult:
    errors: list[str] = []
    required = required_window_columns(latent_size)
    missing_columns = [c for c in required if c not in df.columns]
    if missing_columns:
        errors.append(f"missing columns: {missing_columns}")
        return SchemaValidationResult(len(df), -1, -1, latent_size, False, errors)

    duplicate_count = int(df["window_id"].duplicated().sum())
    if duplicate_count:
        errors.append(f"duplicate window_id: {duplicate_count}")

    missing_values = int(df[required].isna().sum().sum())
    if missing_values:
        errors.append(f"missing required values: {missing_values}")

    if not _all_in(df["y_true"], {0, 1}):
        errors.append("y_true must be binary")
    if not _all_in(df["raw_prediction"], {0, 1}):
        errors.append("raw_prediction must be binary")
    if not _all_in(df["final_prediction"], {0, 1}):
        errors.append("final_prediction must be binary")

    for column in ("anomaly_fraction", "ae_percentile_score", "residual_probability", "hybrid_score"):
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            errors.append(f"{column} contains non-finite values")
        elif ((values < -1e-9) | (values > 1.0 + 1e-9)).any():
            errors.append(f"{column} must be in [0,1]")

    ae_score = pd.to_numeric(df["ae_score"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(ae_score).all() or (ae_score < -1e-12).any():
        errors.append("ae_score must be finite and non-negative")

    thresholds = pd.to_numeric(df["context_threshold"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(thresholds).all() or (thresholds <= 0).any():
        errors.append("context_threshold must be finite and positive")

    starts = pd.to_numeric(df["window_start"], errors="coerce").to_numpy(dtype=np.float64)
    ends = pd.to_numeric(df["window_end"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(starts).all() or not np.isfinite(ends).all() or (ends < starts).any():
        errors.append("window time range is invalid")

    return SchemaValidationResult(
        row_count=int(len(df)),
        duplicate_window_ids=duplicate_count,
        missing_required_values=missing_values,
        latent_size=latent_size,
        valid=not errors,
        errors=errors,
    )
