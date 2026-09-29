"""Column names copied from the priority document, sections 5.1 and 5.2.

No additional columns are required by the shared CSV contract. Diagnostics and
experiment provenance belong in pape_summary.json and module-private logs.
"""
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.common.schemas import (
    latent_columns, required_window_columns,
    validate_window_features as validate_shared_window_features,
)

LATENT_COLUMNS = latent_columns()
WINDOW_FEATURE_COLUMNS = required_window_columns()
PAPE_ESTIMATE_COLUMNS = [
    "target_chunk_id", "input_variant", "estimated_fn", "estimated_fp",
    "estimated_recall", "estimated_f1", "estimated_loss", "actual_fn", "actual_fp",
    "actual_recall", "actual_f1", "dre_auc", "ess_ratio", "max_weight",
]
LABEL_COLUMNS = ["y_true", "anomaly_fraction", "data_class", "anomaly_type"]
SCORE_COLUMNS = ["ae_score", "ae_percentile_score", "residual_probability", "hybrid_score"]
ALLOWED_MODEL_FEATURES = set(SCORE_COLUMNS + LATENT_COLUMNS)
INFERENCE_COLUMNS = [c for c in WINDOW_FEATURE_COLUMNS if c not in LABEL_COLUMNS]
STRING_COLUMNS = [
    "window_id", "source_key", "run_id", "scenario_id", "phase", "data_class", "anomaly_type",
]


@dataclass
class PapeResult:
    """A single Target Run estimate, without access to Target labels."""

    estimates: dict[str, Any]
    diagnostics: dict[str, Any]
    source_weights: pd.DataFrame
    target_probabilities: pd.DataFrame


def model_features(columns: list[str]) -> list[str]:
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("Model feature list must be nonempty and unique")
    forbidden = set(columns) - ALLOWED_MODEL_FEATURES
    if forbidden:
        raise ValueError(f"Forbidden model features (label/ID leakage): {sorted(forbidden)}")
    return columns


def hide_target_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Select an allowlist before any validation or fitting; ignore extra columns."""
    missing = set(INFERENCE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing Target inference columns: {sorted(missing)}")
    return df.loc[:, INFERENCE_COLUMNS].copy()


def validate_window_features(df: pd.DataFrame, *, labeled: bool = True) -> None:
    # The shared model-adapter contract remains unchanged. PAPE adds local
    # checks and validates unlabeled Target views without fabricating labels.
    if labeled:
        shared_result = validate_shared_window_features(df)
        if not shared_result.valid:
            raise ValueError("Shared Window Feature validation: " + "; ".join(shared_result.errors))
    required = WINDOW_FEATURE_COLUMNS if labeled else INFERENCE_COLUMNS
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing Window Feature columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Window Feature data must not be empty")
    if df[required].isna().any().any():
        raise ValueError("Required Window Feature columns contain missing values")
    for col in set(STRING_COLUMNS) & set(required):
        if not df[col].map(lambda x: isinstance(x, str) and bool(x)).all():
            raise ValueError(f"{col} must contain nonempty strings")
    if not df.run_id.str.fullmatch(r"\d{3}").all():
        raise ValueError("run_id must preserve the three-digit string format, e.g. '000'")
    if not df.scenario_id.str.fullmatch(r"[ABCD][12]").all():
        raise ValueError("scenario_id must be A1 through D2")
    if df.window_id.duplicated().any():
        raise ValueError("Duplicate window_id; coordinate the ID rule with the feature exporter")
    numeric = [c for c in required if c not in STRING_COLUMNS]
    if any(not pd.api.types.is_numeric_dtype(df[c]) for c in numeric):
        raise ValueError("Scores, times, labels and latent columns must be numeric")
    if not np.isfinite(df[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Window Feature values must be finite")
    if (df.window_start < 0).any() or (df.window_end <= df.window_start).any():
        raise ValueError("Invalid Window time interval")
    if (df.ae_score < 0).any():
        raise ValueError("ae_score (MSE) must be nonnegative")
    for col in ["ae_percentile_score", "residual_probability"]:
        if not df[col].between(0, 1).all():
            raise ValueError(f"{col} must be within [0, 1]")
    for col in ["raw_prediction", "final_prediction"] + (["y_true"] if labeled else []):
        if not pd.api.types.is_integer_dtype(df[col]) or not df[col].isin([0, 1]).all():
            raise ValueError(f"{col} must be binary integers")
    if labeled:
        if not df.data_class.isin(["normal", "abnormal"]).all():
            raise ValueError("data_class must be normal or abnormal")
        if not df.anomaly_fraction.between(0, 1).all():
            raise ValueError("anomaly_fraction must be within [0, 1]")
        if not np.array_equal((df.anomaly_fraction >= 0.20).astype(int), df.y_true):
            raise ValueError("y_true must equal int(anomaly_fraction >= 0.20)")
