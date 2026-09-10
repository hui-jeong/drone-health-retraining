from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.common.schemas import required_window_columns, validate_window_features
from src.feature_export.model_adapter import CompanyModelAdapter


def _window_id(row: pd.Series, strategy: str) -> str:
    start_idx = int(row["start_idx"])
    if strategy == "source_start":
        return f"{row['source_key']}::{start_idx:06d}"
    if strategy == "source_record_start":
        record_id = row.get("record_id", row.get("record_idx"))
        return f"{row['source_key']}::r{int(record_id):04d}::{start_idx:06d}"
    raise ValueError(f"Unknown window_id_strategy: {strategy}")


def _score_manifest(
    adapter: CompanyModelAdapter,
    manifest: pd.DataFrame,
    split: str | None,
) -> pd.DataFrame:
    """Score records.

    When exporting the complete dataset, score train/validation/test separately.
    The company evaluator scores the test split by itself. With CUDA AMP enabled,
    changing batch boundaries by scoring all 360 records in one DataLoader can
    introduce very small floating-point differences even though decisions are
    identical. Scoring each split independently reproduces the company's split
    evaluation path and makes the exported test scores numerically consistent
    with reports/test_window_predictions.csv.
    """
    if split is not None:
        selected = manifest[manifest["split"].astype(str) == str(split)].reset_index(drop=True)
        if selected.empty:
            raise RuntimeError(f"No records found for split={split}")
        frame = adapter.score_records(selected)
        # record_idx inside the company scorer is local to the selected split.
        # Keep globally stable traceability in the exported parquet.
        if "record_id" in frame.columns:
            frame["record_idx"] = frame["record_id"].astype(int)
        return frame

    frames: list[pd.DataFrame] = []
    preferred_order = ["train", "validation", "test"]
    available = [str(x) for x in manifest["split"].dropna().astype(str).unique().tolist()]
    split_order = [s for s in preferred_order if s in available] + [
        s for s in available if s not in preferred_order
    ]

    for split_name in split_order:
        selected = manifest[manifest["split"].astype(str) == split_name].reset_index(drop=True)
        if selected.empty:
            continue
        part = adapter.score_records(selected)
        if "record_id" in part.columns:
            part["record_idx"] = part["record_id"].astype(int)
        frames.append(part)

    if not frames:
        raise RuntimeError("No records were available for export")

    return pd.concat(frames, ignore_index=True)


def export_window_features(
    company_model_root: str | Path,
    company_config_path: str | Path,
    model_output_root: str | Path,
    output_path: str | Path,
    *,
    split: str | None = None,
    window_id_strategy: str = "source_record_start",
    latent_size: int = 32,
    compression: str = "snappy",
) -> tuple[Path, dict[str, Any]]:
    model_output_root = Path(model_output_root).resolve()
    manifest_path = model_output_root / "artifacts" / "prepared_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"prepared_manifest.csv not found: {manifest_path}. "
            "Run company run_all.py/prepare_dataset.py first."
        )

    manifest = pd.read_csv(manifest_path, dtype={"run_id": str})

    adapter = CompanyModelAdapter(
        company_model_root,
        company_config_path,
        model_output_root,
    )
    frame = _score_manifest(adapter, manifest, split)

    if latent_size != int(adapter.model.latent_size):
        raise ValueError(
            f"Configured latent_size={latent_size}, "
            f"but company checkpoint uses latent_size={adapter.model.latent_size}"
        )

    frame["window_id"] = frame.apply(
        lambda row: _window_id(row, window_id_strategy),
        axis=1,
    )
    frame = frame.rename(
        columns={
            "start_time": "window_start",
            "end_time": "window_end",
            "score": "hybrid_score",
            "effective_threshold": "context_threshold",
            "prediction": "final_prediction",
        }
    )

    required = required_window_columns(latent_size)
    trace_columns = [
        c
        for c in (
            "record_id",
            "record_idx",
            "start_idx",
            "split",
            "anomaly_severity",
            "table_path",
        )
        if c in frame.columns and c not in required
    ]
    frame = frame[required + trace_columns].copy()

    validation = validate_window_features(frame, latent_size)
    if not validation.valid:
        raise RuntimeError(
            "window feature schema validation failed: "
            + "; ".join(validation.errors)
        )

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(
        output_path,
        index=False,
        compression=compression,
    )
    return output_path, validation.to_dict()
