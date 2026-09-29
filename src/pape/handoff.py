"""Read-only model-adapter handoff audit and protected Reference loading.

Shared schemas.py is imported without modification. Test labels and scores are
never loaded: inventory uses metadata columns, then Reference uses Parquet filters.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.common.io import sha256_file
from .contracts import WINDOW_FEATURE_COLUMNS, validate_window_features
from .io import config_hash, utc_now


def validate_window_geometry(frame, config, *, check_runs):
    settings = config["window"]
    convention = settings.get("timestamp_convention", "end_exclusive")
    if convention == "last_sample":
        duration = (settings["timesteps"] - 1) / settings["sampling_hz"]
    elif convention == "end_exclusive":
        duration = settings["duration_seconds"]
    else:
        raise ValueError(f"Unknown Window timestamp convention: {convention}")
    tolerance = settings.get("time_tolerance", 1e-6)
    if not np.allclose(frame.window_end - frame.window_start, duration, atol=tolerance, rtol=0):
        raise ValueError("Window duration must match the configured Hybrid Window length and timestamp convention")
    if not check_runs:
        return
    # Full handoff views have record_id; the estimator receives a single hidden Run.
    groups = frame.groupby("record_id", sort=True) if "record_id" in frame else [(None, frame)]
    stride = settings["stride_seconds"]
    for _, run in groups:
        ordered = run.sort_values("window_start")
        steps = np.diff(ordered.window_start.to_numpy())
        if len(steps):
            regular = np.isclose(steps, stride, atol=tolerance, rtol=0)
            if settings.get("allow_end_aligned_tail", False) and 0 < steps[-1] < stride:
                regular[-1] = True
            if not regular.all():
                raise ValueError("Target must be one contiguous Run with the configured Window stride (optional short final stride only)")
        if "start_idx" in ordered:
            indexes = ordered.start_idx.to_numpy()
            if (indexes < 0).any() or not np.isclose(indexes, np.rint(indexes)).all():
                raise ValueError("start_idx must contain nonnegative sample indexes")
            relative_seconds = ordered.window_start.to_numpy() - ordered.window_start.iloc[0]
            if not np.allclose(relative_seconds, (indexes - indexes[0]) / settings["sampling_hz"],
                               atol=tolerance, rtol=0):
                raise ValueError("Window timestamps disagree with start_idx and sampling rate")


def inspect_handoff(window_path, manifest_path, config):
    window_path, manifest_path = Path(window_path), Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    parquet = pq.ParquetFile(window_path)
    names = parquet.schema_arrow.names
    required = set(WINDOW_FEATURE_COLUMNS) | {"record_id", "start_idx", "split"}
    missing = required - set(names)
    if missing:
        raise ValueError(f"Handoff columns missing: {sorted(missing)}")
    metadata_columns = ["window_id", "source_key", "record_id", "run_id", "scenario_id", "split", "data_class"]
    metadata = pd.read_parquet(window_path, columns=metadata_columns)
    errors = []
    if metadata.isna().any().any():
        errors.append("Missing inventory metadata")
    if metadata.window_id.duplicated().any():
        errors.append("Duplicate window_id")
    if metadata.groupby("source_key")["split"].nunique().gt(1).any():
        errors.append("A source_key spans baseline splits")
    record_identity = metadata.groupby("record_id")[["source_key", "run_id", "scenario_id", "split", "data_class"]].nunique()
    if record_identity.gt(1).any().any():
        errors.append("A record_id identifies multiple Runs")
    expected_ids = {
        "train": config["split"]["baseline_train_run_ids"],
        "validation": config["split"]["baseline_validation_run_ids"],
        "test": config["split"]["official_test_run_ids"],
    }
    if not metadata["split"].isin(expected_ids).all():
        errors.append("Unexpected baseline split name")
    for split, run_ids in expected_ids.items():
        if not metadata.loc[metadata["split"] == split, "run_id"].isin(run_ids).all():
            errors.append(f"run_id does not agree with {split} membership")
    declared = manifest.get("window_feature_export", {})
    observed = {
        "row_count": len(metadata), "column_count": len(names),
        "source_key_count": int(metadata.source_key.nunique()),
        "record_count": int(metadata.record_id.nunique()),
        "split_window_counts": {str(k): int(v) for k, v in metadata["split"].value_counts().items()},
        "data_class_window_counts": {str(k): int(v) for k, v in metadata.data_class.value_counts().items()},
    }
    for key, value in observed.items():
        if key not in declared or declared[key] != value:
            errors.append(f"Manifest mismatch: window_feature_export.{key}")
    if manifest.get("model_version") != config["model_version"]:
        errors.append("Baseline model_version does not match PAPE configuration")
    export_schema = declared.get("schema_validation", {})
    if not export_schema.get("valid", False):
        errors.append("Upstream manifest does not attest a valid export schema")
    baseline_differences = manifest.get("baseline_difference_documentation", {})
    if errors:
        raise ValueError("Handoff audit failed: " + "; ".join(errors))
    return {
        "status": "metadata_pass", "executed_at": utc_now(), "config_hash": config_hash(config),
        "model_version": manifest["model_version"], "baseline_seed": manifest.get("seed"),
        "baseline_timestamp_utc": manifest.get("integration_timestamp_utc"),
        "window_features_sha256": sha256_file(window_path),
        "baseline_manifest_sha256": sha256_file(manifest_path),
        "observed": observed, "extra_columns": sorted(set(names) - set(WINDOW_FEATURE_COLUMNS)),
        "shared_required_columns": WINDOW_FEATURE_COLUMNS,
        "window_id_note": declared.get("window_id_note"),
        "id_handling": "preserve_exported_window_id_verbatim; use record_id for a single Target Run",
        "test_labels_or_scores_loaded": False,
        "baseline_difference_documentation": baseline_differences,
        "baseline_metrics_source": "upstream_manifest_only; no test evaluation rerun by PAPE",
        "upstream_run_all_executed_flag": manifest.get("run_all_executed"),
        "upstream_execution_time_source": manifest.get("execution_time_source"),
    }


def load_reference_pool(window_path, config):
    settings = config["pape_data"]
    if settings["reference_split"] != "validation":
        raise ValueError("This exploratory protocol allows baseline Validation only")
    if not settings.get("allow_reused_validation", False):
        raise ValueError("Reference protocol requires the user's choice about reused baseline Validation")
    run_ids = settings["reference_run_ids"]
    if set(run_ids) & set(config["split"]["official_test_run_ids"]):
        raise ValueError("Reference run IDs must be in baseline Validation, not Train or Test (Official Test overlap)")
    if not run_ids or not set(run_ids) <= set(config["split"]["baseline_validation_run_ids"]):
        raise ValueError("Reference run IDs must be in baseline Validation, not Train or Test")
    # Exclude unrelated absolute table_path and anomaly_severity metadata from
    # PAPE memory. Target label proxies are further removed by the estimator.
    columns = [*WINDOW_FEATURE_COLUMNS, "record_id", "record_idx", "start_idx", "split"]
    names = pq.ParquetFile(window_path).schema_arrow.names
    columns = [c for c in columns if c in names]
    frame = pd.read_parquet(window_path, columns=columns,
                            filters=[("split", "=", settings["reference_split"]), ("run_id", "in", run_ids)])
    validate_window_features(frame)
    validate_window_geometry(frame, config, check_runs=True)
    if frame.record_id.isna().any() or frame.groupby("record_id").source_key.nunique().gt(1).any():
        raise ValueError("Each Target record_id must identify exactly one source_key")
    return frame


def iter_handoff_target_runs(frame):
    for record_id, run in frame.groupby("record_id", sort=True):
        if run[["source_key", "run_id", "scenario_id"]].drop_duplicates().shape[0] != 1:
            raise ValueError("record_id mixes different Runs")
        yield f"{run.source_key.iloc[0]}::record_{int(record_id):04d}", run.sort_values("window_start").copy()
