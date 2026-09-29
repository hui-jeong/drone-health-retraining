"""Recompute saved results without retraining the detector or PAPE estimators."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from src.common.metrics import actual_confusion, expected_confusion
from src.pape.contracts import PAPE_ESTIMATE_COLUMNS
from src.pape.fixed_reference import load_fixed_features, sha256
from src.pape.io import config_hash, load_config, utc_now, write_json


def verify(config):
    source, target, provenance = load_fixed_features(config)
    output = Path(config["fixed_reference"]["evaluation_root"])
    summary = json.loads((output / "pape_summary.json").read_text(encoding="utf-8"))
    csv = pd.read_csv(output / "pape_estimates.csv")
    if csv.columns.tolist() != PAPE_ESTIMATE_COLUMNS:
        raise ValueError("Public CSV contract changed")
    rows = summary["records"]
    expected_count = len(config["experiment"]["seeds"]) * len(config["experiment"]["input_names"]) * len(config["experiment"]["weightings"]) * target.record_id.nunique()
    if len(rows) != len(csv) or len(rows) != expected_count:
        raise ValueError("Trial coverage incomplete")
    table = pd.DataFrame(rows)
    if table.duplicated(["seed", "input_name", "weighting", "target_record_id"]).any():
        raise ValueError("Duplicate trials")
    keys = set(source.source_key)
    source_ids = set(source.window_id)
    target_groups = {int(key): group for key, group in target.groupby("record_id")}
    checked = 0
    for index, row in enumerate(rows):
        if row["study_config_hash"] != config_hash(config) or row["target_labels_used"]:
            raise ValueError("Configuration or label-blocking declaration mismatch")
        group = target_groups[row["target_record_id"]].sort_values("window_start")
        if (set(row["source_keys"]) != keys
                or set(row["source_run_ids"]) != set(config["fixed_reference"]["roles"]["pape_source"])
                or row["run_id"] not in config["fixed_reference"]["roles"]["pape_validation"]
                or row["run_id"] != group.run_id.iloc[0]):
            raise ValueError("A trial used another Source/Target partition")
        directory = output / "trials" / str(row["seed"]) / row["input_name"] / row["weighting"]
        logs = {}
        for kind, artifact in row["trial_artifacts"].items():
            path = directory / artifact["file"]
            if sha256(path) != artifact["sha256"]:
                raise ValueError("Trial log hash changed")
            logs[kind] = pd.read_parquet(path)
        weights = logs["source_weights"]
        probabilities = logs["target_probabilities"]
        if len(weights) != len(source) or set(weights.window_id) != source_ids or set(weights.source_key) != keys:
            raise ValueError("Source weight log contains missing or foreign windows")
        if probabilities.window_id.tolist() != group.window_id.tolist() or not np.array_equal(probabilities.final_prediction, group.final_prediction):
            raise ValueError("Target predictions or alignment changed")
        params = [config["loss"]["lambda_fn"], config["loss"]["lambda_fp"], config["metrics"]["zero_division"]]
        expected = expected_confusion(probabilities.calibrated_probability, probabilities.final_prediction, *params)
        actual = actual_confusion(group.y_true, group.final_prediction, *params)
        for prefix, recomputed in [("estimated", expected), ("actual", actual)]:
            for name, value in recomputed.items():
                if not np.isclose(row[f"{prefix}_{name}"], value, rtol=1e-10, atol=1e-10):
                    raise ValueError(f"Confusion metric mismatch at row {index}: {prefix}_{name}")
        for name in PAPE_ESTIMATE_COLUMNS:
            if isinstance(row[name], str):
                if row[name] != csv.iloc[index][name]:
                    raise ValueError("CSV/JSON string mismatch")
            elif row[name] is None:
                if not pd.isna(csv.iloc[index][name]):
                    raise ValueError("CSV/JSON null mismatch")
            elif not np.isclose(row[name], csv.iloc[index][name], rtol=1e-10, atol=1e-10):
                raise ValueError("CSV/JSON numeric mismatch")
        checked += 1
    report = {"status": "PASS", "executed_at": utc_now(), "trials_checked": checked,
        "unique_targets": len(target_groups), "public_schema_columns": len(PAPE_ESTIMATE_COLUMNS),
        "source_target_fixed": True, "saved_confusion_metrics_recomputed": True,
        "model_and_input_hashes_valid": config['fixed_reference']['protocol'] != 'frozen_export_source_v1',
        "frozen_feature_export_hashes_valid": True,
        "checkpoint_hash_verified": config['fixed_reference']['protocol'] != 'frozen_export_source_v1',
        "trial_artifact_hashes_valid": True,
        "test_used": False, "experiment_rerun": False,
        "not_a_performance_or_production_approval": True}
    write_json(output / "verification_report.json", report)
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", required=True)
    args = parser.parse_args()
    verify(load_config(*args.config))
