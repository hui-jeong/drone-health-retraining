"""Portable 007 handoff reader. Oracle inputs require an explicit opt-in.

The shared pape_estimates.csv remains an evaluation file and contains actual_*.
Only the feature allowlist below is exposed to a downstream policy developer.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import PAPE_ESTIMATE_COLUMNS
from src.common.metrics import confusion_metrics

VERSION = "version_1.0"
FEATURES = ["estimated_tp", "estimated_fp", "estimated_fn", "estimated_tn",
            "estimated_recall", "estimated_f1", "estimated_loss",
            "dre_auc", "ess_ratio", "max_weight"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_release(directory, *, allow_oracle=False):
    """Verify file integrity, paired counts and the unchanged 14-column contract."""
    root = Path(directory).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("release_version") != VERSION:
        raise ValueError("Unsupported release version")
    kind = manifest.get("artifact_kind")
    if kind not in {"pape_estimates", "lstm_ae_true_data"}:
        raise ValueError("Unknown handoff artifact kind")
    if kind == "lstm_ae_true_data" and not allow_oracle:
        raise ValueError("Oracle uses Target labels; set allow_oracle=True for offline comparison only")
    required = {"pape_estimates.csv", "pape_summary.json", "chunk_index.csv"}
    if not required <= set(manifest.get("files", {})):
        raise ValueError("Incomplete release manifest")
    for name, digest in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256(path) != digest:
            raise ValueError("Changed or invalid release file: " + name)
    summary = json.loads((root / "pape_summary.json").read_text(encoding="utf-8"))
    oracle = kind == "lstm_ae_true_data"
    if (summary.get("artifact_kind") != kind or summary.get("release_version") != VERSION
            or summary.get("target_labels_used_for_inputs") is not oracle
            or summary.get("official_test_used") is not False
            or summary.get("chronology_verified") is not False):
        raise ValueError("Unsafe or inconsistent release provenance")
    frame = pd.read_csv(root / "pape_estimates.csv")
    if list(frame.columns) != PAPE_ESTIMATE_COLUMNS:
        raise ValueError("Shared PAPE CSV schema changed")
    rows = summary["records"]
    if len(rows) != len(frame) or frame.target_chunk_id.duplicated().any():
        raise ValueError("Ambiguous Chunk mapping")
    index_frame = pd.read_csv(root / "chunk_index.csv")
    if index_frame.target_chunk_id.tolist() != frame.target_chunk_id.tolist():
        raise ValueError("Chunk index does not match the handoff")
    costs = summary["loss_definition"]
    for index, row in enumerate(rows):
        if row["csv_row_index"] != index or row["run_id"] != "007":
            raise ValueError("Invalid row identity or Target split")
        for key in PAPE_ESTIMATE_COLUMNS:
            observed, expected = frame.iloc[index][key], row[key]
            ok = pd.isna(observed) if expected is None else (
                observed == expected if isinstance(expected, str) else
                np.isclose(observed, expected, rtol=1e-9, atol=1e-9))
            if not ok:
                raise ValueError("CSV/JSON mismatch: " + key)
        for prefix in ["estimated_", "actual_"]:
            values = np.array([row[prefix+k] for k in ["tp", "fp", "fn", "tn"]], dtype=float)
            if not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), row["n_target"]):
                raise ValueError("Invalid confusion counts")
            loss = costs["lambda_fn"]*row[prefix+"fn"] + costs["lambda_fp"]*row[prefix+"fp"]
            if not np.isclose(loss, row[prefix+"loss"]):
                raise ValueError("Loss differs from declared costs")
            recomputed = confusion_metrics(*values, costs["lambda_fn"], costs["lambda_fp"],
                                          summary.get("metrics", {}).get("zero_division", 0.0))
            if any(not np.isclose(row[prefix+k], recomputed[k]) for k in ["recall", "f1"]):
                raise ValueError("Ratios differ from confusion counts")
        if not np.isclose(row["estimated_tp"]+row["estimated_fp"], row["actual_tp"]+row["actual_fp"]):
            raise ValueError("Fixed detector predictions changed")
        if oracle:
            if any(row["estimated_"+k] != row["actual_"+k] for k in ["tp", "fp", "fn", "tn", "recall", "f1", "loss"]):
                raise ValueError("Oracle does not match actual detector results")
        elif row.get("target_labels_used") is not False:
            raise ValueError("Target-label isolation not attested")
    return summary


def load_release_inputs(directory, *, allow_review=False, allow_oracle=False):
    """Return current-loss features only; order is NOT verified flight chronology.

    No actual_*, anomaly type, data class, future loss, Gain or action is exposed.
    allow_review permits research consumption of estimates with diagnostic flags.
    allow_oracle permits separately marked, label-derived offline comparator data.
    """
    summary = read_release(directory, allow_oracle=allow_oracle)
    oracle = summary["artifact_kind"] == "lstm_ae_true_data"
    result = []
    for row in summary["records"]:
        if row["readiness"]["status"] == "requires_review" and not allow_review:
            raise ValueError("PAPE estimate requires review; inspect warnings or use allow_review=True")
        result.append({
            "target_chunk_id": row["target_chunk_id"], "input_variant": row["input_variant"],
            "features": {key: row[key] for key in FEATURES},
            "provenance": {
                **{key: row[key] for key in ["seed", "source_key", "run_id", "scenario_id", "n_target", "model_version",
                    "config_hash", "study_config_hash", "executed_at", "reference_pape_config_hash"] if key in row},
                **{key: summary[key] for key in ["code_revision", "original_study_config_hash", "original_executed_at",
                    "created_at", "loss_definition", "metrics", "bundle_id", "selection_status"] if key in summary},
            },
            "readiness": row["readiness"], "warnings": row["warnings"],
            "label_derived_oracle": oracle, "chronology_verified": False,
            "release_version": VERSION,
        })
    return result
