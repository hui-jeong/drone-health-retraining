"""A fixed PAPE reference held out before any detector fitting.

Upstream model calibration is model development, not PAPE validation.
The historical 007 pilot loader is deliberately not reused here.
"""
from pathlib import Path
import hashlib
import json

import pandas as pd

from .io import config_hash

ROLES = ("model_fit", "model_calibration", "pape_source", "pape_validation")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def role_mapping(config):
    roles = config["fixed_reference"]["roles"]
    if set(roles) != set(ROLES):
        raise ValueError("All four fixed-reference roles are required")
    mapping = {}
    for role, ids in roles.items():
        if not ids and not (role == "model_calibration" and config["fixed_reference"].get("development_mode") == "train_only"):
            raise ValueError(f"Empty role: {role}")
        for value in ids:
            run = str(value).zfill(3)
            if run in mapping or run in {"008", "009"}:
                raise ValueError(f"Overlapping or forbidden run: {run}")
            mapping[run] = role
    if set(mapping) != {f"{i:03d}" for i in range(8)} or mapping["007"] != "pape_validation":
        raise ValueError("000-006 must cover development/reference; 007 is validation only")
    if config["fixed_reference"].get("development_mode") == "train_only":
        if roles["model_calibration"] or config["fixed_reference"]["expected_source_keys"]["model_calibration"] != 0:
            raise ValueError("Train-only protocol must have no held-out model calibration")
    return mapping


def assign_roles(frame, config, *, complete=True):
    result = frame.copy()
    result["run_id"] = result.run_id.astype(str).str.zfill(3)
    mapping = role_mapping(config)
    if not set(result.run_id) <= set(mapping):
        raise ValueError("Unexpected run or official Test in fixed-reference data")
    result["role"] = result.run_id.map(mapping)
    if result.groupby("source_key").role.nunique().max() != 1:
        raise ValueError("One source_key appears in multiple roles")
    if complete:
        for role in ROLES:
            part = result[result.role.eq(role)]
            expected = config["fixed_reference"]["expected_source_keys"][role]
            if part.source_key.nunique() != expected:
                raise ValueError(f"Incomplete {role}: expected {expected} source_keys")
            classes = part.groupby("source_key").data_class.agg(set)
            if not classes.map(lambda v: v == {"normal", "abnormal"}).all():
                raise ValueError(f"Missing normal/derived pair in {role}")
    return result


def assert_model_manifest(manifest, config):
    assigned = assign_roles(manifest, config, complete=False)
    expected_split = {"model_fit": "train", "model_calibration": "validation"}
    if not set(assigned.role) <= set(expected_split):
        raise ValueError("PAPE Source/007 entered model development")
    if not assigned.split.eq(assigned.role.map(expected_split)).all():
        raise ValueError("Company train/validation mapping differs from approved roles")
    for role in expected_split:
        if assigned[assigned.role.eq(role)].source_key.nunique() != config["fixed_reference"]["expected_source_keys"][role]:
            raise ValueError(f"Incomplete model-development role: {role}")
    return assigned


def model_artifact_hashes(model_root):
    root = Path(model_root)
    required = ["checkpoints/best_lstm_ae.pt", "artifacts/scaler.joblib",
                "artifacts/feature_columns.json", "artifacts/residual_classifier.joblib",
                "artifacts/hybrid_policy.json", "artifacts/threshold.json",
                "artifacts/prepared_manifest.csv", "company_config.yaml"]
    return {name: sha256(root / name) for name in required}


def load_fixed_features(config):
    """Reject old/altered exports, checkpoints and splits before PAPE fitting."""
    if config['fixed_reference']['protocol'] == 'frozen_export_source_v1':
        from src.pape.frozen_handoff import load
        return load(config)
    spec = config["fixed_reference"]
    root = Path(spec["export_root"])
    manifest = json.loads((root / "baseline_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != spec["protocol"] or not manifest.get("fresh_training"):
        raise ValueError("Fresh fixed-reference training provenance is required")
    if manifest["role_config_hash"] != config_hash(spec["roles"]):
        raise ValueError("Role configuration has changed")
    if manifest["model_artifact_hashes"] != model_artifact_hashes(spec["model_root"]):
        raise ValueError("Frozen detector artifacts changed after export")
    model_manifest = pd.read_csv(Path(spec["model_root"]) / "artifacts/prepared_manifest.csv", dtype={"run_id": str})
    assert_model_manifest(model_manifest, config)
    frames = {}
    for role in ("pape_source", "pape_validation"):
        path = root / role / "window_features.parquet"
        if sha256(path) != manifest["exports"][role]["sha256"]:
            raise ValueError(f"Feature export hash mismatch: {role}")
        frame = pd.read_parquet(path)
        checked = assign_roles(frame, config, complete=False)
        if set(checked.role) != {role}:
            raise ValueError(f"Wrong role in {role} features")
        expected = manifest["exports"][role]
        if (sorted(frame.source_key.unique()) != expected["source_keys"]
                or sorted(frame.record_id.unique().tolist()) != expected["record_ids"]
                or len(frame) != expected["window_count"]):
            raise ValueError(f"Incomplete export: {role}")
        frames[role] = frame
    if set(frames["pape_source"].source_key) & set(frames["pape_validation"].source_key):
        raise ValueError("Source and Target overlap")
    return frames["pape_source"], frames["pape_validation"], manifest
