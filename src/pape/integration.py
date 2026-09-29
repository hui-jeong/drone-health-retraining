"""PAPE-private inference boundary; not a new shared Gain/RCCDA schema.

The mandated CSV includes evaluation labels. Never feed that whole table to a
Gain model. This adapter exposes only an allowlist of current estimates and
non-label provenance. It does not implement predict_gain or decide_retrain.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.io import sha256_file
from .contracts import PapeResult, PAPE_ESTIMATE_COLUMNS
from .diagnostics import diagnostic_readiness
from .io import config_hash, load_config


ESTIMATE_FIELDS = ["estimated_tp", "estimated_fp", "estimated_fn", "estimated_tn",
                   "estimated_recall", "estimated_f1", "estimated_loss",
                   "dre_auc", "ess_ratio", "max_weight"]
PROVENANCE_FIELDS = ["seed", "config_hash", "model_version", "executed_at",
                     "source_key", "source_keys", "run_id", "source_run_ids",
                     "n_source", "n_target", "loss_definition"]


@dataclass(frozen=True)
class PapeInput:
    """Reviewable current-loss input; no actual_*, future loss, Gain or action."""
    target_chunk_id: str
    input_variant: str
    features: dict
    provenance: dict
    readiness: dict


def build_pape_input(result: PapeResult, config, *, allow_review=False) -> PapeInput:
    estimates, diag = result.estimates, result.diagnostics
    if diag.get("target_labels_used") is not False:
        raise ValueError("Target-label isolation is not attested")
    if diag.get("config_hash") != config_hash(config):
        raise ValueError("PAPE config hash mismatch")
    if diag.get("model_version") != config["model_version"]:
        raise ValueError("PAPE model_version mismatch")
    missing = set(PROVENANCE_FIELDS) - set(diag)
    if missing:
        raise ValueError(f"Missing PAPE provenance: {sorted(missing)}")
    if diag["source_key"] in diag["source_keys"]:
        raise ValueError("Source/Target source_key leakage")
    forbidden = set(config["split"]["official_test_run_ids"])
    if ({diag["run_id"]} | set(diag["source_run_ids"])) & forbidden:
        raise ValueError("Official Test is prohibited at the integration boundary")
    if not isinstance(diag["run_id"], str) or len(diag["run_id"]) != 3 or not diag["run_id"].isdigit():
        raise ValueError("run_id must preserve the three-digit string format")
    features = {key: estimates[key] for key in ESTIMATE_FIELDS}
    for key, value in features.items():
        if value is None and key == "dre_auc" and diag["weighting"] == "uniform":
            continue
        if value is None or not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid PAPE estimate: {key}")
    for key in ["estimated_recall", "estimated_f1", "ess_ratio", "dre_auc"]:
        if features[key] is not None and features[key] > 1 + 1e-12:
            raise ValueError(f"PAPE ratio outside [0, 1]: {key}")
    counts = [features[f"estimated_{k}"] for k in ["tp", "fp", "fn", "tn"]]
    if not np.isclose(sum(counts), diag["n_target"], rtol=1e-9, atol=1e-9):
        raise ValueError("Expected confusion does not sum to n_target")
    cost = {k: config["loss"][k] for k in ["status", "lambda_fn", "lambda_fp"]}
    if diag["loss_definition"] != cost:
        raise ValueError("PAPE loss definition mismatch")
    expected_loss = cost["lambda_fn"] * features["estimated_fn"] + cost["lambda_fp"] * features["estimated_fp"]
    if not np.isclose(features["estimated_loss"], expected_loss, rtol=1e-9, atol=1e-9):
        raise ValueError("PAPE loss differs from configured costs")
    readiness = diagnostic_readiness(diag["warnings"])
    if readiness != diag.get("readiness"):
        raise ValueError("PAPE warning/readiness mismatch")
    if readiness["status"] == "requires_review" and not allow_review:
        raise ValueError("PAPE estimate requires review: " + ", ".join(readiness["review_reasons"]))
    return PapeInput(
        target_chunk_id=estimates["target_chunk_id"], input_variant=estimates["input_variant"],
        features=features, provenance={key: diag[key] for key in PROVENANCE_FIELDS}, readiness=readiness,
    )


def read_verified_handoff(output_dir):
    """Check the two mandatory handoff files against computational verification."""
    output = Path(output_dir)
    verification = json.loads((output / "verification.json").read_text(encoding="utf-8"))
    if verification.get("status") != "pass":
        raise ValueError("PAPE computational verification has not passed")
    for name in ["pape_estimates.csv", "pape_summary.json", "config/resolved.yaml", "run_status.json"]:
        expected = verification.get("handoff_file_sha256", {}).get(name)
        if expected is None or expected != sha256_file(output / name):
            raise ValueError(f"Unverified or changed handoff file: {name}")
    summary = json.loads((output / "pape_summary.json").read_text(encoding="utf-8"))
    config = load_config(output / "config/resolved.yaml")
    status = json.loads((output / "run_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise ValueError("PAPE run is incomplete")
    if summary.get("official_test_used") is not False or summary.get("target_labels_used_during_estimation") is not False:
        raise ValueError("Unsafe PAPE experiment provenance")
    if config_hash(config) != summary["config_hash"] or verification["config_hash"] != summary["config_hash"]:
        raise ValueError("Handoff config hashes differ")
    if summary["model_version"] != config["model_version"]:
        raise ValueError("Handoff model_version differs")
    csv = pd.read_csv(output / "pape_estimates.csv")
    if list(csv.columns) != PAPE_ESTIMATE_COLUMNS:
        raise ValueError("PAPE CSV schema differs from the shared contract")
    records = summary["records"]
    if len(csv) != len(records) or len(csv) != verification["verified_estimates"]:
        raise ValueError("PAPE handoff row count mismatch")
    seen = set()
    for i, row in enumerate(records):
        key = (row["seed"], row["target_chunk_id"], row["input_variant"])
        if key in seen or row["csv_row_index"] != i:
            raise ValueError("Ambiguous PAPE trial identity or CSV mapping")
        seen.add(key)
        for col in PAPE_ESTIMATE_COLUMNS:
            actual, expected = csv.iloc[i][col], row[col]
            if expected is None:
                same = pd.isna(actual)
            elif isinstance(expected, str):
                same = actual == expected
            else:
                same = np.isclose(actual, expected, rtol=1e-9, atol=1e-9)
            if not same:
                raise ValueError(f"PAPE CSV/JSON mismatch: {col}")
    return summary


def load_pape_inputs(output_dir, *, seed, input_variant, split="validation", allow_review=False):
    """Select one experiment identity explicitly; never concatenate seeds as time.

    Return order is the experiment's file order, NOT an operational chronology.
    The caller must supply real chronology before any Gain/RCCDA streaming use.
    """
    if split != "validation":
        raise ValueError("Integration checks use PAPE validation rows only")
    summary = read_verified_handoff(output_dir)
    rows = [r for r in summary["records"] if r["seed"] == seed and
            r["input_variant"] == input_variant and r["split"] == split]
    if not rows:
        raise ValueError("No matching PAPE validation rows")
    result = []
    for row in rows:
        estimate = {key: row[key] for key in ["target_chunk_id", "input_variant", *ESTIMATE_FIELDS]}
        pape = PapeResult(estimate, row, pd.DataFrame(), pd.DataFrame())
        result.append(build_pape_input(pape, row["effective_config"], allow_review=allow_review))
    return result
