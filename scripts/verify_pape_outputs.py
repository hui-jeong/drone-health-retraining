"""Recompute real PAPE handoff results from saved logs and the read-only input.

This does not select inputs or read official Test labels/scores. CSV names are
checked against an independent transcription of sections 5.1/5.2 of the plan.
"""
import argparse
from collections import defaultdict
import inspect
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.common.io import sha256_file
from src.common.schemas import required_window_columns
from src.pape.contracts import PAPE_ESTIMATE_COLUMNS, LABEL_COLUMNS, PapeResult
from src.pape.estimator import estimate_target_performance
from src.pape.handoff import load_reference_pool
from src.pape.io import config_hash, load_config, utc_now, write_json

DOCUMENT_WINDOW_COLUMNS = [
    "window_id", "source_key", "run_id", "scenario_id", "phase", "data_class",
    "anomaly_type", "window_start", "window_end", "anomaly_fraction", "y_true",
    "ae_score", "ae_percentile_score", "residual_probability", "hybrid_score",
    "context_threshold", "raw_prediction", "final_prediction",
] + [f"latent_{i:02d}" for i in range(32)]
DOCUMENT_PAPE_COLUMNS = [
    "target_chunk_id", "input_variant", "estimated_fn", "estimated_fp",
    "estimated_recall", "estimated_f1", "estimated_loss", "actual_fn", "actual_fp",
    "actual_recall", "actual_f1", "dre_auc", "ess_ratio", "max_weight",
]
IDENTITY = ["seed", "target_chunk_id", "input_variant"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(observed, expected, message):
    require(np.allclose(observed, expected, atol=1e-8, rtol=1e-8), message)


def check_document_contract(project_root=PROJECT_ROOT):
    require(required_window_columns() == DOCUMENT_WINDOW_COLUMNS, "Window column names/order differ from plan")
    require(PAPE_ESTIMATE_COLUMNS == DOCUMENT_PAPE_COLUMNS, "PAPE column names/order differ from plan")
    signature = inspect.signature(estimate_target_performance)
    require(list(signature.parameters) == ["source_df", "target_df", "config"], "Public parameter names differ from plan")
    require(signature.return_annotation is PapeResult, "Public return type differs from plan")
    for name in ["src/pape/split.py", "src/pape/density_ratio.py", "src/pape/calibrator.py",
                 "src/pape/estimator.py", "config/pape.yaml", "scripts/run_pape_pilot.py"]:
        require((Path(project_root) / name).is_file(), f"Required file missing: {name}")


def metrics(counts, config):
    tp, fp, fn, tn = counts
    zero = config["metrics"]["zero_division"]
    return dict(zip(["tp", "fp", "fn", "tn"], counts)) | {
        "recall": tp / (tp + fn) if tp + fn else zero,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else zero,
        "loss": config["loss"]["lambda_fn"] * fn + config["loss"]["lambda_fp"] * fp,
    }


def scan_log(path, records, pool, *, weights):
    """Bound RAM while validating every logged Window and trial, including duplicates."""
    index = pd.Index(pool.window_id)
    seen = {key: np.zeros(len(pool), dtype=np.uint16) for key in records}
    stats = defaultdict(lambda: np.zeros(8, dtype=float))
    count = 0
    for chunk in pd.read_csv(path, chunksize=100000, dtype={"run_id": str}):
        require(not set(LABEL_COLUMNS) & set(chunk.columns), "Labels leaked into an inference log")
        for key, group in chunk.groupby(IDENTITY, sort=False):
            require(key in records, f"Unknown trial in log: {key}")
            record = records[key]
            positions = index.get_indexer(group.window_id)
            require((positions >= 0).all(), "Logged window_id absent from supplied input")
            original = pool.iloc[positions]
            for col in ["source_key", "run_id"]:
                require(np.array_equal(group[col], original[col]), f"Input {col} changed in log")
            for col in ["config_hash", "model_version", "executed_at", "case", "split"]:
                require(group[col].eq(record[col]).all(), f"Log provenance differs: {col}")
            np.add.at(seen[key], positions, 1)
            current = stats[key]
            current[0] += len(group)
            if weights:
                raw = group.raw_weight.to_numpy()
                clipped = group.clipped_weight.to_numpy()
                require(np.isfinite(raw).all() and np.isfinite(clipped).all() and (clipped > 0).all(), "Invalid weights")
                if record["weighting"] == "dre":
                    cfg = record["effective_config"]["dre"]
                    close(clipped, np.clip(raw, cfg["clip_min"], cfg["clip_max"]), "Clipping differs from YAML")
                    p = np.clip(group.domain_probability, cfg["probability_epsilon"], 1-cfg["probability_epsilon"])
                    close(raw, group.prior_source_over_target * p / (1-p), "DRE odds/prior formula mismatch")
                else:
                    close(raw, np.ones(len(group)), "Uniform raw weights are not one")
                    close(clipped, np.ones(len(group)), "Uniform weights are not one")
                    require(group[["fold", "domain_probability", "prior_source_over_target"]].isna().all().all(),
                            "Uniform logs must not fabricate DRE diagnostics")
                require(np.array_equal(group.was_clipped, raw != clipped), "was_clipped differs from weights")
                current[1] += clipped.sum()
                current[2] += np.square(clipped).sum()
                current[3] = max(current[3], clipped.max())
                current[4] += (raw != clipped).sum()
                current[5] += raw.sum()
                current[6] += np.square(raw).sum()
                current[7] = max(current[7], raw.max())
            else:
                require(np.array_equal(group.final_prediction, original.final_prediction), "Supplied final_prediction changed")
                p = group.calibrated_probability.to_numpy()
                require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid calibrated probabilities")
                positive = group.final_prediction.to_numpy() == 1
                current[1:5] += [p[positive].sum(), (1-p[positive]).sum(), p[~positive].sum(), (1-p[~positive]).sum()]
            count += len(group)
    for key, record in records.items():
        mask = (pool.source_key.isin(record["source_keys"]) if weights
                else pool.record_id.eq(record["target_record_id"]))
        require(np.array_equal(seen[key], mask.to_numpy(dtype=np.uint16)), "Missing, duplicated, or wrong-split logged Windows")
    return stats, count


def verify_real_output(output_dir, *, check_label_invariance=False):
    check_document_contract()
    output = Path(output_dir)
    summary = json.loads((output / "pape_summary.json").read_text(encoding="utf-8"))
    config = load_config(output / "config/resolved.yaml")
    require(summary["data_origin"] == "model_adapter_handoff", "Not a real handoff experiment")
    require(json.loads((output / "run_status.json").read_text())["status"] == "complete", "Pilot incomplete")
    require(not summary["official_test_used"] and not summary["target_labels_used_during_estimation"], "Unsafe experiment flags")
    require(summary["config_hash"] == config_hash(config), "Saved config hash mismatch")
    for relative, digest in summary.get("code_file_sha256", {}).items():
        require(sha256_file(PROJECT_ROOT / relative) == digest, f"Experiment code changed since run: {relative}")
    data = config["pape_data"]
    for field in ["window_features", "baseline_manifest"]:
        require(sha256_file(PROJECT_ROOT / data[field]) == summary["handoff_validation"][field+"_sha256"], "Input file changed")
    pool = load_reference_pool(PROJECT_ROOT / data["window_features"], config).reset_index(drop=True)
    csv = pd.read_csv(output / "pape_estimates.csv")
    require(csv.columns.tolist() == DOCUMENT_PAPE_COLUMNS, "CSV schema differs from document")
    require(len(csv) == len(summary["records"]), "CSV/JSON row count mismatch")
    records = {}
    for i, row in enumerate(summary["records"]):
        key = tuple(row[col] for col in IDENTITY)
        require(key not in records, "Duplicate trial identity")
        records[key] = row
        require(row["csv_row_index"] == i, "CSV row index mismatch")
        require(row["config_hash"] == config_hash(row["effective_config"]), "Trial config hash mismatch")
        for col in DOCUMENT_PAPE_COLUMNS:
            if row[col] is None:
                require(pd.isna(csv.iloc[i][col]), f"CSV/JSON null mismatch: {col}")
            elif isinstance(row[col], str):
                require(csv.iloc[i][col] == row[col], f"CSV/JSON text mismatch: {col}")
            else:
                close(csv.iloc[i][col], row[col], f"CSV/JSON numeric mismatch: {col}")
        require(row["source_key"] not in row["source_keys"], "Source/Target overlap")
        target = pool[pool.record_id == row["target_record_id"]]
        require(len(target) == row["n_target"] and set(target.source_key) == {row["source_key"]}, "Wrong Target Run")
        y, pred = target.y_true.to_numpy(), target.final_prediction.to_numpy()
        counts = [int(((y == truth) & (pred == prediction)).sum())
                  for truth, prediction in [(1, 1), (0, 1), (1, 0), (0, 0)]]
        actual = metrics(counts, config)
        for name, value in actual.items():
            close(row["actual_"+name], value, "Actual confusion/metric mismatch: "+name)
            close(row[name+"_absolute_error"], abs(row["estimated_"+name]-value), "MAE component mismatch")
        for cost in row["loss_sensitivity"]:
            for prefix in ["estimated", "actual"]:
                close(cost[prefix+"_loss"], cost["lambda_fn"]*row[prefix+"_fn"] + cost["lambda_fp"]*row[prefix+"_fp"], "Sensitivity loss mismatch")
            close(cost["loss_absolute_error"], abs(cost["estimated_loss"]-cost["actual_loss"]), "Sensitivity error mismatch")
    w_stats, weight_count = scan_log(output / "source_weights.csv.gz", records, pool, weights=True)
    p_stats, probability_count = scan_log(output / "target_probabilities.csv.gz", records, pool, weights=False)
    for key, row in records.items():
        n, total, squares, maximum, clipped, raw_total, raw_squares, raw_max = w_stats[key]
        close(n, row["n_source"], "Source count mismatch")
        close([total**2/squares, total**2/squares/n, maximum, clipped/n, raw_total**2/raw_squares, raw_max],
              [row[c] for c in ["ess", "ess_ratio", "max_weight", "clip_fraction", "raw_ess", "max_raw_weight"]], "Weight statistics mismatch")
        close(p_stats[key][0], row["n_target"], "Probability count mismatch")
        expected = metrics(p_stats[key][1:5], config)
        close(sum(p_stats[key][1:5]), row["n_target"], "Expected counts do not sum to Target size")
        for name, value in expected.items():
            close(value, row["estimated_"+name], "Expected confusion/metric mismatch: "+name)
    table = pd.DataFrame(summary["records"])
    for _, pair in table.groupby(["seed", "target_chunk_id", "input_name"]):
        require(set(pair.weighting) == {"uniform", "dre"} and len(pair) == 2, "Unpaired baseline/DRE comparison")
        require(pair.actual_loss.nunique() == 1 and len({tuple(v) for v in pair.source_keys}) == 1, "Comparison splits differ")
    for group_columns, field in [(["case", "split", "input_name", "weighting"], "metrics"),
                                  (["case", "split", "seed", "input_name", "weighting"], "metrics_by_seed")]:
        for entry in summary[field]:
            group = table
            for col in group_columns:
                group = group[group[col] == entry[col]]
            require(len(group) == entry["n_target_chunks"], "Aggregate group count mismatch")
            for name in ["fn", "recall", "f1", "loss"]:
                close(group[name+"_absolute_error"].mean(), entry[name+"_mae"], "Aggregate MAE mismatch")
    for seed in config["experiment"]["seeds"]:
        assignments = pd.DataFrame([v for v in summary["split_manifest"] if v["seed"] == seed])
        require(not assignments.source_key.duplicated().any(), "A source_key spans PAPE roles")
        for role in ["source", "selection", "validation"]:
            require(len(assignments[assignments["split"] == role]) > 0, "Empty PAPE split")
        winner = summary["selection"]["validation_candidate_by_seed"][str(seed)]
        require(winner == summary["selection"]["seed_winners"][str(seed)], "Validation candidate used pooled selection")
        require(set(table[(table.seed == seed) & (table.split == "validation")].input_name)
                == {config["pape"]["input_variant"], winner}, "Wrong per-seed validation inputs")
        if config["experiment"].get("selection_protocol") == "grouped_cv":
            held = set(assignments.loc[assignments.split.eq("validation"), "source_key"])
            development = set(assignments.source_key) - held
            folds = [f for f in summary["selection_cv_manifest"] if f["seed"] == seed]
            require(len(folds) == config["experiment"]["selection_cv"]["n_splits"], "Wrong inner CV fold count")
            require({f["fold"] for f in folds} == set(range(len(folds))), "Duplicate or missing inner fold")
            all_targets = []
            for fold in folds:
                source, target = set(fold["source_keys"]), set(fold["target_keys"])
                require(not source & target and source | target == development, "Inner CV Source/Target partition mismatch")
                require(set(fold["outer_validation_keys"]) == held and not (source | target) & held,
                        "Outer validation leaked into inner CV")
                all_targets.extend(fold["target_keys"])
            require(len(all_targets) == len(set(all_targets)) and set(all_targets) == development,
                    "Inner CV must evaluate each development source_key exactly once")
            for row in table[table.seed.eq(seed)].to_dict("records"):
                if row["split"] == "selection":
                    fold = next(f for f in folds if f["fold"] == row["selection_fold"])
                    require(set(row["source_keys"]) == set(fold["source_keys"]) and row["source_key"] in fold["target_keys"],
                            "Actual selection trial violates inner CV assignment")
                else:
                    require(set(row["source_keys"]) == development and row["source_key"] in held,
                            "Validation refit must use development only")
    invariance, invariant_variants = [], []
    if check_label_invariance:
        representatives = {}
        for row in records.values():
            representatives.setdefault((row["input_name"], row["weighting"]), row)
        for (input_name, method), row in representatives.items():
            source = pool[pool.source_key.isin(row["source_keys"])]
            target = pool[pool.record_id == row["target_record_id"]].copy()
            for col in LABEL_COLUMNS:
                target[col] = "DO_NOT_READ_TARGET_LABEL"
            result = estimate_target_performance(source, target, row["effective_config"])
            for name in ["tp", "fp", "fn", "tn", "recall", "f1", "loss"]:
                close(result.estimates["estimated_"+name], row["estimated_"+name], "Target labels affected real-data estimates")
            if "readiness" in row:
                require(result.diagnostics["readiness"] == row["readiness"], "Target labels affected readiness")
            invariant_variants.append({"input_name": input_name, "weighting": method})
        invariance = [method for method in ["uniform", "dre"] if any(r["weighting"] == method for r in invariant_variants)]
    report = {
        "status": "pass", "executed_at": utc_now(), "data_origin": summary["data_origin"],
        "verified_estimates": len(records), "verified_source_weight_rows": weight_count,
        "verified_probability_rows": probability_count, "config_hash": summary["config_hash"],
        "handoff_file_sha256": {name: sha256_file(output / name) for name in
            ["pape_estimates.csv", "pape_summary.json", "config/resolved.yaml", "run_status.json"]},
        "document_window_columns": DOCUMENT_WINDOW_COLUMNS, "document_pape_columns": DOCUMENT_PAPE_COLUMNS,
        "checks": ["document_column_names_order_and_interface", "csv_json_correspondence",
                   "input_file_hashes_unchanged", "experiment_code_hashes_unchanged", "config_hashes",
                   "logged_window_ids_preserved_exactly_once_per_trial", "provided_predictions_unchanged",
                   "source_target_disjointness", "paired_uniform_dre_splits", "per_seed_frozen_validation_candidates",
                   "expected_confusion_from_saved_probabilities", "actual_confusion_from_reference_labels",
                   "loss_and_mae_recomputed", "clipping_ess_and_weights_recomputed"],
        "real_target_label_invariance_methods": invariance,
        "real_target_label_invariance_variants": invariant_variants,
        "official_test_labels_or_scores_loaded": False,
        "upstream_id_rule_exception": summary["handoff_validation"]["window_id_note"],
        "interpretation": "Computational and contract verification only; not proof of statistical accuracy or final evaluation independence.",
    }
    write_json(output / "verification.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--check-label-invariance", action="store_true")
    args = parser.parse_args()
    report = verify_real_output(PROJECT_ROOT / args.output_dir, check_label_invariance=args.check_label_invariance)
    print(json.dumps({k: v for k, v in report.items() if k in ["status", "verified_estimates", "verified_source_weight_rows", "verified_probability_rows", "real_target_label_invariance_methods"]}, ensure_ascii=False))
