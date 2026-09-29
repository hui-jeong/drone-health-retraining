"""Public PAPE interface. Target labels never enter density estimation or calibration."""
import numpy as np

from src.pape.io import config_hash, utc_now
from src.common.metrics import expected_confusion
from src.pape.contracts import PapeResult, hide_target_labels, validate_window_features
from .calibrator import calibrated_target_probabilities
from .density_ratio import estimate_density_ratio
from .split import assert_disjoint_source_keys, assert_not_official_test
from .handoff import validate_window_geometry
from .diagnostics import diagnose_reference_support, diagnostic_readiness
from .randomness import resolved_seeds


def estimate_target_performance(source_df, target_df, config) -> PapeResult:
    loss = config["loss"]
    costs = [loss["lambda_fn"], loss["lambda_fp"]]
    if any(v is None for v in costs) or not np.isfinite(costs).all() or any(v < 0 for v in costs):
        raise ValueError("Configure finite nonnegative lambda_fn and lambda_fp before estimating loss")
    assert_not_official_test(target_df, config)
    target = hide_target_labels(target_df).reset_index(drop=True)
    source = source_df.reset_index(drop=True).copy()
    validate_window_features(source)
    validate_window_features(target, labeled=False)
    assert_disjoint_source_keys(source, target)
    assert_not_official_test(source, config)
    assert_not_official_test(target, config)
    if target[["source_key", "run_id", "scenario_id"]].drop_duplicates().shape[0] != 1:
        raise ValueError("Target must be one Run, not multiple source_keys or runs")
    if target.window_start.duplicated().any():
        raise ValueError("Target mixes overlapping file variants; pass one Run file per Chunk")
    validate_window_geometry(source, config, check_runs=False)
    validate_window_geometry(target, config, check_runs=True)
    # The shared source_key deliberately groups normal/synthetic siblings.
    # Hash globally unique Window IDs, never labels or label-proxy columns.
    key = str(target.source_key.iloc[0])
    target_chunk_id = config.get("target_chunk_id")
    if target_chunk_id is None:
        identity = config_hash({"window_ids": sorted(target.window_id.tolist())})
        target_chunk_id = f"{key}::run_{target.run_id.iloc[0]}::windows_{identity}"
    if not isinstance(target_chunk_id, str) or not target_chunk_id.strip():
        raise ValueError("target_chunk_id must be a nonempty string")
    variant = config["pape"]["input_variant"]
    definition = config["pape"]["variants"][variant]
    weighting = config["pape"]["weighting"]
    if weighting == "dre":
        ratio = estimate_density_ratio(source, target, definition["dre_features"], config)
        weights, table, diagnostics = ratio.weights, ratio.table, ratio.diagnostics
    elif weighting == "uniform":
        weights = np.ones(len(source))
        table = source[["window_id", "source_key", "run_id"]].copy()
        table["raw_weight"] = weights
        table["clipped_weight"] = weights
        table["was_clipped"] = False
        diagnostics = {"dre_auc": None, "ess": float(len(source)), "ess_ratio": 1.0,
                       "raw_ess": float(len(source)), "max_weight": 1.0, "max_raw_weight": 1.0,
                       "clip_fraction": 0.0, "weight_method": "uniform", "folds": []}
    else:
        raise ValueError("pape.weighting must be 'uniform' or 'dre'")
    probabilities, calibration = calibrated_target_probabilities(
        source, target, definition["calibration_features"], weights, config)
    diagnostics.update(diagnose_reference_support(source, target, definition["calibration_features"]))
    metrics = expected_confusion(probabilities, target.final_prediction,
                                 loss["lambda_fn"], loss["lambda_fp"], config["metrics"]["zero_division"])
    warnings = []
    limits = config["diagnostics"]
    for field, threshold, flag in [
        ("clip_fraction", limits["max_clip_fraction"], "excessive_weight_clipping"),
        ("target_outside_source_range_fraction", limits["max_outside_range_fraction"], "target_outside_source_range"),
    ]:
        if diagnostics.get(field, 0) > threshold:
            warnings.append(flag)
    if diagnostics["ess_ratio"] < limits["min_ess_ratio"]:
        warnings.append("low_effective_sample_size")
    if len(target) < limits["min_target_windows"]:
        warnings.append("small_target_sample")
    if metrics["tp"] + metrics["fn"] < limits["min_expected_positives"]:
        warnings.append("too_few_expected_positives_for_stable_recall")
    if not diagnostics["target_scenario_in_source"]:
        warnings.append("target_scenario_absent_from_source")
    if diagnostics["calibration_outside_source_range_fraction"] > limits["max_outside_range_fraction"]:
        warnings.append("calibration_target_outside_source_range")
    diagnostics.update(calibration)
    diagnostics.update({
        "seed": config["seed"], "config_hash": config_hash(config),
        "resolved_seeds": resolved_seeds(config),
        "model_version": config["model_version"], "source_key": key,
        "source_keys": sorted(source.source_key.unique()), "executed_at": utc_now(),
        "n_source": len(source), "n_target": len(target), "dre_features": definition["dre_features"],
        "input_name": variant, "weighting": weighting, "warnings": warnings,
        "target_labels_used": False, "prediction_column": "final_prediction",
        "run_id": str(target.run_id.iloc[0]), "source_run_ids": sorted(source.run_id.unique()),
        "loss_definition": {k: loss[k] for k in ["status", "lambda_fn", "lambda_fp"]},
        "readiness": diagnostic_readiness(warnings),
        "metric_definition": "plugin_metrics_of_expected_confusion_counts",
    })
    estimates = {"target_chunk_id": target_chunk_id, "input_variant": f"{variant}__{weighting}",
                 **{f"estimated_{k}": v for k, v in metrics.items()},
                 **{k: diagnostics[k] for k in ["dre_auc", "ess_ratio", "max_weight"]}}
    probabilities_table = target[["window_id", "source_key", "run_id", "final_prediction"]].copy()
    probabilities_table["calibrated_probability"] = probabilities
    return PapeResult(estimates, diagnostics, table, probabilities_table)
