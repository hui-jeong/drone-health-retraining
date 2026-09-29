from copy import deepcopy
from dataclasses import asdict

import pytest

from src.pape.diagnostics import diagnostic_readiness
from src.pape.estimator import estimate_target_performance
from src.pape.integration import build_pape_input, ESTIMATE_FIELDS


def test_integration_output_is_allowlisted_and_label_invariant(config, frames):
    result = estimate_target_performance(*frames, config)
    before = build_pape_input(result, config, allow_review=True)
    for field in ["actual_fn", "actual_fp", "actual_recall", "actual_f1", "y_true",
                  "anomaly_fraction", "data_class", "anomaly_type", "fn_absolute_error",
                  "realized_gain", "future_keep_loss", "gate_pass"]:
        result.estimates[field] = "SECRET"
        result.diagnostics[field] = "SECRET"
    after = build_pape_input(result, config, allow_review=True)
    assert asdict(before) == asdict(after)
    assert set(after.features) == set(ESTIMATE_FIELDS)
    assert "SECRET" not in str(asdict(after))
    assert not after.readiness["production_approved"]
    assert after.readiness["policy_action"] is None


def test_review_requires_explicit_diagnostic_opt_in(config, frames):
    result = estimate_target_performance(*frames, config)
    result.diagnostics["warnings"].append("target_scenario_absent_from_source")
    result.diagnostics["readiness"] = diagnostic_readiness(result.diagnostics["warnings"])
    with pytest.raises(ValueError, match="requires review"):
        build_pape_input(result, config)
    assert build_pape_input(result, config, allow_review=True).readiness["status"] == "requires_review"


@pytest.mark.parametrize("field,value,match", [
    ("run_id", "008", "Official Test"),
    ("run_id", "009", "Official Test"),
    ("run_id", 7, "three-digit"),
    ("source_run_ids", ["009"], "Official Test"),
    ("target_labels_used", True, "Target-label"),
    ("config_hash", "bad", "config hash"),
    ("model_version", "wrong", "model_version"),
    ("n_target", 999999, "sum to n_target"),
    ("readiness", {}, "readiness mismatch"),
])
def test_invalid_provenance_and_test_are_rejected(config, frames, field, value, match):
    result = estimate_target_performance(*frames, config)
    result.diagnostics[field] = value
    with pytest.raises(ValueError, match=match):
        build_pape_input(result, config, allow_review=True)


@pytest.mark.parametrize("field,value,match", [
    ("estimated_loss", -1, "Invalid PAPE"),
    ("estimated_loss", 1e20, "configured costs"),
    ("estimated_recall", 2, "outside"),
    ("estimated_fn", float("nan"), "Invalid PAPE"),
])
def test_invalid_estimates_rejected(config, frames, field, value, match):
    result = estimate_target_performance(*frames, config)
    result.estimates[field] = value
    with pytest.raises(ValueError, match=match):
        build_pape_input(result, config, allow_review=True)


def test_uniform_auc_remains_null(config, frames):
    config["pape"]["weighting"] = "uniform"
    result = estimate_target_performance(*frames, config)
    assert build_pape_input(result, config, allow_review=True).features["dre_auc"] is None
