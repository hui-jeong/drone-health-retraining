from copy import deepcopy
import numpy as np
import pandas as pd
import pytest

from scripts.dummy_data import iter_target_runs, make_dummy_dataset
from src.pape.contracts import LABEL_COLUMNS, hide_target_labels
from src.pape.density_ratio import estimate_density_ratio
from src.pape.estimator import estimate_target_performance


@pytest.mark.parametrize("weighting", ["uniform", "dre"])
def test_target_labels_and_label_proxies_cannot_affect_estimate(config, frames, weighting):
    source, target = frames
    config["pape"]["weighting"] = weighting
    before = target.copy(deep=True)
    result = estimate_target_performance(source, target, config)
    hidden = estimate_target_performance(source, hide_target_labels(target), config)
    poisoned = target.copy()
    for col in LABEL_COLUMNS:
        poisoned[col] = "DO_NOT_READ_TARGET_LABEL_OR_PROXY"
    changed = estimate_target_performance(source, poisoned, config)
    assert result.estimates == hidden.estimates == changed.estimates
    pd.testing.assert_frame_equal(target, before)
    n = len(target)
    assert sum(result.estimates[f"estimated_{k}"] for k in ["tp", "fp", "fn", "tn"]) == pytest.approx(n)
    assert result.estimates["estimated_tp"] + result.estimates["estimated_fp"] == pytest.approx(target.final_prediction.sum())
    assert result.target_probabilities.calibrated_probability.between(0, 1).all()
    assert not result.diagnostics["target_labels_used"]
    assert 0 < result.estimates["ess_ratio"] <= 1 + 1e-12


def test_source_target_sibling_leak_is_rejected(config, frames):
    source, target = frames
    leaked = target.copy()
    leaked["source_key"] = source.source_key.iloc[0]
    with pytest.raises(ValueError, match="source_key leakage"):
        estimate_target_performance(source, leaked, config)


@pytest.mark.parametrize("leaked_column", ["y_true", "anomaly_fraction", "data_class", "anomaly_type", "source_key", "final_prediction"])
def test_no_label_or_id_features_in_dre(config, frames, leaked_column):
    config["pape"]["variants"]["hybrid_score"]["dre_features"] = [leaked_column]
    with pytest.raises(ValueError, match="Forbidden model features"):
        estimate_target_performance(*frames, config)


def test_calibrator_feature_allowlist(config, frames):
    config["pape"]["weighting"] = "uniform"
    config["pape"]["variants"]["hybrid_score"]["calibration_features"] = ["anomaly_fraction"]
    with pytest.raises(ValueError, match="Forbidden model features"):
        estimate_target_performance(*frames, config)


def test_one_class_source_fails_explicitly(config, frames):
    source, target = frames
    with pytest.raises(ValueError, match="both normal and anomalous"):
        estimate_target_performance(source[source.y_true == 0], target, config)


def test_empty_target_is_rejected(config, frames):
    with pytest.raises(ValueError, match="must not be empty"):
        estimate_target_performance(frames[0], frames[1].iloc[:0], config)


def test_mixed_run_target_rejected(config, frames):
    target = frames[1].copy()
    target.loc[target.index[0], "run_id"] = "006"
    with pytest.raises(ValueError, match="one Run"):
        estimate_target_performance(frames[0], target, config)


def test_wrong_window_length_rejected(config, frames):
    target = frames[1].copy()
    target["window_end"] += 1
    with pytest.raises(ValueError, match="Window duration"):
        estimate_target_performance(frames[0], target, config)


def test_run_gap_rejected(config, frames):
    target = frames[1].iloc[[0, 2, 3, 4, 5]].copy()
    with pytest.raises(ValueError, match="contiguous Run"):
        estimate_target_performance(frames[0], target, config)


def test_missing_loss_choice_rejected_before_fit(config, frames):
    config["loss"]["lambda_fn"] = None
    with pytest.raises(ValueError, match="lambda_fn"):
        estimate_target_performance(*frames, config)


def test_official_test_never_reaches_estimator(config, frames):
    target = frames[1].copy()
    target["run_id"] = "009"
    with pytest.raises(ValueError, match="Official Test"):
        estimate_target_performance(frames[0], target, config)


def test_clipping_audit_and_uniform_baseline(config, frames):
    config["dre"].update(clip_min=1.0, clip_max=1.0)
    clipped = estimate_target_performance(*frames, config)
    config["pape"]["weighting"] = "uniform"
    baseline = estimate_target_performance(*frames, config)
    assert clipped.estimates["max_weight"] == 1
    assert clipped.estimates["ess_ratio"] == pytest.approx(1)
    assert clipped.source_weights.clipped_weight.eq(1).all()
    for metric in ["fn", "fp", "tp", "tn", "recall", "f1", "loss"]:
        assert clipped.estimates[f"estimated_{metric}"] == baseline.estimates[f"estimated_{metric}"]
    assert baseline.estimates["dre_auc"] is None


def test_dre_fold_prior_with_constant_domain_classifier(config, frames, monkeypatch):
    class PriorClassifier:
        def __init__(self, **kwargs):
            pass

        def fit(self, x, domains):
            self.p = domains.mean()

        def predict_proba(self, x):
            return np.tile([1 - self.p, self.p], (len(x), 1))

    monkeypatch.setattr("src.pape.density_ratio.LGBMClassifier", PriorClassifier)
    result = estimate_density_ratio(*frames, ["hybrid_score"], config)
    assert np.allclose(result.table.raw_weight, 1)
    assert result.diagnostics["dre_auc"] == pytest.approx(0.5)
    for fold in result.diagnostics["folds"]:
        assert fold["prior_source_over_target"] == pytest.approx(fold["n_source_train"] / fold["n_target_train"])


def test_concept_shift_changes_only_hidden_evaluation_labels(config):
    covariate, _ = make_dummy_dataset(config, config["seed"], "covariate_shift")
    concept, _ = make_dummy_dataset(config, config["seed"], "concept_shift")
    pd.testing.assert_frame_equal(covariate["source"], concept["source"])
    for (_, original), (_, changed) in zip(iter_target_runs(covariate["validation"]),
                                         iter_target_runs(concept["validation"])):
        pd.testing.assert_frame_equal(hide_target_labels(original), hide_target_labels(changed))
        if original.data_class.iloc[0] == "abnormal":
            assert np.array_equal(original.y_true.to_numpy(), 1 - changed.y_true.to_numpy())


def test_unsupported_target_is_diagnosed(config):
    data, _ = make_dummy_dataset(config, config["seed"], "support_shift")
    _, target = next(iter_target_runs(data["validation"]))
    config["pape"]["input_variant"] = "hybrid_score_dre_combined"
    result = estimate_target_performance(data["source"], target, config)
    assert "target_outside_source_range" in result.diagnostics["warnings"]


def test_default_chunk_id_distinguishes_siblings_without_extra_columns(config):
    datasets, _ = make_dummy_dataset(config, config["seed"])
    key = datasets["selection"].source_key.iloc[0]
    siblings = [run for _, run in iter_target_runs(datasets["selection"]) if run.source_key.iloc[0] == key]
    config["pape"]["weighting"] = "uniform"
    ids = [estimate_target_performance(datasets["source"], run, config).estimates["target_chunk_id"] for run in siblings]
    assert len(ids) == 2 and len(set(ids)) == 2
    reordered = estimate_target_performance(datasets["source"], siblings[0].iloc[::-1], config)
    assert reordered.estimates["target_chunk_id"] == ids[0]
    config["target_chunk_id"] = "caller-file-identity"
    explicit = estimate_target_performance(datasets["source"], siblings[0], config)
    assert explicit.estimates["target_chunk_id"] == "caller-file-identity"


@pytest.mark.parametrize("identity", ["", 123, " "])
def test_invalid_explicit_chunk_id_rejected(config, frames, identity):
    config["target_chunk_id"] = identity
    with pytest.raises(ValueError, match="target_chunk_id"):
        estimate_target_performance(*frames, config)
