from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from scripts.dummy_data import make_dummy_dataset
from scripts.run_pape_real import _branch_info, run_real_pilot
from scripts.verify_pape_outputs import verify_real_output
from src.pape.handoff import inspect_handoff, load_reference_pool, validate_window_geometry
from src.pape.integration import load_pape_inputs
from scripts.validate_pape_integration import validate_pape_integration
from src.pape.split import split_source_keys


@pytest.fixture
def handoff(config, tmp_path):
    config["data_origin"] = "model_adapter_handoff"
    config["dummy"]["run_ids"] = ["007"]
    config["window"].update(timestamp_convention="last_sample", allow_end_aligned_tail=True)
    config["pape_data"] = {
        "reference_split": "validation", "reference_run_ids": ["007"],
        "allow_reused_validation": True, "reference_status": "test_fixture_only",
    }
    config["experiment"]["selection_case"] = "real_validation_reference"
    config["experiment"]["diagnostic_cases"] = []
    frames, _ = make_dummy_dataset(config, config["seed"])
    frame = pd.concat(frames.values(), ignore_index=True)
    frame["split"] = "validation"
    frame["record_id"] = frame.groupby(["source_key", "data_class"], sort=True).ngroup()
    first_start = frame.groupby("record_id").window_start.transform("min")
    frame["start_idx"] = np.rint((frame.window_start-first_start)*10).astype(int)
    frame["window_end"] -= 0.1
    held = frame.iloc[:2].copy()
    held["run_id"], held["split"], held["record_id"], held["source_key"] = "008", "test", 900, "held_test"
    held["window_id"] = ["held_test::0", "held_test::1"]
    held["y_true"] = 99  # Intentionally invalid: reference loading must not inspect Test labels.
    held["hybrid_score"] = np.nan
    frame = pd.concat([frame, held], ignore_index=True)
    manifest = {
        "model_version": config["model_version"], "seed": 42,
        "window_feature_export": {
            "row_count": len(frame), "column_count": len(frame.columns),
            "source_key_count": int(frame.source_key.nunique()), "record_count": int(frame.record_id.nunique()),
            "split_window_counts": frame["split"].value_counts().to_dict(),
            "data_class_window_counts": frame.data_class.value_counts().to_dict(),
            "schema_validation": {"valid": True},
        },
    }
    window_path, manifest_path = tmp_path / "window_features.parquet", tmp_path / "baseline_manifest.json"
    frame.to_parquet(window_path, index=False)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config["pape_data"].update(window_features=str(window_path), baseline_manifest=str(manifest_path))
    return config, window_path, manifest_path, frame


def test_reference_filter_does_not_load_test_labels_or_scores(handoff):
    config, windows, manifest, frame = handoff
    report = inspect_handoff(windows, manifest, config)
    pool = load_reference_pool(windows, config)
    assert report["status"] == "metadata_pass"
    assert not report["test_labels_or_scores_loaded"]
    assert set(pool.run_id) == {"007"}
    assert len(pool) == len(frame)-2
    assert not pool.y_true.eq(99).any()
    assert pool.hybrid_score.notna().all()
    assert pool.window_id.tolist() == frame[frame["split"] == "validation"].window_id.tolist()


def test_manifest_mismatch_fails(handoff):
    config, windows, manifest, _ = handoff
    value = json.loads(manifest.read_text())
    value["window_feature_export"]["row_count"] += 1
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Manifest mismatch"):
        inspect_handoff(windows, manifest, config)


def test_reference_requires_protocol_choice(handoff):
    config, windows, _, _ = handoff
    config["pape_data"]["allow_reused_validation"] = False
    with pytest.raises(ValueError, match="user's choice"):
        load_reference_pool(windows, config)


@pytest.mark.parametrize("run_id", ["006", "008", "009"])
def test_real_reference_excludes_train_and_test(handoff, run_id):
    config, windows, _, _ = handoff
    config["pape_data"]["reference_run_ids"] = [run_id]
    with pytest.raises(ValueError, match="not Train or Test"):
        load_reference_pool(windows, config)


def test_shared_source_key_cannot_span_baseline_splits(handoff):
    config, windows, manifest, frame = handoff
    frame.loc[frame["split"] == "test", "source_key"] = frame.source_key.iloc[0]
    frame.to_parquet(windows, index=False)
    with pytest.raises(ValueError, match="source_key spans baseline splits"):
        inspect_handoff(windows, manifest, config)


def test_inclusive_timestamp_and_last_short_stride(handoff):
    config, windows, _, _ = handoff
    pool = load_reference_pool(windows, config)
    run = pool[pool.record_id == pool.record_id.iloc[0]].copy()
    idx = run.index[-1]
    run.loc[idx, ["window_start", "window_end"]] -= 0.2
    run.loc[idx, "start_idx"] -= 2
    validate_window_geometry(run, config, check_runs=True)
    config["window"]["allow_end_aligned_tail"] = False
    with pytest.raises(ValueError, match="stride"):
        validate_window_geometry(run, config, check_runs=True)


def test_interior_short_stride_is_rejected(handoff):
    config, windows, _, _ = handoff
    pool = load_reference_pool(windows, config)
    run = pool[pool.record_id == pool.record_id.iloc[0]].copy()
    run.loc[run.index[3], ["window_start", "window_end"]] -= 0.2
    with pytest.raises(ValueError, match="stride"):
        validate_window_geometry(run, config, check_runs=True)


@pytest.mark.parametrize("branch", ["feature/pape", "main", ""])
def test_execution_records_branch_without_requiring_a_development_branch(monkeypatch, branch):
    def git_output(args, **kwargs):
        return {"branch": branch, "rev-parse": "abc123", "status": ""}[args[3]]
    monkeypatch.setattr("scripts.run_pape_real.subprocess.check_output", git_output)
    assert _branch_info("unused") == {"branch": branch, "head": "abc123", "working_tree_dirty": False}


def test_streamed_real_handoff_end_to_end(handoff, tmp_path, monkeypatch):
    config, _, _, _ = handoff
    config["experiment"]["seeds"] = [11, 23]
    config["pape"]["variants"] = {"hybrid_score": config["pape"]["variants"]["hybrid_score"]}
    monkeypatch.setattr("scripts.run_pape_real._branch_info", lambda *args: {"branch": "feature/pape", "head": "fixture", "working_tree_dirty": True})
    result, output = run_real_pilot(config, [], tmp_path / "result", project_root=tmp_path)
    rows = pd.DataFrame(result["records"])
    assert len(rows) > 0
    assert rows.target_record_id.ne(900).all()
    assert rows.data_origin.eq("model_adapter_handoff").all()
    assert result["selection"]["status"] == "exploratory_real_candidate_only"
    assert not result["reference_protocol"]["independent_final_evaluation"]
    assert not result["official_test_used"]
    weights = pd.read_csv(output / "source_weights.csv.gz")
    probabilities = pd.read_csv(output / "target_probabilities.csv.gz")
    assert len(weights) == rows.n_source.sum()
    assert len(probabilities) == rows.n_target.sum()
    expected = rows.groupby(["seed", "target_chunk_id", "input_variant"]).n_source.sum()
    observed = weights.groupby(["seed", "target_chunk_id", "input_variant"]).size()
    assert expected.to_dict() == observed.to_dict()
    assert json.loads((output / "run_status.json").read_text())["status"] == "complete"
    assert "탐색 Pilot" in (output / "pape_report.md").read_text(encoding="utf-8")
    verification = verify_real_output(output, check_label_invariance=True)
    assert verification["status"] == "pass"
    assert verification["real_target_label_invariance_methods"] == ["uniform", "dre"]
    assert {r["input_name"] for r in verification["real_target_label_invariance_variants"]} == {"hybrid_score"}
    integration = validate_pape_integration(output)
    assert integration["status"] == "pape_boundary_pass"
    assert not integration["actual_downstream_integration_executed"]
    assert not integration["policy_decisions_generated"]
    assert integration["checked_validation_rows"] == int(rows.split.eq("validation").sum())
    with pytest.raises(ValueError, match="No matching"):
        load_pape_inputs(output, seed=999, input_variant="hybrid_score__dre")
    with pytest.raises(ValueError, match="validation rows only"):
        load_pape_inputs(output, seed=11, input_variant="hybrid_score__dre", split="selection")
    # Prove that schema checking is independent of the writer's schema constant.
    csv_path = output / "pape_estimates.csv"
    pd.read_csv(csv_path).rename(columns={"estimated_fn": "wrong_fn"}).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="changed handoff"):
        load_pape_inputs(output, seed=11, input_variant="hybrid_score__dre", allow_review=True)
    with pytest.raises(ValueError, match="CSV schema"):
        verify_real_output(output)


def test_group_cv_selection_cannot_use_outer_validation_labels(handoff, tmp_path, monkeypatch):
    config, windows, _, frame = handoff
    config["experiment"].update(seeds=[11], selection_protocol="grouped_cv",
        selection_cv={"n_splits": 3, "allocation": "scenario_balanced_source_keys"})
    config["dre"]["source_fold_allocation"] = "scenario_balanced_source_keys"
    config["pape"]["variants"] = {k: config["pape"]["variants"][k] for k in ["hybrid_score", "residual_probability"]}
    monkeypatch.setattr("scripts.run_pape_real._branch_info", lambda *args: {"branch": "feature/pape", "head": "fixture", "working_tree_dirty": True})
    first, output = run_real_pilot(config, [], tmp_path / "cv_first", project_root=tmp_path)
    verified = verify_real_output(output, check_label_invariance=True)
    assert verified["status"] == "pass"
    pool = load_reference_pool(windows, config)
    outer, _ = split_source_keys(pool, config, 11)
    held_keys = set(outer["validation"].source_key)
    assert len(first["selection_cv_manifest"]) == 3
    assert all(not set(r["source_keys"]) & held_keys for r in first["records"])
    frame = frame.copy()
    held = frame.source_key.isin(held_keys)
    frame.loc[held, "y_true"] = 1 - frame.loc[held, "y_true"]
    frame.loc[held, "anomaly_fraction"] = frame.loc[held, "y_true"].astype(float)
    frame.to_parquet(windows, index=False)
    second, _ = run_real_pilot(config, [], tmp_path / "cv_poisoned_outer", project_root=tmp_path)
    assert first["selection"]["ranking"] == second["selection"]["ranking"]
    for a, b in zip(first["records"], second["records"]):
        for metric in ["fn", "fp", "recall", "f1", "loss"]:
            assert a["estimated_"+metric] == b["estimated_"+metric]
