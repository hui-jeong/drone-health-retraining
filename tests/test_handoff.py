from copy import deepcopy
import json

import pandas as pd
import pytest

from scripts.run_pape_pilot import run_pilot, select_input
from src.pape.io import load_config
from src.pape.contracts import PAPE_ESTIMATE_COLUMNS


def test_selection_uses_errors_not_auc_and_ignores_validation(config):
    records = []
    for seed in config["experiment"]["seeds"]:
        for name, error, auc in [("hybrid_score", 1, 0.51), ("ae_score", 4, 0.99)]:
            records.append({"input_name": name, "seed": seed, "split": "selection",
                            "case": "covariate_shift", "weighting": "dre", "dre_auc": auc,
                            "fn_absolute_error": error, "recall_absolute_error": error / 10,
                            "f1_absolute_error": error / 10, "loss_absolute_error": error,
                            "actual_recall_defined": True})
    bad_validation = deepcopy(records[-1])
    bad_validation.update(input_name="ae_score", split="validation", fn_absolute_error=0,
                          recall_absolute_error=0, f1_absolute_error=0, loss_absolute_error=0)
    selected = select_input(records + [bad_validation], config)
    assert selected["selected_input_variant"] == "hybrid_score"
    assert selected["same_winner_all_seeds"]


def test_small_end_to_end_handoff(config, tmp_path):
    # Exercise CLI workflow on a small fixture, including paired baselines,
    # input selection, frozen-choice validation, provenance and cost sensitivity.
    config["dummy"]["scenario_ids"] = ["A1", "A2", "B1", "B2"]
    config["experiment"]["seeds"] = [11, 23]
    config["experiment"]["diagnostic_cases"] = []
    config["pape"]["variants"] = {k: config["pape"]["variants"][k]
                                     for k in ["hybrid_score", "ae_score"]}
    config["loss"]["sensitivity"] = [{"name": "unit_test_10_to_1", "lambda_fn": 10.0, "lambda_fp": 1.0}]
    output = tmp_path / "experiment"
    summary, _ = run_pilot(config, [], output)
    csv = pd.read_csv(output / "pape_estimates.csv")
    with (output / "pape_summary.json").open(encoding="utf-8") as stream:
        saved = json.load(stream)
    assert list(csv.columns) == PAPE_ESTIMATE_COLUMNS
    assert len(csv) == len(saved["records"]) > 0
    assert saved["config_hash"] == summary["config_hash"]
    assert len(saved["loss_sensitivity_metrics"]) > 0
    for aggregate in saved["loss_sensitivity_metrics"]:
        contributing = [r for r in saved["records"] if all(
            r[key] == aggregate[key] for key in ["case", "split", "input_name", "weighting"])]
        assert aggregate["loss_mae"] == pytest.approx(sum(
            r["loss_sensitivity"][0]["loss_absolute_error"] for r in contributing) / len(contributing))
    assert not saved["official_test_used"]
    assert not saved["real_hybrid_model_executed"]
    assert not saved["target_labels_used_during_estimation"]
    resolved = load_config(output / "config/resolved.yaml")
    assert resolved == config
    for row_index, record in enumerate(saved["records"]):
        assert record["csv_row_index"] == row_index
        for key in ["seed", "config_hash", "model_version", "source_key", "executed_at",
                    "estimated_tp", "estimated_tn", "actual_tp", "actual_tn", "actual_loss",
                    "fn_absolute_error", "recall_absolute_error", "f1_absolute_error", "loss_absolute_error"]:
            assert key in record
        assert csv.iloc[row_index].estimated_fn == pytest.approx(record["estimated_fn"])
        assert record["loss_sensitivity"][0]["estimated_loss"] == pytest.approx(
            10 * record["estimated_fn"] + record["estimated_fp"])
    table = pd.DataFrame(saved["records"])
    for _, group in table.groupby(["seed", "case", "split", "target_chunk_id", "input_name"]):
        assert set(group.weighting) == {"uniform", "dre"}
        assert group.actual_loss.nunique() == 1
        assert group.actual_fn.nunique() == 1
        assert len({tuple(keys) for keys in group.source_keys}) == 1
    assert (output / "source_weights.csv.gz").exists()
    weights = pd.read_csv(output / "source_weights.csv.gz")
    assert len(weights) == sum(r["n_source"] for r in saved["records"])
    assert (output / "pape_report.md").exists()
    with pytest.raises(FileExistsError):
        run_pilot(config, [], output)
