from copy import deepcopy
from pathlib import Path

import pytest

from scripts.dummy_data import iter_target_runs, make_dummy_dataset
from src.pape.io import load_config


@pytest.fixture
def config():
    root = Path(__file__).resolve().parents[1]
    result = load_config(root / "config/base.yaml", root / "config/pape.yaml")
    # Mathematical unit-test constants, independent of the pilot cost decision.
    result["loss"] = {"status": "unit_test_only", "lambda_fn": 1.0, "lambda_fp": 1.0, "sensitivity": []}
    result["dummy"]["windows_per_run"] = 80
    result["dre"]["lgbm"]["n_estimators"] = 20
    result["calibrator"]["lgbm"]["n_estimators"] = 20
    return deepcopy(result)


@pytest.fixture
def frames(config):
    datasets, _ = make_dummy_dataset(config, config["seed"])
    target = next(run for _, run in iter_target_runs(datasets["selection"])
                  if run.data_class.iloc[0] == "abnormal")
    return datasets["source"], target
