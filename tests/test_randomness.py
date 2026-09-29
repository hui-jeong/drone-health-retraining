from copy import deepcopy

import pandas as pd
import pytest

from scripts.dummy_data import make_dummy_dataset
from src.pape.estimator import estimate_target_performance
from src.pape.randomness import resolve_seed, resolved_seeds
from src.pape.split import split_source_keys


def test_legacy_seeds_unchanged(config):
    assert resolved_seeds(config) == {"split_seed": config["seed"], "dre_fold_seed": config["seed"], "model_seed": config["seed"]}
    assert resolve_seed(config, "split", default_seed=23) == 23


@pytest.mark.parametrize("seed", [-1, 1.5, "11", True])
def test_invalid_seed_rejected(config, seed):
    config["randomness"] = {"model_seed": seed}
    with pytest.raises(ValueError, match="integer"):
        resolve_seed(config, "model")


def test_split_override_preserves_group_assignments(config):
    datasets, _ = make_dummy_dataset(config, 11)
    pool = pd.concat(datasets.values(), ignore_index=True)
    base, _ = split_source_keys(pool, config, 11)
    config["randomness"] = {"split_seed": 11}
    changed, _ = split_source_keys(pool, config, 71)
    for role in base:
        pd.testing.assert_frame_equal(base[role], changed[role])


def test_model_seed_does_not_change_dre_folds(config, frames):
    config["randomness"] = {"dre_fold_seed": 11, "model_seed": 11}
    a = estimate_target_performance(*frames, config)
    config["randomness"]["model_seed"] = 71
    b = estimate_target_performance(*frames, config)
    pd.testing.assert_series_equal(a.source_weights.fold, b.source_weights.fold)
    assert a.diagnostics["resolved_seeds"]["model_seed"] == 11
    assert b.diagnostics["resolved_seeds"]["model_seed"] == 71


def test_fold_seed_is_actually_varied(config, frames):
    config["randomness"] = {"dre_fold_seed": 11, "model_seed": 11}
    a = estimate_target_performance(*frames, config)
    config["randomness"]["dre_fold_seed"] = 71
    b = estimate_target_performance(*frames, config)
    assert not a.source_weights.fold.equals(b.source_weights.fold)
    assert b.diagnostics["resolved_seeds"]["model_seed"] == 11
