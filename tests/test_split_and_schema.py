import numpy as np
import pandas as pd
import pytest

from scripts.dummy_data import make_dummy_dataset
from src.pape.io import config_hash, read_window_features
from src.pape.contracts import WINDOW_FEATURE_COLUMNS, validate_window_features
from src.pape.split import density_ratio_folds, split_source_keys


def test_group_split_keeps_normal_and_abnormal_together(config):
    datasets, manifest = make_dummy_dataset(config, config["seed"])
    assert not manifest.source_key.duplicated().any()
    sets = [set(df.source_key) for df in datasets.values()]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    for df in datasets.values():
        assert list(df.columns) == WINDOW_FEATURE_COLUMNS
        assert df.groupby("source_key").data_class.nunique().eq(2).all()
        assert not df.window_id.duplicated().any()
        assert df.window_id.eq(df.source_key + "::" + df.window_start.map(lambda t: f"{t:.3f}")).all()
        assert np.allclose(df.window_end - df.window_start, config["window"]["duration_seconds"])


def test_split_reproducibility_and_seed_change(config):
    datasets, manifest = make_dummy_dataset(config, config["seed"])
    all_data = pd.concat(datasets.values())
    _, same = split_source_keys(all_data, config, config["seed"])
    _, different = split_source_keys(all_data, config, config["seed"] + 1)
    pd.testing.assert_frame_equal(manifest, same)
    assert not same.set_index("source_key").split.sort_index().equals(
        different.set_index("source_key").split.sort_index())


def test_dre_folds_hold_out_source_groups_and_purge_target_overlap(config, frames):
    source, target = [df.reset_index(drop=True) for df in frames]
    source_seen, target_seen = [], []
    for _, si, so, ti, to in density_ratio_folds(source, target, 3, config["seed"]):
        assert not set(source.iloc[si].source_key) & set(source.iloc[so].source_key)
        assert not set(ti) & set(to)
        train, test = target.iloc[ti], target.iloc[to]
        overlapping = (train.window_start.to_numpy()[:, None] < test.window_end.to_numpy()) & (
            train.window_end.to_numpy()[:, None] > test.window_start.to_numpy())
        assert not overlapping.any()
        source_seen.extend(so)
        target_seen.extend(to)
    assert sorted(source_seen) == list(range(len(source)))
    assert sorted(target_seen) == list(range(len(target)))


def test_official_test_is_rejected(config, frames):
    frame = frames[0].copy()
    frame["run_id"] = "008"
    with pytest.raises(ValueError, match="Official Test"):
        split_source_keys(frame, config, config["seed"])


@pytest.mark.parametrize("defect", ["duplicate", "nan", "infinite", "label", "run_id", "time", "scenario"])
def test_invalid_schema_fails(frames, defect):
    df = frames[0].copy().reset_index(drop=True)
    if defect == "duplicate":
        df.loc[1, "window_id"] = df.loc[0, "window_id"]
    elif defect == "nan":
        df.loc[0, "hybrid_score"] = np.nan
    elif defect == "infinite":
        df.loc[0, "ae_score"] = np.inf
    elif defect == "label":
        df.loc[0, "y_true"] = 1 - df.loc[0, "y_true"]
    elif defect == "run_id":
        df["run_id"] = 1
    elif defect == "time":
        df.loc[0, "window_end"] = -1
    else:
        df["scenario_id"] = "A3"
    with pytest.raises(ValueError):
        validate_window_features(df)


def test_csv_reader_preserves_leading_zero_run_id(frames, tmp_path):
    path = tmp_path / "window_features.csv"
    frames[0].to_csv(path, index=False)
    reloaded = read_window_features(path)
    assert set(reloaded.run_id) <= {"000", "001"}
    validate_window_features(reloaded)


def test_config_hash_is_order_independent_and_value_sensitive():
    assert config_hash({"b": 2, "a": 1}) == config_hash({"a": 1, "b": 2})
    assert config_hash({"a": 1}) != config_hash({"a": 2})
