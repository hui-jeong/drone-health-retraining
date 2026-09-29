"""Group splits and purged domain-validation folds for a one-Run Target."""
import numpy as np
import pandas as pd
from .randomness import resolve_seed


def assert_disjoint_source_keys(source_df, target_df):
    overlap = set(source_df.source_key) & set(target_df.source_key)
    if overlap:
        raise ValueError(f"Source/Target source_key leakage: {sorted(overlap)}")


def assert_not_official_test(df, config):
    forbidden = set(config["split"]["official_test_run_ids"])
    test_split = "split" in df and df["split"].eq("test").any()
    if set(df.run_id) & forbidden or test_split:
        raise ValueError("Official Test runs are prohibited in PAPE selection and pilot")


def split_source_keys(df, config, seed):
    """Pair normal/abnormal derivatives in source, selection or validation."""
    assert_not_official_test(df, config)
    keys = np.array(sorted(df.source_key.unique()), dtype=object)
    np.random.default_rng(resolve_seed(config, "split", default_seed=seed)).shuffle(keys)
    fractions = config["split"]["fractions"]
    if set(fractions) != {"source", "selection", "validation"}:
        raise ValueError("Expected source, selection, validation split fractions")
    if any(v <= 0 for v in fractions.values()) or not np.isclose(sum(fractions.values()), 1):
        raise ValueError("Split fractions must be positive and sum to one")
    ns = int(len(keys) * fractions["source"])
    nv = int(len(keys) * fractions["selection"])
    parts = {"source": keys[:ns], "selection": keys[ns:ns + nv], "validation": keys[ns + nv:]}
    if any(not len(v) for v in parts.values()):
        raise ValueError("Not enough source_key groups for all three splits")
    manifest = pd.DataFrame([{"source_key": k, "split": role, "seed": seed}
                             for role, group in parts.items() for k in group])
    return {role: df[df.source_key.isin(group)].copy() for role, group in parts.items()}, manifest


def balanced_source_key_blocks(frame, n_splits, seed):
    """Balance group counts and spread scenarios using metadata, never labels."""
    metadata = frame[["source_key", "scenario_id"]].drop_duplicates()
    if metadata.source_key.duplicated().any():
        raise ValueError("A source_key must identify one scenario")
    if n_splits < 2 or len(metadata) < n_splits:
        raise ValueError("Insufficient source_key groups for balanced folds")
    rng = np.random.default_rng(seed)
    blocks = [[] for _ in range(n_splits)]
    scenarios = {name: sorted(group.source_key.tolist()) for name, group in metadata.groupby("scenario_id")}
    for name in sorted(scenarios, key=lambda name: (-len(scenarios[name]), name)):
        keys = np.array(scenarios[name], dtype=object)
        rng.shuffle(keys)
        scenario_counts = np.zeros(n_splits, dtype=int)
        for key in keys:
            choices = np.flatnonzero(scenario_counts == scenario_counts.min())
            sizes = np.array([len(blocks[i]) for i in choices])
            choices = choices[sizes == sizes.min()]
            fold = int(rng.choice(choices))
            blocks[fold].append(key)
            scenario_counts[fold] += 1
    return [np.array(block, dtype=object) for block in blocks]


def group_selection_folds(development, config, seed):
    """Inner selection CV; the caller must exclude the outer validation groups."""
    assert_not_official_test(development, config)
    settings = config["experiment"]["selection_cv"]
    if settings["allocation"] != "scenario_balanced_source_keys":
        raise ValueError("Unknown selection CV allocation")
    blocks = balanced_source_key_blocks(development, settings["n_splits"], seed)
    for fold, held in enumerate(blocks):
        source = development[~development.source_key.isin(held)].copy()
        target = development[development.source_key.isin(held)].copy()
        assert_disjoint_source_keys(source, target)
        if source.source_key.nunique() < config["dre"]["n_splits"]:
            raise ValueError("Insufficient inner-CV Source groups for DRE")
        yield fold, source, target


def density_ratio_folds(source_df, target_df, n_splits, seed, *, end_inclusive=False,
                        source_allocation="random_source_keys"):
    """Source OOF by source_key; Target chronological blocks, overlapping windows purged.

    A single Target Run has just one source_key, so holding out its entire key
    would remove all Target training data. This is explicitly a within-Run DRE
    diagnostic, not evidence of generalization to other Target Runs.
    """
    keys = np.array(sorted(source_df.source_key.unique()), dtype=object)
    if n_splits < 2 or len(keys) < n_splits or len(target_df) < n_splits:
        raise ValueError("Insufficient Source groups or Target windows for DRE folds")
    if source_allocation == "random_source_keys":
        np.random.default_rng(seed).shuffle(keys)
        key_blocks = np.array_split(keys, n_splits)
    elif source_allocation == "scenario_balanced_source_keys":
        key_blocks = balanced_source_key_blocks(source_df, n_splits, seed)
    else:
        raise ValueError("Unknown DRE Source fold allocation")
    order = np.argsort(target_df.window_start.to_numpy(), kind="stable")
    target_blocks = np.array_split(order, n_splits)
    all_source = np.arange(len(source_df))
    for fold, (held_keys, target_test) in enumerate(zip(key_blocks, target_blocks)):
        source_test = np.flatnonzero(source_df.source_key.isin(held_keys).to_numpy())
        source_train = np.setdiff1d(all_source, source_test)
        start = target_df.iloc[target_test].window_start.min()
        end = target_df.iloc[target_test].window_end.max()
        if end_inclusive:
            # Exported end is the last sample itself: equality shares a sample.
            separate = (target_df.window_end < start) | (target_df.window_start > end)
        else:
            separate = (target_df.window_end <= start) | (target_df.window_start >= end)
        target_train = np.flatnonzero(separate.to_numpy())
        if not len(source_train) or not len(target_train):
            raise ValueError("Too few non-overlapping Target windows for purged DRE validation")
        yield fold, source_train, source_test, target_train, target_test
