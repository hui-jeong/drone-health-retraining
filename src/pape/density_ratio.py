"""LGBM domain classifier and PAPE density ratio (paper equation 3)."""
from dataclasses import dataclass
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
import platform
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from src.pape.contracts import model_features
from .split import density_ratio_folds
from .randomness import resolve_seed


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _frame_values(frame, columns):
    """Canonical values preserve row order, but deliberately omit the pandas index."""
    return frame.loc[:, columns].to_numpy(dtype=object).tolist()


def dre_audit_fingerprint(source_df, target_df, features, config):
    """Bind a label-free purged DRE audit to data, settings and the running environment.

    Calibration settings, input names, anomaly labels and weight clipping do not
    determine the fitted domain classifier and therefore are not part of this key.
    """
    features = model_features(features)
    columns = ["window_id", "source_key", "run_id", "scenario_id", "window_start", "window_end", *features]
    settings = config["dre"]
    return _digest({
        "version": "pape_dre_audit_v1", "columns": columns,
        "source": _frame_values(source_df, columns), "target": _frame_values(target_df, columns),
        "n_splits": settings["n_splits"],
        "source_fold_allocation": settings.get("source_fold_allocation", "random_source_keys"),
        "timestamp_convention": config["window"].get("timestamp_convention"),
        "dre_fold_seed": resolve_seed(config, "dre_fold"), "model_seed": resolve_seed(config, "model"),
        "lgbm": settings["lgbm"],
        "environment": {"python": platform.python_version(), "platform": platform.system(),
                        "packages": {name: version(name) for name in
                                     ["numpy", "pandas", "scikit-learn", "lightgbm"]}},
    })


def _fold_metadata(fold, source, target, si, so, ti, to):
    return {
        "fold": fold, "n_source_train": len(si), "n_source_test": len(so),
        "n_target_train": len(ti), "n_target_test": len(to),
        "target_purged_count": len(target) - len(ti) - len(to),
        "source_train_keys": sorted(source.iloc[si].source_key.unique()),
        "source_test_keys": sorted(source.iloc[so].source_key.unique()),
        "target_holdout_start": float(target.iloc[to].window_start.min()),
        "target_holdout_end": float(target.iloc[to].window_end.max()),
        "prior_source_over_target": len(si) / len(ti),
        "held_source_scenarios_absent_from_fold_training": sorted(
            set(source.iloc[so].scenario_id) - set(source.iloc[si].scenario_id)),
    }


def _domain_rows(frame, probabilities, domain, role, fold):
    rows = frame[["window_id", "source_key", "run_id"]].copy()
    rows["model_role"], rows["fold"] = role, fold
    rows["domain_label"], rows["domain_probability"] = domain, probabilities
    return rows.to_dict("records")


def _validate_reference_metadata(value):
    """Reference provenance is never a channel for anomaly labels or metric errors."""
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str) or key in {"y_true", "anomaly_fraction", "data_class", "anomaly_type"} or \
                    key.startswith(("actual_", "estimated_", "calibrated_")) or \
                    key.endswith(("_mae", "_absolute_error")):
                raise ValueError("Unsafe DRE audit reference metadata")
            _validate_reference_metadata(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_reference_metadata(child)
    elif value is not None and not isinstance(value, (str, bool, int, float)):
        raise ValueError("Invalid DRE audit reference metadata type")
    elif isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Nonfinite DRE audit reference metadata")


def _validate_audit_reference(reference, source, target, features, config):
    required = {"fingerprint", "dre_auc", "folds", "origin", "reference"}
    if not isinstance(reference, dict) or set(reference) != required:
        raise ValueError("DRE audit reference must contain only fingerprint, dre_auc, folds, origin, reference")
    if reference["fingerprint"] != dre_audit_fingerprint(source, target, features, config):
        raise ValueError("DRE audit fingerprint mismatch")
    if reference["origin"] not in {"archived_purged_holdout", "computed_purged_holdout"}:
        raise ValueError("Invalid DRE audit reference origin")
    if not isinstance(reference["reference"], dict) or not reference["reference"]:
        raise ValueError("DRE audit reference provenance is required")
    _validate_reference_metadata(reference["reference"])
    auc = reference["dre_auc"]
    if isinstance(auc, bool) or not isinstance(auc, (int, float)) or not np.isfinite(auc) or not 0 <= auc <= 1:
        raise ValueError("Invalid DRE audit AUC")
    settings = config["dre"]
    folds = reference["folds"]
    if not isinstance(folds, list) or len(folds) != settings["n_splits"]:
        raise ValueError("DRE audit fold count mismatch")
    expected_folds = density_ratio_folds(
        source, target, settings["n_splits"], resolve_seed(config, "dre_fold"),
        end_inclusive=config["window"].get("timestamp_convention") == "last_sample",
        source_allocation=settings.get("source_fold_allocation", "random_source_keys"))
    aucs = []
    for observed, (fold, si, so, ti, to) in zip(folds, expected_folds):
        expected = _fold_metadata(fold, source, target, si, so, ti, to)
        if not isinstance(observed, dict) or set(observed) != set(expected) | {"dre_auc"}:
            raise ValueError("DRE audit fold metadata fields mismatch")
        for name, value in expected.items():
            if observed[name] != value:
                raise ValueError("DRE audit fold metadata mismatch: " + name)
        fold_auc = observed["dre_auc"]
        if isinstance(fold_auc, bool) or not isinstance(fold_auc, (int, float)) or \
                not np.isfinite(fold_auc) or not 0 <= fold_auc <= 1:
            raise ValueError("Invalid DRE audit fold AUC")
        aucs.append(fold_auc)
    if not np.isclose(auc, np.mean(aucs), atol=1e-12, rtol=1e-12):
        raise ValueError("DRE audit AUC differs from fold mean")
    return deepcopy(reference)


@dataclass
class DensityRatioResult:
    weights: np.ndarray
    table: pd.DataFrame
    diagnostics: dict


def density_ratio_from_probability(probability, n_source, n_target, epsilon):
    p = np.asarray(probability, dtype=float)
    if n_source <= 0 or n_target <= 0 or not 0 < epsilon < 0.5:
        raise ValueError("Invalid domain sample counts or probability epsilon")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Domain probability must be finite and within [0, 1]")
    p = np.clip(p, epsilon, 1 - epsilon)
    return (n_source / n_target) * p / (1 - p)


def effective_sample_size(weights):
    w = np.asarray(weights, dtype=float)
    if not len(w) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("ESS requires finite nonnegative weights with positive total")
    return float(w.sum() ** 2 / np.square(w).sum())


def _estimate_oof(source_df, target_df, features, config, *, capture_domain=False):
    features = model_features(features)
    settings = config["dre"]
    params = dict(settings["lgbm"])
    if params.get("class_weight") is not None or params.get("is_unbalance", False) or params.get("scale_pos_weight", 1) != 1:
        raise ValueError("DRE class balancing would invalidate the empirical prior correction")
    if params.get("objective", "binary") != "binary":
        raise ValueError("DRE requires the binary objective")
    lo, hi = settings["clip_min"], settings["clip_max"]
    if not 0 < lo <= hi or not np.isfinite([lo, hi]).all():
        raise ValueError("DRE clipping must satisfy 0 < clip_min <= clip_max < infinity")
    ns = len(source_df)
    raw = np.empty(ns)
    source_probability = np.empty(ns)
    folds = np.empty(ns, dtype=int)
    priors = np.empty(ns)
    aucs, fold_details, domain_rows = [], [], []
    for fold, si, so, ti, to in density_ratio_folds(
        source_df, target_df, settings["n_splits"], resolve_seed(config, "dre_fold"),
        end_inclusive=config["window"].get("timestamp_convention") == "last_sample",
        source_allocation=settings.get("source_fold_allocation", "random_source_keys"),
    ):
        x_train = pd.concat([source_df.iloc[si][features], target_df.iloc[ti][features]], ignore_index=True)
        domains = np.r_[np.zeros(len(si), dtype=int), np.ones(len(ti), dtype=int)]
        model = LGBMClassifier(**params, random_state=resolve_seed(config, "model") + fold)
        model.fit(x_train, domains)
        ps = model.predict_proba(source_df.iloc[so][features])[:, 1]
        pt = model.predict_proba(target_df.iloc[to][features])[:, 1]
        if capture_domain:
            domain_rows.extend(_domain_rows(source_df.iloc[so], ps, 0, "purged_holdout", fold))
            domain_rows.extend(_domain_rows(target_df.iloc[to], pt, 1, "purged_holdout", fold))
        raw[so] = density_ratio_from_probability(ps, len(si), len(ti), settings["probability_epsilon"])
        source_probability[so], folds[so], priors[so] = ps, fold, len(si) / len(ti)
        auc = float(roc_auc_score(np.r_[np.zeros(len(so)), np.ones(len(to))], np.r_[ps, pt]))
        aucs.append(auc)
        fold_details.append({
            "fold": fold, "dre_auc": auc, "n_source_train": len(si), "n_source_test": len(so),
            "n_target_train": len(ti), "n_target_test": len(to),
            "target_purged_count": len(target_df) - len(ti) - len(to),
            "source_train_keys": sorted(source_df.iloc[si].source_key.unique()),
            "source_test_keys": sorted(source_df.iloc[so].source_key.unique()),
            "target_holdout_start": float(target_df.iloc[to].window_start.min()),
            "target_holdout_end": float(target_df.iloc[to].window_end.max()),
            "prior_source_over_target": len(si) / len(ti),
            "held_source_scenarios_absent_from_fold_training": sorted(
                set(source_df.iloc[so].scenario_id) - set(source_df.iloc[si].scenario_id)),
        })
    clipped = np.clip(raw, lo, hi)
    ess = effective_sample_size(clipped)
    table = source_df[["window_id", "source_key", "run_id"]].copy()
    table["fold"] = folds
    table["domain_probability"] = source_probability
    table["prior_source_over_target"] = priors
    table["raw_weight"] = raw
    table["clipped_weight"] = clipped
    table["was_clipped"] = raw != clipped
    support = ((target_df[features] < source_df[features].min()) |
               (target_df[features] > source_df[features].max())).any(axis=1).mean()
    diagnostics = {
        "dre_auc": float(np.mean(aucs)), "dre_auc_aggregation": "mean_purged_holdout_fold_auc",
        "ess": ess, "ess_ratio": ess / ns, "raw_ess": effective_sample_size(raw),
        "max_weight": float(clipped.max()), "max_raw_weight": float(raw.max()),
        "clip_fraction": float((raw != clipped).mean()),
        "target_outside_source_range_fraction": float(support),
        "folds": fold_details,
        "weight_method": "source_group_out_of_fold_with_fold_training_prior_correction",
        "source_fold_allocation": settings.get("source_fold_allocation", "random_source_keys"),
    }
    if capture_domain:
        diagnostics["_domain_diagnostics"] = domain_rows
    return DensityRatioResult(clipped, table, diagnostics)


def _estimate_full(source, target, features, config):
    settings = config["dre"]
    if settings.get("diagnostic_auc_method", "purged_holdout") != "purged_holdout":
        raise ValueError("Full DRE diagnostic_auc_method must be 'purged_holdout'")
    params = dict(settings["lgbm"])
    if params.get("class_weight") is not None or params.get("is_unbalance", False) or params.get("scale_pos_weight", 1) != 1:
        raise ValueError("DRE class balancing would invalidate the empirical prior correction")
    if params.get("objective", "binary") != "binary":
        raise ValueError("DRE requires the binary objective")
    lo, hi = settings["clip_min"], settings["clip_max"]
    if not 0 < lo <= hi or not np.isfinite([lo, hi]).all():
        raise ValueError("DRE clipping must satisfy 0 < clip_min <= clip_max < infinity")
    ns, nt = len(source), len(target)
    if not ns or not nt:
        raise ValueError("Full DRE requires nonempty Source and Target")
    # Validate external audit provenance before fitting either model.
    reference = settings.get("audit_reference")
    if reference is not None:
        reference = _validate_audit_reference(reference, source, target, features, config)
    fingerprint = dre_audit_fingerprint(source, target, features, config)
    model_seed = resolve_seed(config, "model")
    x_train = pd.concat([source[features], target[features]], ignore_index=True)
    domains = np.r_[np.zeros(ns, dtype=int), np.ones(nt, dtype=int)]
    model = LGBMClassifier(**params, random_state=model_seed)
    model.fit(x_train, domains)
    ps = model.predict_proba(source[features])[:, 1]
    pt = model.predict_proba(target[features])[:, 1]
    raw = density_ratio_from_probability(ps, ns, nt, settings["probability_epsilon"])
    if not np.isfinite(pt).all() or ((pt < 0) | (pt > 1)).any():
        raise ValueError("Domain probability must be finite and within [0, 1]")
    clipped = np.clip(raw, lo, hi)
    training_auc = float(roc_auc_score(domains, np.r_[ps, pt]))
    domain_rows = (_domain_rows(source, ps, 0, "full_training", None) +
                   _domain_rows(target, pt, 1, "full_training", None))
    if reference is None:
        # This separate audit never supplies weights to the calibration model.
        audit = _estimate_oof(source, target, features, config, capture_domain=True).diagnostics
        domain_rows.extend(audit.pop("_domain_diagnostics"))
        reference = {
            "fingerprint": fingerprint, "dre_auc": audit["dre_auc"], "folds": audit["folds"],
            "origin": "computed_purged_holdout", "reference": {"computed_for_fingerprint": fingerprint},
        }
        auc_role = "separate_purged_holdout_diagnostic_not_full_model_validation"
    else:
        auc_role = "shared_purged_holdout_diagnostic_not_full_model_validation"
    table = source[["window_id", "source_key", "run_id"]].copy()
    table["fold"] = None
    table["domain_probability"] = ps
    table["prior_source_over_target"] = ns / nt
    table["raw_weight"], table["clipped_weight"] = raw, clipped
    table["was_clipped"] = raw != clipped
    ess = effective_sample_size(clipped)
    id_columns = ["window_id", "source_key", "run_id"]
    support = ((target[features] < source[features].min()) |
               (target[features] > source[features].max())).any(axis=1).mean()
    diagnostics = {
        "dre_auc": reference["dre_auc"], "dre_auc_aggregation": "mean_purged_holdout_fold_auc",
        "dre_auc_role": auc_role, "dre_auc_origin": reference["origin"],
        "dre_audit_fingerprint": fingerprint, "audit_reference": reference,
        "dre_training_auc": training_auc, "dre_training_auc_role": "in_sample_not_for_selection",
        "ess": ess, "ess_ratio": ess / ns, "raw_ess": effective_sample_size(raw),
        "max_weight": float(clipped.max()), "max_raw_weight": float(raw.max()),
        "clip_fraction": float((raw != clipped).mean()),
        "target_outside_source_range_fraction": float(support), "folds": reference["folds"],
        "weight_estimation_method": "full_source_target",
        "weight_method": "full_source_target_with_full_training_prior_correction",
        "source_fold_allocation": settings.get("source_fold_allocation", "random_source_keys"),
        "full_fit": {"n_source": ns, "n_target": nt, "model_seed": model_seed,
                     "source_id_sha256": _digest(_frame_values(source, id_columns)),
                     "target_id_sha256": _digest(_frame_values(target, id_columns))},
        "_domain_diagnostics": domain_rows,
    }
    return DensityRatioResult(clipped, table, diagnostics)


def estimate_density_ratio(source_df, target_df, features, config):
    """Preserve legacy OOF by default; full fitting is an explicit comparison arm."""
    features = model_features(features)
    method = config["dre"].get("weight_estimation_method", "source_group_oof")
    if method == "source_group_oof":
        if config["dre"].get("audit_reference") is not None:
            raise ValueError("DRE audit references are valid only for full_source_target")
        return _estimate_oof(source_df, target_df, features, config)
    if method == "full_source_target":
        return _estimate_full(source_df, target_df, features, config)
    raise ValueError("Unknown DRE weight_estimation_method")
