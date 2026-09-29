"""Feature-level fixtures, not simulated executions of the actual Hybrid model.

Labels use a shared deterministic rule in (x, z) under covariate shift. Normal
samples are constrained to the negative-label region. The concept-shift fixture
keeps Target observables identical and changes only the evaluation labels.
"""
import numpy as np
import pandas as pd

from src.pape.contracts import LATENT_COLUMNS, WINDOW_FEATURE_COLUMNS, validate_window_features
from src.pape.split import split_source_keys


def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def _postprocess(scores, settings):
    state, consecutive = 0, 0
    output = []
    for score in scores:
        consecutive = consecutive + 1 if score >= settings["prediction_threshold"] else 0
        if consecutive >= settings["consecutive_on_windows"]:
            state = 1
        elif score < settings["hysteresis_off_threshold"]:
            state = 0
        output.append(state)
    return np.asarray(output, dtype=int)


def make_dummy_dataset(config, seed, case="covariate_shift"):
    if case not in {"covariate_shift", "no_shift", "concept_shift", "support_shift"}:
        raise ValueError(f"Unknown dummy diagnostic case: {case}")
    settings = config["dummy"]
    n = settings["windows_per_run"]
    if n < 2:
        raise ValueError("Dummy runs require at least two windows")
    keys = [{"source_key": f"dummy_s{seed}_{scenario}_{run}", "scenario_id": scenario, "run_id": run}
            for scenario in settings["scenario_ids"] for run in settings["run_ids"]]
    _, manifest = split_source_keys(pd.DataFrame(keys), config, seed)
    roles = dict(zip(manifest.source_key, manifest.split))
    reference_rng = np.random.default_rng(seed + settings["validation_seed_offset"])
    normal_validation_ae = np.sort(np.exp(reference_rng.normal(-0.7, settings["noise_scale"],
                                                              settings["normal_validation_windows"])))
    rng = np.random.default_rng(seed)
    frames = []
    for key in keys:
        is_target = roles[key["source_key"]] != "source"
        shifted = is_target and case != "no_shift"
        for data_class in ["normal", "abnormal"]:
            normal = data_class == "normal"
            x = rng.normal(-0.7 if normal else 0.2, settings["noise_scale"], n)
            z = rng.normal(settings["source_z_mean"], settings["noise_scale"], n)
            if shifted:
                z += settings["target_z_shift"]
                if normal:
                    x += settings["normal_target_x_shift"]
            if is_target and case == "support_shift":
                x += settings["support_x_shift"]
            if normal:
                z = np.minimum(z, -x - 0.05)
            y = (x + z > 0).astype(int)
            if is_target and case == "concept_shift" and not normal:
                y = 1 - y
            ae = np.exp(x)
            percentile = np.searchsorted(normal_validation_ae, ae, side="right") / len(normal_validation_ae)
            residual = _sigmoid(0.65 * x + 0.85 * z)
            hybrid = settings["fusion_ae_weight"] * percentile + settings["fusion_residual_weight"] * residual
            raw = (hybrid >= settings["prediction_threshold"]).astype(int)
            final = _postprocess(hybrid, settings)
            # Keep the exact source_key + start rule. Synthetic paired files use
            # disjoint relative starts until the real exporter ID ambiguity is resolved.
            offset = 0 if normal else n * config["window"]["stride_seconds"] + config["window"]["duration_seconds"]
            start = offset + np.arange(n) * config["window"]["stride_seconds"]
            end = start + config["window"]["duration_seconds"]
            fraction = np.zeros(n) if normal else np.where(
                y == 1, settings["abnormal_positive_fraction"], settings["abnormal_negative_fraction"])
            frame = pd.DataFrame({
                "window_id": [f"{key['source_key']}::{t:.3f}" for t in start],
                **key, "phase": "cruise", "data_class": data_class,
                "anomaly_type": "none" if normal else "synthetic_fixture",
                "window_start": start, "window_end": end, "anomaly_fraction": fraction,
                "y_true": y, "ae_score": ae, "ae_percentile_score": percentile,
                "residual_probability": residual, "hybrid_score": hybrid,
                "context_threshold": settings["prediction_threshold"],
                "raw_prediction": raw, "final_prediction": final,
            })
            latent = rng.normal(size=(n, len(LATENT_COLUMNS)))
            latent[:, 0], latent[:, 1] = x, z
            frames.append(pd.concat([frame, pd.DataFrame(latent, columns=LATENT_COLUMNS)], axis=1))
    df = pd.concat(frames, ignore_index=True).loc[:, WINDOW_FEATURE_COLUMNS]
    validate_window_features(df)
    return {role: df[df.source_key.isin(manifest.loc[manifest.split == role, "source_key"])].copy()
            for role in ["source", "selection", "validation"]}, manifest


def iter_target_runs(df):
    """One model_ready file = one Chunk; normal/abnormal siblings remain distinct Runs."""
    for identity, run in df.groupby(["source_key", "run_id", "scenario_id", "data_class", "anomaly_type"], sort=True):
        source_key, run_id, _, data_class, anomaly_type = identity
        yield f"{source_key}::{run_id}::{data_class}::{anomaly_type}", run.sort_values("window_start").copy()
