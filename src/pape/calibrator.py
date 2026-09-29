"""Target-specific weighted score calibration, following PAPE section 3.3."""
import numpy as np
from lightgbm import LGBMRegressor, record_evaluation
from sklearn.isotonic import IsotonicRegression

from src.pape.contracts import model_features
from .randomness import resolve_seed


def calibrated_target_probabilities(source_df, target_df, features, weights, config):
    features = model_features(features)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(source_df),) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Calibration requires one finite positive weight per Source window")
    if source_df.y_true.nunique() < 2:
        raise ValueError("Source calibration requires both normal and anomalous labels")
    definition = config["pape"]["variants"][config["pape"]["input_variant"]]
    method = definition.get("calibration_method", "lgbm")
    training_history = {}
    if method == "lgbm":
        model = LGBMRegressor(**config["calibrator"]["lgbm"], random_state=resolve_seed(config, "model"))
        fit_features = source_df[features]
        recording = config['calibrator'].get('record_training_loss', False)
        evaluation = {'eval_set': [(fit_features, source_df.y_true)],
                      'eval_names': ['source_fit'], 'eval_sample_weight': [weights],
                      'eval_metric': 'l2', 'callbacks': [record_evaluation(training_history)]} if recording else {}
        model.fit(fit_features, source_df.y_true, sample_weight=weights, **evaluation)
        unbounded = model.predict(target_df[features])
    elif method == "isotonic":
        if len(features) != 1:
            raise ValueError("Isotonic calibration requires exactly one score feature")
        model = IsotonicRegression(**config["calibrator"]["isotonic"])
        model.fit(source_df[features[0]], source_df.y_true, sample_weight=weights)
        unbounded = model.predict(target_df[features[0]])
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    if not np.isfinite(unbounded).all():
        raise ValueError("Calibration produced nonfinite probabilities")
    return np.clip(unbounded, 0, 1), {
        "calibration_features": features,
        "calibration_probability_clip_fraction": float(((unbounded < 0) | (unbounded > 1)).mean()),
        "calibration_model": type(model).__name__,
        **({'calibration_training_history': training_history,
            'calibration_training_metric_definition': 'Source sample-weighted MSE of probability regression; in-sample, not Target performance',
            'calibration_training_windows': len(source_df),
            'calibration_training_positive_windows': int(source_df.y_true.sum())} if training_history else {}),
    }
