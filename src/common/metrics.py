"""Expected confusion counts and post-estimation evaluation."""
import numpy as np


def confusion_metrics(tp, fp, fn, tn, lambda_fn, lambda_fp, zero_division=0.0):
    values = np.asarray([tp, fp, fn, tn, lambda_fn, lambda_fp], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Counts and loss weights must be finite and nonnegative")
    if not np.isfinite(zero_division) or not 0 <= zero_division <= 1:
        raise ValueError("zero_division must be within [0, 1]")
    return {
        "tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn),
        "recall": float(tp / (tp + fn)) if tp + fn > 0 else float(zero_division),
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn > 0 else float(zero_division),
        "loss": float(lambda_fn * fn + lambda_fp * fp),
        "recall_defined": bool(tp + fn > 0),
        "f1_defined": bool(2 * tp + fp + fn > 0),
    }


def expected_confusion(probabilities, predictions, lambda_fn, lambda_fp, zero_division=0.0):
    p, pred = np.asarray(probabilities, dtype=float), np.asarray(predictions)
    if p.ndim != 1 or p.shape != pred.shape or not len(p):
        raise ValueError("Probabilities and predictions must be nonempty aligned 1D arrays")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be within [0, 1]")
    if not np.isin(pred, [0, 1]).all():
        raise ValueError("Predictions must be binary")
    pos = pred == 1
    return confusion_metrics(
        p[pos].sum(), (1 - p[pos]).sum(), p[~pos].sum(), (1 - p[~pos]).sum(),
        lambda_fn, lambda_fp, zero_division,
    )


def actual_confusion(y_true, predictions, lambda_fn, lambda_fp, zero_division=0.0):
    y = np.asarray(y_true)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Evaluation labels must be binary")
    return expected_confusion(y, predictions, lambda_fn, lambda_fp, zero_division)


def estimation_errors(estimated, actual):
    return {f"{metric}_absolute_error": abs(estimated[metric] - actual[metric])
            for metric in ["tp", "fp", "fn", "tn", "recall", "f1", "loss"]}
