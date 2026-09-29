"""Label-free Target diagnostics, shared by direct API and experiment callers.

These checks cannot establish covariate shift or guarantee an accurate estimate.
Source labels describe evidence availability only; Target labels are never read.
"""
def diagnose_reference_support(source, target, features):
    source_scenarios = sorted(source.scenario_id.unique().tolist())
    scenario = str(target.scenario_id.iloc[0])
    calibration_outside = ((target[features] < source[features].min()) |
                           (target[features] > source[features].max())).any(axis=1)
    # Evidence inside the Target bounding box is descriptive, not a proof of
    # multivariate overlap, and never modifies probabilities or Source weights.
    in_box = ((source[features] >= target[features].min()) &
              (source[features] <= target[features].max())).all(axis=1)
    positive_in_box = in_box & source.y_true.eq(1)
    return {
        "scenario_id": scenario,
        "source_scenarios": source_scenarios,
        "target_scenario_in_source": scenario in source_scenarios,
        "calibration_outside_source_range_fraction": float(calibration_outside.mean()),
        "source_windows_in_target_calibration_box": int(in_box.sum()),
        "source_positive_windows_in_target_calibration_box": int(positive_in_box.sum()),
        "source_positive_keys_in_target_calibration_box": int(source.loc[positive_in_box].source_key.nunique()),
        "support_diagnostic_uses_target_labels": False,
    }


def diagnostic_readiness(warnings):
    """No policy action is inferred from an unreliable current-loss estimate."""
    return {
        "status": "requires_review" if warnings else "no_diagnostic_flags",
        "review_reasons": sorted(set(warnings)),
        "accuracy_guaranteed": False,
        "production_approved": False,
        "policy_action": None,
    }
