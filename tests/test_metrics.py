import numpy as np
import pytest

from src.common.metrics import actual_confusion, confusion_metrics, expected_confusion
from src.pape.density_ratio import density_ratio_from_probability, effective_sample_size


def test_expected_confusion_hand_calculation():
    result = expected_confusion([0.9, 0.2, 0.6, 0.1], [1, 1, 0, 0], 10, 1)
    for key, expected in {"tp": 1.1, "fp": 0.9, "fn": 0.7, "tn": 1.3,
                          "recall": 1.1 / 1.8, "f1": 2.2 / 3.8, "loss": 7.9}.items():
        assert result[key] == pytest.approx(expected)
    assert sum(result[k] for k in ["tp", "fp", "fn", "tn"]) == pytest.approx(4)


def test_oracle_probabilities_equal_actual_metrics():
    y, pred = [1, 0, 1, 0, 0], [1, 1, 0, 0, 0]
    assert expected_confusion(y, pred, 3, 2) == actual_confusion(y, pred, 3, 2)


@pytest.mark.parametrize("y,pred,recall_defined,f1_defined", [
    ([0, 0], [0, 0], False, False), ([0, 0], [1, 0], False, True),
    ([1, 1], [0, 0], True, True),
])
def test_zero_denominators_are_explicit(y, pred, recall_defined, f1_defined):
    result = actual_confusion(y, pred, 1, 1)
    assert result["recall"] == 0
    assert result["f1"] == 0
    assert result["recall_defined"] == recall_defined
    assert result["f1_defined"] == f1_defined


@pytest.mark.parametrize("probability", [[-0.1], [1.1], [np.nan], [np.inf]])
def test_invalid_probability_rejected(probability):
    with pytest.raises(ValueError):
        expected_confusion(probability, [0], 1, 1)


def test_cost_ratio_changes_only_loss():
    equal = expected_confusion([0.7, 0.2], [0, 1], 1, 1)
    costly_fn = expected_confusion([0.7, 0.2], [0, 1], 10, 1)
    assert costly_fn["loss"] == pytest.approx(equal["loss"] + 9 * equal["fn"])
    for key in ["tp", "fp", "fn", "tn", "recall", "f1"]:
        assert equal[key] == costly_fn[key]
    with pytest.raises(ValueError):
        confusion_metrics(0, 0, 0, 0, -1, 1)


def test_density_ratio_corrects_unequal_domain_priors():
    assert density_ratio_from_probability([0.2], 400, 100, 1e-6)[0] == pytest.approx(1)
    assert density_ratio_from_probability([0.5], 400, 100, 1e-6)[0] == pytest.approx(4)
    assert np.isfinite(density_ratio_from_probability([0, 1], 10, 10, 1e-6)).all()


def test_ess_invariants():
    assert effective_sample_size(np.ones(7)) == pytest.approx(7)
    assert effective_sample_size([1, 2, 4]) == pytest.approx(49 / 21)
    assert effective_sample_size([10, 20, 40]) == pytest.approx(49 / 21)
    for invalid in [[], [0, 0], [1, -1], [np.nan]]:
        with pytest.raises(ValueError):
            effective_sample_size(invalid)
