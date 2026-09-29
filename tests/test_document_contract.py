import numpy as np
import pandas as pd

from scripts.verify_pape_outputs import check_document_contract
from src.pape.split import density_ratio_folds


def test_document_column_names_file_names_and_public_interface():
    check_document_contract()


def test_inclusive_endpoints_never_share_a_sample_between_dre_folds():
    source = pd.DataFrame({"source_key": np.repeat([f"source_{i}" for i in range(6)], 10)})
    target = pd.DataFrame({"window_start": np.arange(30, dtype=float),
                           "window_end": np.arange(30, dtype=float) + 4})
    for _, _, _, train, holdout in density_ratio_folds(source, target, 3, 11, end_inclusive=True):
        a, b = target.iloc[train], target.iloc[holdout]
        shares_sample = ((a.window_start.to_numpy()[:, None] <= b.window_end.to_numpy()) &
                         (a.window_end.to_numpy()[:, None] >= b.window_start.to_numpy()))
        assert not shares_sample.any()
