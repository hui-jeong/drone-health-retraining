from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FLOAT_COLUMNS = {
    "ae_score": "ae_score",
    "anomaly_fraction": "anomaly_fraction",
    "residual_probability": "residual_probability",
    "ae_percentile_score": "ae_percentile_score",
    "score": "hybrid_score",
    "effective_threshold": "context_threshold",
}

INT_COLUMNS = {
    "y_true": "y_true",
    "raw_prediction": "raw_prediction",
    "prediction": "final_prediction",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify exported test Window Features against the company's test_window_predictions.csv"
    )
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--company-csv", required=True)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()

    parquet_path = Path(args.parquet).resolve()
    company_path = Path(args.company_csv).resolve()

    exported = pd.read_parquet(parquet_path)
    company = pd.read_csv(company_path, dtype={"run_id": str})

    keys = ["table_path", "start_idx"]
    for key in keys:
        if key not in exported.columns:
            raise KeyError(f"Exported parquet is missing key column: {key}")
        if key not in company.columns:
            raise KeyError(f"Company CSV is missing key column: {key}")

    result = {
        "exported_rows": int(len(exported)),
        "company_rows": int(len(company)),
        "exported_duplicate_keys": int(exported.duplicated(keys).sum()),
        "company_duplicate_keys": int(company.duplicated(keys).sum()),
        "joined_rows": 0,
        "unmatched_exported_rows": 0,
        "unmatched_company_rows": 0,
        "integer_mismatches": {},
        "float_comparisons": {},
        "pass": False,
    }

    exported_keys = exported[keys].drop_duplicates()
    company_keys = company[keys].drop_duplicates()

    result["unmatched_exported_rows"] = int(
        len(exported_keys.merge(company_keys, on=keys, how="left", indicator=True).query("_merge != 'both'"))
    )
    result["unmatched_company_rows"] = int(
        len(company_keys.merge(exported_keys, on=keys, how="left", indicator=True).query("_merge != 'both'"))
    )

    exported_renamed = exported.rename(
        columns={v: f"export__{v}" for v in list(FLOAT_COLUMNS.values()) + list(INT_COLUMNS.values())}
    )

    company_keep = keys + list(FLOAT_COLUMNS.keys()) + list(INT_COLUMNS.keys())
    company_small = company[company_keep].rename(
        columns={c: f"company__{c}" for c in list(FLOAT_COLUMNS.keys()) + list(INT_COLUMNS.keys())}
    )

    merged = exported_renamed.merge(company_small, on=keys, how="inner", validate="one_to_one")
    result["joined_rows"] = int(len(merged))

    all_ok = (
        len(exported) == len(company)
        and result["exported_duplicate_keys"] == 0
        and result["company_duplicate_keys"] == 0
        and result["unmatched_exported_rows"] == 0
        and result["unmatched_company_rows"] == 0
        and len(merged) == len(company)
    )

    for company_col, export_col in INT_COLUMNS.items():
        left = pd.to_numeric(merged[f"export__{export_col}"], errors="coerce")
        right = pd.to_numeric(merged[f"company__{company_col}"], errors="coerce")
        mismatch = ~(left.eq(right) | (left.isna() & right.isna()))
        count = int(mismatch.sum())
        result["integer_mismatches"][f"{company_col}->{export_col}"] = count
        all_ok = all_ok and count == 0

    for company_col, export_col in FLOAT_COLUMNS.items():
        left = pd.to_numeric(merged[f"export__{export_col}"], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(merged[f"company__{company_col}"], errors="coerce").to_numpy(dtype=float)

        both_nan = np.isnan(left) & np.isnan(right)
        finite_mask = np.isfinite(left) & np.isfinite(right)
        close = np.zeros(len(left), dtype=bool)
        close[both_nan] = True
        close[finite_mask] = np.isclose(
            left[finite_mask], right[finite_mask], atol=args.atol, rtol=args.rtol
        )

        mismatch_count = int((~close).sum())
        abs_diff = np.abs(left - right)
        finite_diff = abs_diff[np.isfinite(abs_diff)]
        max_abs_diff = float(finite_diff.max()) if finite_diff.size else 0.0
        mean_abs_diff = float(finite_diff.mean()) if finite_diff.size else 0.0

        result["float_comparisons"][f"{company_col}->{export_col}"] = {
            "mismatch_count": mismatch_count,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
        }
        all_ok = all_ok and mismatch_count == 0

    result["pass"] = bool(all_ok)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if not all_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
