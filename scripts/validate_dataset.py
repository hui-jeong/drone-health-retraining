from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io import load_integration_config, resolve_config_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate company v2.3 dataset against retraining-project contract")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "base.yaml"))
    parser.add_argument("--local-config", default=None)
    args = parser.parse_args()

    cfg = load_integration_config(args.config, args.local_config)
    paths = cfg["paths"]
    company_root = resolve_config_path(paths["company_model_root"], args.config)
    company_cfg_path = resolve_config_path(paths["company_config"], args.config)
    data_root = resolve_config_path(paths["data_root"], args.config)

    sys.path.insert(0, str(company_root))
    from drone_ae.discovery import discover_records
    from drone_ae.preparation import load_feature_columns
    from drone_ae.utils import load_yaml

    company_cfg = load_yaml(company_cfg_path)
    records = discover_records(data_root, company_cfg)
    features = load_feature_columns(data_root, company_cfg)
    expected = cfg.get("baseline", {})

    unique_sources = records.drop_duplicates("source_key")
    normal_count = int((records["data_class"] == "normal").sum())
    abnormal_count = int((records["data_class"] == "abnormal").sum())
    leakage = int((records.groupby("source_key")["split"].nunique() > 1).sum())
    pair_counts = records.groupby("source_key")["data_class"].nunique()
    incomplete_pairs = int((pair_counts < 2).sum())

    checks = {
        "model_ready_files": {"actual": int(len(records)), "expected": expected.get("expected_model_ready_files")},
        "source_runs": {"actual": int(records["source_key"].nunique()), "expected": expected.get("expected_source_runs")},
        "normal_files": {"actual": normal_count, "expected": expected.get("expected_normal_files")},
        "abnormal_files": {"actual": abnormal_count, "expected": expected.get("expected_abnormal_files")},
        "source_feature_count": {"actual": int(len(features)), "expected": expected.get("expected_source_feature_count")},
        "split_leakage_source_keys": {"actual": leakage, "expected": 0},
        "incomplete_normal_abnormal_source_pairs": {"actual": incomplete_pairs, "expected": 0},
    }
    for value in checks.values():
        value["pass"] = value["expected"] is None or value["actual"] == value["expected"]

    payload = {
        "data_root": str(data_root),
        "checks": checks,
        "run_ids_by_split": {
            split: sorted(group["run_id"].astype(str).unique().tolist())
            for split, group in unique_sources.groupby("split")
        },
        "data_class_counts": records["data_class"].value_counts().to_dict(),
        "scenario_counts": records.groupby(["scenario_id", "data_class"]).size().astype(int).to_dict(),
        "valid": all(v["pass"] for v in checks.values()),
    }

    print("=" * 72)
    print("DATASET CONTRACT VALIDATION")
    print("=" * 72)
    for name, value in checks.items():
        status = "PASS" if value["pass"] else "FAIL"
        print(f"{name:42s}: {value['actual']} (expected {value['expected']}) [{status}]")
    print("-" * 72)
    print("Run IDs by split:")
    print(json.dumps(payload["run_ids_by_split"], ensure_ascii=False, indent=2))
    print("=" * 72)
    if not payload["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
