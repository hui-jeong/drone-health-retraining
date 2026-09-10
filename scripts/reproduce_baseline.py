from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io import load_integration_config, resolve_config_path, sha256_file, write_json


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _environment() -> dict:
    payload = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    try:
        import torch
        payload.update({
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
    except Exception as exc:
        payload["torch_error"] = repr(exc)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run/read company v2.3 baseline and write baseline_manifest.json")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "base.yaml"))
    parser.add_argument("--local-config", default=None)
    parser.add_argument("--run-all", action="store_true", help="Actually invoke company run_all.py before collecting metrics")
    args = parser.parse_args()

    cfg = load_integration_config(args.config, args.local_config)
    p = cfg["paths"]
    company_root = resolve_config_path(p["company_model_root"], args.config)
    company_cfg = resolve_config_path(p["company_config"], args.config)
    data_root = resolve_config_path(p["data_root"], args.config)
    output_root = resolve_config_path(p["model_output_root"], args.config)
    manifest_path = resolve_config_path(p["baseline_manifest"], args.config)

    elapsed = None
    command = [
        sys.executable,
        str(company_root / "run_all.py"),
        "--config", str(company_cfg),
        "--data-root", str(data_root),
        "--output-root", str(output_root),
    ]
    if args.run_all:
        start = time.perf_counter()
        subprocess.run(command, cwd=company_root, check=True)
        elapsed = time.perf_counter() - start

    window_path = output_root / "reports" / "test_metrics.json"
    run_path = output_root / "reports" / "test_run_metrics.json"
    prep_path = output_root / "artifacts" / "preparation_summary.json"
    required = [window_path, run_path, prep_path]
    missing = [str(x) for x in required if not x.is_file()]
    if missing:
        raise FileNotFoundError(
            "Baseline outputs are missing. Use --run-all after setting data paths, or provide an existing output folder:\n"
            + "\n".join(missing)
        )

    window_metrics = _read_json(window_path)
    run_metrics = _read_json(run_path)
    prep = _read_json(prep_path)
    expected = cfg.get("baseline", {})
    tol = float(expected.get("metric_tolerance", 0.005))
    window_f1 = float(window_metrics["f1"])
    run_f1 = float(run_metrics["f1"])
    expected_window_f1 = float(expected.get("expected_window_f1", window_f1))
    expected_run_f1 = float(expected.get("expected_run_f1", run_f1))

    payload = {
        "model_version": "Hybrid LSTM-AE v2.3",
        "integration_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": cfg.get("project", {}).get("seed", 42),
        "company_config": str(company_cfg),
        "company_config_sha256": sha256_file(company_cfg),
        "data_root": str(data_root),
        "model_output_root": str(output_root),
        "command": command,
        "run_all_executed": bool(args.run_all),
        "execution_time_sec": elapsed,
        "source_feature_count": prep.get("source_feature_count"),
        "model_feature_count": prep.get("feature_count"),
        "dropped_zero_variance_features": prep.get("dropped_zero_variance_features", []),
        "window_metrics": window_metrics,
        "run_metrics": run_metrics,
        "baseline_comparison": {
            "expected_window_f1": expected_window_f1,
            "actual_window_f1": window_f1,
            "window_f1_delta": window_f1 - expected_window_f1,
            "window_f1_within_tolerance": abs(window_f1 - expected_window_f1) <= tol,
            "expected_run_f1": expected_run_f1,
            "actual_run_f1": run_f1,
            "run_f1_delta": run_f1 - expected_run_f1,
            "run_f1_within_tolerance": abs(run_f1 - expected_run_f1) <= tol,
            "tolerance": tol,
        },
        "environment": _environment(),
    }
    write_json(manifest_path, payload)
    print(json.dumps(payload["baseline_comparison"], indent=2))
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()
