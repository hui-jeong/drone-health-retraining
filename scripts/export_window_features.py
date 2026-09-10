from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io import load_integration_config, resolve_config_path
from src.feature_export.export_window_features import export_window_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Export common Window Feature parquet from company v2.3")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "base.yaml"))
    parser.add_argument("--local-config", default=None)
    parser.add_argument("--split", choices=["train", "validation", "test"], default=None)
    args = parser.parse_args()

    cfg = load_integration_config(args.config, args.local_config)
    paths = cfg["paths"]
    export_cfg = cfg.get("export", {})

    output, validation = export_window_features(
        company_model_root=resolve_config_path(paths["company_model_root"], args.config),
        company_config_path=resolve_config_path(paths["company_config"], args.config),
        model_output_root=resolve_config_path(paths["model_output_root"], args.config),
        output_path=resolve_config_path(paths["window_features"], args.config),
        split=args.split,
        window_id_strategy=str(export_cfg.get("window_id_strategy", "source_record_start")),
        latent_size=int(export_cfg.get("latent_size", 32)),
        compression=str(export_cfg.get("output_compression", "snappy")),
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
