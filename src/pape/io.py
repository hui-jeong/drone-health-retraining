"""YAML configuration and strict, traceable experiment output."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import yaml

from .contracts import PAPE_ESTIMATE_COLUMNS, STRING_COLUMNS


def merge_config(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(*paths):
    result = {}
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            content = yaml.safe_load(stream)
        if not isinstance(content, dict):
            raise ValueError(f"Expected a YAML mapping: {path}")
        result = merge_config(result, content)
    return result


def config_hash(config):
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported JSON value: {type(value)}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False,
                  default=_json_default)
        stream.write("\n")


def read_window_features(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={c: "string" for c in STRING_COLUMNS})
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("Window Feature input must be .csv or .parquet")


def write_handoff_metadata(output_dir, records, summary, config, config_paths):
    """Finalize an output directory already reserved by the current experiment."""
    output_dir = Path(output_dir)
    for i, record in enumerate(records):
        record["csv_row_index"] = i
    pd.DataFrame(records).loc[:, PAPE_ESTIMATE_COLUMNS].to_csv(
        output_dir / "pape_estimates.csv", index=False, encoding="utf-8-sig")
    summary["records"] = records
    write_json(output_dir / "pape_summary.json", summary)
    config_dir = output_dir / "config"
    config_dir.mkdir()
    for path in config_paths:
        shutil.copyfile(path, config_dir / Path(path).name)
    with (config_dir / "resolved.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
    write_json(config_dir / "config_hash.json", {"sha256": config_hash(config)})
    return output_dir


def save_handoff(output_dir, records, summary, weights, probabilities, config, config_paths):
    """Use the exact shared CSV schema; rich records are indexed in the JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_handoff_metadata(output_dir, records, summary, config, config_paths)
    weights.to_csv(output_dir / "source_weights.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")
    probabilities.to_csv(output_dir / "target_probabilities.csv", index=False, encoding="utf-8-sig")
    return output_dir


def append_trial_logs(output_dir, weights, probabilities, *, first):
    """Stream real-data logs to keep memory bounded by one trial, not the whole pilot."""
    output_dir = Path(output_dir)
    # Uniform trials have no domain-classifier diagnostics. Keep empty cells for
    # those fields so every appended member has the same CSV column order.
    weight_columns = [
        "window_id", "source_key", "run_id", "fold", "domain_probability",
        "prior_source_over_target", "raw_weight", "clipped_weight", "was_clipped",
        "target_chunk_id", "input_variant", "case", "split", "seed",
        "config_hash", "model_version", "executed_at",
    ]
    weights = weights.reindex(columns=weight_columns)
    for table, name in [(weights, "source_weights.csv.gz"), (probabilities, "target_probabilities.csv.gz")]:
        table.to_csv(output_dir / name, mode="w" if first else "a", header=first,
                     index=False, encoding="utf-8", compression="gzip")
