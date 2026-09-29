"""Configuration and external diagnostic evidence for the frozen PAPE release."""
from copy import deepcopy
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil

import yaml

from src.common.metrics import confusion_metrics
from .io import config_hash, write_json
from .release import sha256


def export_settings(original, source, seed=None):
    resolved = yaml.safe_load((Path(original) / "config/resolved.yaml").read_text(encoding="utf-8"))
    if config_hash(resolved) != source["config_hash"]:
        raise ValueError("Original resolved configuration hash mismatch")
    costs = resolved["loss"]
    zero = resolved["metrics"]["zero_division"]
    confusion_metrics(0, 0, 0, 0, costs["lambda_fn"], costs["lambda_fp"], zero)
    selected_seed = resolved["seed"] if seed is None else seed
    if selected_seed not in source["seeds"]:
        raise ValueError("Selected seed was not run in the archived experiment")
    return resolved, selected_seed


def portable_config(resolved):
    """Keep every setting; replace only local paths and disclose each replacement."""
    changes = []

    def visit(value, keys):
        if isinstance(value, dict):
            return {k: visit(v, keys + [str(k)]) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(v, keys + [str(i)]) for i, v in enumerate(value)]
        if isinstance(value, str) and (PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()):
            # Keep private source paths outside the portable package and Git.
            replacement = "external_inputs/" + "/".join(keys) + "/" + PureWindowsPath(value).name
            changes.append({"key": ".".join(keys), "replacement": replacement})
            return replacement
        return deepcopy(value)

    return visit(resolved, []), changes


def write_reference_artifacts(original, output_root, rows):
    """Copy and hash-check large logs separately from the two small ZIPs."""
    original = Path(original).resolve()
    root = Path(output_root) / "PAPE_reference_artifacts_version_1.0"
    root.mkdir(exist_ok=False)
    files = {}

    def copy_verified(relative, expected=None):
        source = (original / relative).resolve()
        if not source.is_relative_to(original) or not source.is_file():
            raise ValueError("Invalid diagnostic artifact path: " + relative)
        digest = sha256(source)
        if expected is not None and digest != expected:
            raise ValueError("Diagnostic artifact hash mismatch: " + relative)
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha256(destination) != digest:
            raise ValueError("Diagnostic artifact copy failed: " + relative)
        files[relative] = {"sha256": digest, "bytes": destination.stat().st_size}

    configs = {}
    mapping = []
    for row in rows:
        effective = row["effective_config"]
        if config_hash(effective) != row["config_hash"]:
            raise ValueError("Original effective configuration hash mismatch")
        configs[row["config_hash"]] = effective
        artifacts = {}
        for kind, artifact in row["trial_artifacts"].items():
            relative = f"trials/{row['seed']}/{row['input_name']}/{row['weighting']}/{artifact['file']}"
            if relative not in files:
                copy_verified(relative, artifact["sha256"])
            elif files[relative]["sha256"] != artifact["sha256"]:
                raise ValueError("Conflicting diagnostic hashes")
            artifacts[kind] = {"path": relative, **files[relative]}
        mapping.append({k: row[k] for k in ["csv_row_index", "target_chunk_id", "seed", "input_variant", "config_hash"]}
                       | {"artifacts": artifacts})
    for path in sorted((original / "config").iterdir()):
        if path.is_file():
            copy_verified(path.relative_to(original).as_posix())
    config_file = root / "config/original_effective_configs.json"
    write_json(config_file, configs)
    files[config_file.relative_to(root).as_posix()] = {"sha256": sha256(config_file), "bytes": config_file.stat().st_size}
    manifest = {"artifact_kind": "archived_pape_reference_evidence", "files": files, "records": mapping,
                "contains_original_local_paths": True, "target_labels_used_for_estimation": False}
    write_json(root / "manifest.json", manifest)
    (root / "README.md").write_text(
        "# PAPE 원본 진단 자료\n\n두 인계 ZIP과 같은 폴더에 보관합니다. "
        "trials/에는 원시·clipping 가중치와 Target 확률이 있습니다. config/에는 실제 사용한 YAML과 "
        "행별 effective config가 있습니다. 원본 설정에는 작성 당시 개인 PC 경로가 포함됩니다. "
        "이 폴더는 Git에 올리지 않습니다. 재실행에는 각 인계 ZIP의 이식 가능한 설정을 사용합니다.\n",
        encoding="utf-8")
    return {"directory": root.name, "manifest_sha256": sha256(root / "manifest.json"),
            "files": files, "records": mapping}
