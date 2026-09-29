"""Implementation behind scripts/run_pape_pilot.py --mode real (or audit)."""
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
import platform
import subprocess
import time
import pandas as pd

from src.pape.handoff import inspect_handoff, iter_handoff_target_runs, load_reference_pool
from src.pape.io import append_trial_logs, config_hash, utc_now, write_handoff_metadata, write_json
from src.pape.split import split_source_keys, group_selection_folds
from src.common.io import sha256_file


def _branch_info(project_root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(project_root), *args], text=True).strip()
    branch = git("branch", "--show-current")
    # Branch ownership is a development rule, not a runtime prerequisite.
    # Tags and reviewed integration branches must be executable too.
    return {"branch": branch, "head": git("rev-parse", "HEAD"),
            "working_tree_dirty": bool(git("status", "--porcelain"))}


def run_real_pilot(config, config_paths, output_dir, *, project_root, audit_only=False):
    from scripts.run_pape_pilot import (
        _run_one, aggregate_records, aggregate_loss_sensitivity, select_input,
        summarize_diagnostics, write_pilot_report,
    )
    if config["dre"].get("weight_estimation_method", "source_group_oof") != "source_group_oof":
        raise ValueError("Full DRE comparison requires scripts/run_pape_dre_comparison.py; "
                         "the normal pilot would rerun archived controls")
    if config.get("data_origin") != "model_adapter_handoff":
        raise ValueError("Load the real handoff profile for audit/real mode")
    git_info = _branch_info(project_root)
    data = config["pape_data"]
    window_path = Path(project_root) / data["window_features"]
    manifest_path = Path(project_root) / data["baseline_manifest"]
    audit = inspect_handoff(window_path, manifest_path, config)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Output already exists; choose another --output-dir: {output}")
    if audit_only:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "handoff_validation.json", {**audit, "git": git_info})
        print(f"Metadata audit passed: {output}", flush=True)
        return audit, output

    # Permission/protocol and geometric checks finish before reserving results.
    pool = load_reference_pool(window_path, config)
    seeds = config["experiment"]["seeds"]
    protocol = config["experiment"].get("selection_protocol", "single_split")
    if protocol not in {"single_split", "grouped_cv"}:
        raise ValueError("Unknown PAPE selection protocol")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Experiment seeds must be nonempty and unique")
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    records, assignments, seed_data, seed_choices, split_profiles = [], [], {}, {}, []
    selection_cv_manifest, validation_sources = [], {}
    audit["reference_pool"] = {
        "split": data["reference_split"], "run_ids": data["reference_run_ids"],
        "rows": len(pool), "source_keys": int(pool.source_key.nunique()),
        "records": int(pool.record_id.nunique()), "schema_and_geometry_valid": True,
    }
    write_json(output / "handoff_validation.json", audit)
    write_json(output / "run_status.json", {"status": "running", "git": git_info, "executed_at": utc_now()})

    def record_trial(source, target, chunk, seed, role, name, method, selection_fold=None):
        row, weights, probabilities = _run_one(source, target, chunk, seed,
            config["experiment"]["selection_case"], role, name, method, config)
        row["target_record_id"] = int(target.record_id.iloc[0])
        row["baseline_split"] = data["reference_split"]
        row["baseline_seed"] = audit["baseline_seed"]
        row["input_window_features_sha256"] = audit["window_features_sha256"]
        row["input_baseline_manifest_sha256"] = audit["baseline_manifest_sha256"]
        row["scenario_id"] = str(target.scenario_id.iloc[0])
        row["source_scenarios"] = sorted(source.scenario_id.unique().tolist())
        row["selection_fold"] = selection_fold
        append_trial_logs(output, weights, probabilities, first=not records)
        records.append(row)

    try:
        for seed in seeds:
            frames, assignment = split_source_keys(pool, config, seed)
            if protocol == "grouped_cv":
                development = pd.concat([frames["source"], frames["selection"]]).sort_index()
                selection_sets = list(group_selection_folds(development, config, seed))
                validation_sources[seed] = development
                for fold, source, target in selection_sets:
                    selection_cv_manifest.append({"seed": seed, "fold": fold,
                        "source_keys": sorted(source.source_key.unique()),
                        "target_keys": sorted(target.source_key.unique()),
                        "outer_validation_keys": sorted(frames["validation"].source_key.unique())})
            else:
                selection_sets = [(None, frames["source"], frames["selection"])]
                validation_sources[seed] = frames["source"]
            seed_data[seed] = frames
            assignments.extend(assignment.to_dict(orient="records"))
            split_profiles.append({"seed": seed, "roles": {
                role: {"source_key_count": int(frame.source_key.nunique()),
                       "record_count": int(frame.record_id.nunique()), "window_count": len(frame),
                       "source_keys_by_scenario": {
                           str(k): int(v) for k, v in frame.groupby("scenario_id").source_key.nunique().items()}}
                for role, frame in frames.items()}})
            names = config["experiment"].get("input_variants", list(config["pape"]["variants"]))
            if not names or len(set(names)) != len(names) or not set(names) <= set(config["pape"]["variants"]):
                raise ValueError("Experiment input_variants must name unique configured variants")
            for name in names:
                for fold, source, targets in selection_sets:
                    for chunk, target in iter_handoff_target_runs(targets):
                        for method in config["experiment"]["weightings"]:
                            record_trial(source, target, chunk, seed, "selection", name, method, fold)
                print(f"real selection seed={seed} input={name} complete", flush=True)
            seed_config = deepcopy(config)
            seed_config["experiment"]["seeds"] = [seed]
            seed_choices[seed] = select_input([r for r in records if r["seed"] == seed], seed_config)
        chosen = select_input(records, config)
        # The same real source_key can change roles across seeds. A pooled
        # winner would leak another seed's selection labels into validation.
        # Freeze a separate choice within each split; pooled ranking is descriptive.
        chosen["validation_candidate_by_seed"] = {
            str(seed): choice["selected_input_variant"] for seed, choice in seed_choices.items()}
        chosen["validation_protocol"] = "nested_per_seed_selection; pooled_ranking_never_controls_validation"
        chosen["selection_protocol"] = protocol
        print(f"Exploratory aggregate candidate: {chosen['selected_input_variant']}; seed stable={chosen['same_winner_all_seeds']}", flush=True)
        for seed, frames in seed_data.items():
            names = list(dict.fromkeys([config["pape"]["input_variant"], seed_choices[seed]["selected_input_variant"]]))
            for name in names:
                for chunk, target in iter_handoff_target_runs(frames["validation"]):
                    for method in config["experiment"]["weightings"]:
                        record_trial(validation_sources[seed], target, chunk, seed, "validation", name, method)
                print(f"real validation seed={seed} input={name} complete", flush=True)
        summary = {
            "schema_reference": "엔스퀘어 과제 구현 및 업무 분담.pdf v1.0 sections 4.2, 5.1, 5.2, 6, 8.1",
            "model_version": config["model_version"], "executed_at": utc_now(),
            "seeds": seeds, "config_hash": config_hash(config), "git": git_info,
            "study_stage": config["experiment"].get("study_stage", "stage2_exploratory_pilot"),
            "evaluation_status": config["experiment"].get("evaluation_status", "exploratory_reused_baseline_validation"),
            "prior_validation_inspected": config["experiment"].get("prior_validation_inspected", False),
            "randomness": config.get("randomness", {}),
            "data_origin": "model_adapter_handoff", "real_hybrid_model_executed": False,
            "real_hybrid_outputs_used": True, "baseline_f1_reproduced_by_pape": False,
            "official_test_used": False, "official_train_used": False,
            "target_labels_used_during_estimation": False,
            "reference_protocol": {**data, "independent_final_evaluation": False,
                "reason": "Baseline Validation was already used for fusion/threshold selection; results are exploratory."},
            "handoff_validation": audit, "selection": chosen,
            "split_profiles": split_profiles,
            "selection_cv_manifest": selection_cv_manifest,
            "split_manifest_role_semantics": ("outer reservation: source+selection form inner-CV development; validation stays held out"
                                              if protocol == "grouped_cv" else "direct Source, selection Target, validation Target roles"),
            "code_file_sha256": {
                path.relative_to(project_root).as_posix(): sha256_file(path)
                for folder in ["src", "scripts"]
                for path in sorted((Path(project_root) / folder).rglob("*.py"))},
            "metrics": aggregate_records(records, ["case", "split", "input_name", "weighting"]),
            "metrics_by_seed": aggregate_records(records, ["case", "split", "seed", "input_name", "weighting"]),
            "loss_sensitivity_metrics": aggregate_loss_sensitivity(records),
            "diagnostics_by_case": summarize_diagnostics(records),
            "split_manifest": assignments, "loss_definition": config["loss"],
            "csv_contract": {"schema_unchanged": True, "input_variant_encoding": "{input_name}__{uniform|dre}",
                             "record_mapping": "records[i] is zero-based CSV data row i (header excluded)",
                             "uniform_dre_auc": "null in JSON, empty in CSV"},
            "failure_conditions": [
                "기준 모델이 이미 보정에 사용한 Validation을 재사용하므로 독립적인 최종 성능 평가가 아닙니다.",
                "제공 데이터는 시뮬레이션 비행 원본과 시계열 변조 합성 이상입니다. 실제 기체 고장 성능 검증이 아닙니다.",
                "Validation은 18개 원본 그룹뿐이며 A1은 1개입니다. Target 시나리오가 Source에 없으면 별도 경고를 기록합니다.",
                "Source/Target이 baseline Validation의 서로 다른 환경입니다. 실제 시간순 운영 스트림 또는 covariate shift임이 검증되지 않았습니다.",
                "Concept shift(P(y|x) 변화)는 라벨 없는 PAPE로 신뢰성 있게 식별하거나 보정할 수 없습니다.",
                "Target이 Source support 밖에 있으면 보정 근거가 부족합니다. 범위 진단만으로 다변량 support를 증명하지 못합니다.",
                "낮은 ESS·과도한 clipping은 추정의 분산·편향을 키울 수 있습니다. 경고를 유지하며 안정성이 확보됐다고 주장하지 않습니다.",
                "Hysteresis·context threshold로 나뉜 판정 부분집합의 calibration은 전체 점수 calibration과 다를 수 있습니다.",
                "Recall 분모가 0인 Run의 0 처리와 실제 양성이 있는 Run만의 MAE를 구분해야 합니다.",
                "중첩 Window는 독립 표본이 아니므로 이 Pilot은 IID 신뢰구간을 제공하지 않습니다.",
                "seed별 반복에서 같은 source_key가 다시 사용됩니다. 검증 후보는 각 seed 내부 선정 결과로만 고정하며 반복 결과는 독립 반복 표본이 아닙니다.",
                "문서 기준 F1과 인계 manifest의 재현 F1 차이는 담당자의 기록을 보존하며 원인을 단정하지 않습니다.",
            ],
            "implementation_notes": [
                "공통 schemas.py와 모델 연동 코드는 변경하지 않고 PAPE 전용 검사로 연결했습니다.",
                "인계 window_id를 그대로 보존하며 record_id 하나를 Target Run 하나로 처리합니다.",
                "원본 시각·점수·판정·라벨을 수정하지 않습니다. 첫/마지막 샘플 시각은 4.9초 간격입니다.",
                "허용된 경우 마지막 한 stride만 짧을 수 있으며 중간 누락/비정상 stride는 거부합니다.",
                "DRE는 Source source_key OOF 및 Target 시간 블록 검증에서 겹치는 Window를 제거합니다.",
                "PAPE는 현재 성능/손실 추정입니다. 미래 Gain과 RCCDA 정책은 다른 담당자의 범위입니다.",
            ],
            "environment": {"python": platform.python_version(), "platform": platform.system(),
                "packages": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "lightgbm", "PyYAML", "pyarrow", "pytest"]}},
            "duration_seconds": time.perf_counter() - start,
        }
        write_handoff_metadata(output, records, summary, config, config_paths)
        write_json(output / "split_manifest.json", {"config_hash": config_hash(config), "assignments": assignments})
        write_pilot_report(summary, output)
        write_json(output / "run_status.json", {"status": "complete", "records": len(records), "executed_at": utc_now(), "git": git_info})
        print(f"Saved {len(records)} real PAPE estimates to {output}", flush=True)
        return summary, output
    except Exception as exc:
        write_json(output / "run_status.json", {"status": "failed", "completed_trials": len(records),
                   "error": str(exc), "executed_at": utc_now(), "git": git_info})
        raise
