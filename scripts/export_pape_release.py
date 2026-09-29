"""Package completed 007 PAPE results and a clearly marked actual-value comparator.

No model fitting, inference, threshold change or Test read is performed.
The Target input must be the pre-separated 007 export, identified by its hash.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import yaml

from src.common.metrics import actual_confusion
from src.pape.contracts import PAPE_ESTIMATE_COLUMNS
from src.pape.io import config_hash, utc_now, write_json
from src.pape.release import VERSION, FEATURES, read_release, load_release_inputs, sha256
from src.pape.release_export import export_settings, portable_config, write_reference_artifacts

METRICS = ["tp", "fp", "fn", "tn", "recall", "f1", "loss"]
PUBLIC = list(dict.fromkeys(PAPE_ESTIMATE_COLUMNS + FEATURES + ["actual_"+k for k in METRICS] + [k+"_absolute_error" for k in METRICS] + [
    "seed", "source_key", "run_id", "scenario_id", "target_record_id", "n_source", "n_target",
    "model_version", "executed_at", "config_hash", "study_config_hash", "input_name", "weighting",
    "warnings", "readiness", "target_labels_used", "csv_row_index", "split", "ess", "raw_ess",
    "max_raw_weight", "clip_fraction", "dre_auc_aggregation", "actual_recall_defined", "actual_f1_defined",
    "estimated_recall_defined", "estimated_f1_defined", "source_windows_in_target_calibration_box",
    "source_positive_windows_in_target_calibration_box", "source_positive_keys_in_target_calibration_box"] ))


def csv(path, records, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).loc[:, columns].to_csv(path, index=False, encoding="utf-8-sig")


def compare(a, b):
    if a is None:
        return pd.isna(b)
    return a == b if isinstance(a, str) else bool(np.isclose(a, b, rtol=1e-9, atol=1e-9))


def package(experiment_dir, target_features, output_dir, *, seed=None, code_revision=None):
    original = Path(experiment_dir)
    summary_path = original / "pape_summary.json"
    source = json.loads(summary_path.read_text(encoding="utf-8"))
    resolved, selected_seed = export_settings(original, source, seed)
    costs = resolved["loss"]
    zero_division = resolved["metrics"]["zero_division"]
    original_csv = original / "pape_estimates.csv"
    table = pd.read_csv(original_csv)
    status = json.loads((original / "run_status.json").read_text(encoding="utf-8"))
    verified = json.loads((original / "verification_report.json").read_text(encoding="utf-8"))
    roles = source["roles"]
    if (roles != {"model_fit": [], "model_calibration": [], "pape_source": [f"{i:03d}" for i in range(7)], "pape_validation": ["007"]}
            or source["official_test_used"] is not False
            or source["target_labels_used_in_estimation"] is not False
            or status["status"] != "complete" or status["config_hash"] != source["config_hash"]
            or verified["status"] != "PASS" or verified["trial_artifact_hashes_valid"] is not True
            or verified["test_used"] is not False):
        raise ValueError("Expected completed, verified frozen 000-006 -> 007 experiment")
    if sha256(target_features) != source["model_provenance"]["exports"]["pape_validation"]["sha256"]:
        raise ValueError("Input is not the pre-separated, verified 007 Target export")
    target = pd.read_parquet(target_features)
    if set(target.run_id) != {"007"} or not target.window_id.is_unique:
        raise ValueError("Only unique 007 validation windows are allowed")
    rows = source["records"]
    if len(rows) != 1440 or len(table) != 1440 or list(table.columns) != PAPE_ESTIMATE_COLUMNS:
        raise ValueError("Expected original 1,440-row shared handoff")
    pins = {str(summary_path): sha256(summary_path), str(original_csv): sha256(original_csv),
            str(Path(target_features)): sha256(target_features),
            str(original / "config/resolved.yaml"): sha256(original / "config/resolved.yaml")}
    for index, row in enumerate(rows):
        if row["csv_row_index"] != index or row["run_id"] != "007" or row["target_labels_used"] is not False:
            raise ValueError("Unsafe original row")
        if any(not compare(row[key], table.iloc[index][key]) for key in PAPE_ESTIMATE_COLUMNS):
            raise ValueError("Original CSV and summary disagree")
    selected = sorted([r for r in rows if r["seed"] == selected_seed and r["input_variant"] == "hybrid_score__dre"], key=lambda r:r["target_record_id"])
    if len(selected) != 36:
        raise ValueError("Expected one default estimate for each of 36 Chunks")
    actual_rows=[]; chunk_rows=[]
    for row in selected:
        group = target[target.record_id.eq(row["target_record_id"])].sort_values("window_start")
        if len(group) != row["n_target"]:
            raise ValueError("Target count changed")
        a=actual_confusion(group.y_true,group.final_prediction,costs["lambda_fn"],costs["lambda_fp"],zero_division)
        if any(not compare(row["actual_"+k],a[k]) for k in METRICS):
            raise ValueError("Saved actual values differ from frozen detector output")
        actual_rows.append({"target_chunk_id":row["target_chunk_id"], "source_key":row["source_key"],
            "run_id":"007", "scenario_id":row["scenario_id"], "record_id":row["target_record_id"],
            "n_windows":len(group), **{"actual_"+k:a[k] for k in METRICS},
            "actual_recall_defined":a["recall_defined"],"actual_f1_defined":a["f1_defined"]})
        chunk_rows.append({"target_chunk_id":row["target_chunk_id"], "source_key":row["source_key"],
            "run_id":"007", "scenario_id":row["scenario_id"], "record_id":row["target_record_id"],
            "n_windows":len(group), "window_start":float(group.window_start.min()),
            "window_end":float(group.window_end.max()), "chronology_verified":False})
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=False)
    created=utc_now();packages=[]
    portable, path_changes = portable_config(resolved)
    diagnostics_manifest = write_reference_artifacts(original, root, rows)
    effective_configs = {}
    for row in rows:
        effective, replacements = portable_config(row["effective_config"])
        effective_configs[row["config_hash"]] = {
            "portable_config_hash": config_hash(effective), "path_replacements": replacements, "config": effective}
    common=dict(release_version=VERSION,created_at=created,model_version=source["model_version"],roles=roles,
        original_study_config_hash=source["config_hash"],original_executed_at=source["executed_at"],
        code_revision=code_revision,official_test_used=False,detector_refit=False,pape_refit=False,
        chronology_verified=False,independent_final_evaluation=False,production_approved=False,
        loss_definition={key:deepcopy(costs[key]) for key in ["status","lambda_fn","lambda_fp"]},
        metrics=deepcopy(resolved["metrics"]),
        metric_definition="Per-Chunk counts; undefined ratios use metrics.zero_division with separate defined flags.",
        export_selection={"configured_seed":resolved["seed"],"seed_override":seed,"selected_seed":selected_seed},
        upstream_hashes={"pape_summary.json":pins[str(summary_path)],"pape_estimates.csv":pins[str(original_csv)],"target007_features":pins[str(Path(target_features))]},
        checkpoint_identity_verified=False)
    for oracle,name in [(False,"PAPE_version_1.0_007_validation"),(True,"LSTM_AE_true_data_version_1.0_007")]:
        out=root/name;out.mkdir()
        packaged=[]
        for index,row in enumerate(selected):
            r={key:deepcopy(row[key]) for key in PUBLIC if key in row}
            r["csv_row_index"]=index
            if oracle:
                r.update(input_variant="lstm_ae_true_data__oracle",input_name="lstm_ae_true_data",
                         weighting="not_applicable",seed=None,target_labels_used=True,
                         dre_auc=None,ess_ratio=None,max_weight=None,
                         warnings=["target_labels_used_offline_oracle_only"],
                         readiness={"status":"oracle_comparison_only","production_approved":False,"policy_action":None})
                for key in METRICS:
                    r["estimated_"+key]=r["actual_"+key]
                    r[key+"_absolute_error"]=0.0
                r["estimated_recall_defined"]=r["actual_recall_defined"]
                r["estimated_f1_defined"]=r["actual_f1_defined"]
                for key in ["ess","raw_ess","max_raw_weight","clip_fraction","dre_auc_aggregation","source_windows_in_target_calibration_box","source_positive_windows_in_target_calibration_box","source_positive_keys_in_target_calibration_box"]:
                    r.pop(key,None)
                # These are original comparator provenance, not oracle fitting hashes.
                r["reference_pape_config_hash"]=r.pop("config_hash")
            packaged.append(r)
        kind="lstm_ae_true_data" if oracle else "pape_estimates"
        summary={**common,"artifact_kind":kind,"target_labels_used_for_inputs":oracle,"records":packaged,
            "bundle_id":f"{VERSION}:{kind}:{source['config_hash']}:{selected_seed}:{code_revision or 'unrecorded'}",
            "selection_status":"oracle_comparator" if oracle else "reference_default_final_selection_pending",
            "selected_input":"lstm_ae_true_data__oracle" if oracle else "hybrid_score__dre",
            "selected_seed":None if oracle else selected_seed,
            "selection_reason":"Actual fixed-Hybrid values for an explicitly label-derived offline comparator." if oracle else
                f"Hybrid is the work-division default; seed {selected_seed} is an archived reference, not a validation-selected winner.",
            "known_failures":["007 A1 has actual FN=39, while PAPE Hybrid+DRE estimates approximately zero.",
                              "No verified flight chronology or new-model future loss is supplied.",
                              "Frozen-export hashes verified; exact detector checkpoint lineage is unverified."]}
        csv(out/"pape_estimates.csv",packaged,PAPE_ESTIMATE_COLUMNS)
        write_json(out/"pape_summary.json",summary)
        csv(out/"chunk_index.csv",chunk_rows,list(chunk_rows[0]))
        (out/'config').mkdir()
        (out/'config/resolved_portable.yaml').write_text(yaml.safe_dump(portable,allow_unicode=True,sort_keys=False),encoding='utf-8')
        write_json(out/'config/effective_configs_portable.json',effective_configs)
        write_json(out/'config/provenance.json',{'original_study_config_hash':source['config_hash'],
            'portable_config_hash':config_hash(portable), 'path_replacements':path_changes,
            'original_config_directory':'../'+diagnostics_manifest['directory']+'/config',
            'effective_config_mapping':'effective_configs_portable.json',
            'changes':'All settings retained; absolute paths replaced. Original and portable hashes are distinct.'})
        write_json(out/'diagnostics_manifest.json',diagnostics_manifest)
        if oracle:
            csv(out/"lstm_ae_true_data.csv",actual_rows,list(actual_rows[0]))
        else:
            comparisons=[{key:deepcopy(r[key]) for key in PUBLIC if key in r} for r in rows]
            csv(out/"comparisons/pape_estimates.csv",comparisons,PAPE_ESTIMATE_COLUMNS)
            write_json(out/"comparisons/pape_summary.json",{**common,"artifact_kind":"comparison_not_policy_stream", "records":comparisons,
                "seeds":source["seeds"],"metrics":source["metrics"],"seed_claims":source["seed_claims"],
                "loss_sensitivity_metrics":source["loss_sensitivity_metrics"],
                "target_labels_used_during_estimation":False})
            sensitivity = [{**{k:r[k] for k in ["target_chunk_id","seed","input_variant","csv_row_index"]},**v}
                           for r in rows for v in r["loss_sensitivity"]]
            if sensitivity:
                csv(out/"comparisons/loss_sensitivity.csv",sensitivity,list(sensitivity[0]))
            write_json(out/"comparisons/loss_sensitivity_metrics.json",source["loss_sensitivity_metrics"])
        readme=("# LSTM_AE_true_data: 실제값 비교 자료\n\n" if oracle else "# PAPE version_1.0: 007 validation 인계\n\n")
        readme += "두 묶음은 동일한 007의 36개 Chunk와 기존 Hybrid 최종 판정을 사용합니다.\n\n"
        readme += ("이 자료는 Target 정답을 사용합니다. pape_estimates.csv의 estimated_*에도 actual_*를 복사한 인터페이스 호환용 oracle입니다. PAPE 예측 또는 라벨 없는 운영 결과로 보고하지 마세요. DRE AUC/ESS/weight는 해당하지 않아 비워 두었습니다. 순수 실제값은 lstm_ae_true_data.csv입니다.\n\n" if oracle else
                   f"주 인계 파일은 Hybrid+DRE, seed {selected_seed}의 36개 결과입니다. 비교용 1,440건은 comparisons/에 있습니다. seed를 시간축처럼 이어 붙이지 마세요. Hybrid는 문서 기본값이며 최종 입력 선정은 보류 상태입니다.\n\n")
        readme += "필수 파일: pape_estimates.csv(공통 14컬럼), pape_summary.json(실제/추정 TP·FP·FN·TN, 오차 계산 자료, 진단, 선택 이유와 실패 조건).\n\nCSV의 actual_*와 summary의 평가 정답을 일반 Gain/PAPE 입력으로 사용하지 마세요. src.pape.release.load_release_inputs를 사용하면 허용된 현재 추정 특징만 반환합니다.\n\n"
        readme += "```python\nfrom src.pape.release import load_release_inputs\ninputs = load_release_inputs('이 폴더 경로', allow_review=True" + (", allow_oracle=True" if oracle else "") + ")\n```\n\n"
        readme += "allow_review=True는 진단 경고가 있는 연구 자료를 읽겠다는 뜻이며 정확도 승인이 아닙니다. 파일 순서는 record_id 순서이며 실제 운항 시간순으로 인증되지 않았습니다. 이 자료는 현재 손실의 연결 검증에 사용하고, 실제 미래 H개 Chunk·재학습 모델 결과·Gain 정답·RCCDA 행동은 별도로 구성해야 합니다.\n\n"
        readme += f"FN:FP 비용은 원본 YAML의 {costs['lambda_fn']}:{costs['lambda_fp']}입니다. 회사 확정 비용이 아닙니다. config/resolved_portable.yaml은 모든 설정을 보존하며 절대경로만 교체합니다. 행별 설정과 원본·이식 설정 hash 대응은 config/effective_configs_portable.json에 있습니다.\n\n"
        readme += "원본 YAML·행별 설정과 대형 가중치/확률 로그는 같은 상위 폴더의 PAPE_reference_artifacts_version_1.0에 있습니다. diagnostics_manifest.json으로 행·파일·SHA256을 연결합니다. 해당 폴더도 별도로 전달하세요. 원본 설정의 개인 경로가 포함되므로 Git에 올리지 않습니다. 공통 Window 데이터와 baseline_manifest는 기존 모델 담당자의 별도 인계 자료를 사용합니다. 새 LSTM 학습이나 Test 008~009 열람은 수행하지 않았습니다.\n"
        (out/"README.md").write_text(readme,encoding="utf-8")
        manifest={"release_version":VERSION,"artifact_kind":kind,"files":{str(p.relative_to(out)).replace('\\','/'):sha256(p) for p in out.rglob('*') if p.is_file()}}
        write_json(out/"manifest.json",manifest)
        read_release(out,allow_oracle=oracle)
        payload=load_release_inputs(out,allow_review=True,allow_oracle=oracle)
        payload_name="rccda_oracle_inputs.json" if oracle else "rccda_inputs.json"
        write_json(out/payload_name,payload)
        manifest["files"][payload_name]=sha256(out/payload_name)
        write_json(out/"manifest.json",manifest)
        read_release(out,allow_oracle=oracle)
        archive=root/(name+'.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.rglob('*')):
                if p.is_file():z.write(p,str(p.relative_to(root)))
        packages.append({"directory":str(out),"archive":str(archive),"sha256":sha256(archive),"chunks":len(packaged),"oracle":oracle})
    for path,digest in pins.items():
        if sha256(path)!=digest:raise ValueError("Original source changed during packaging")
    result={"release_version":VERSION,"packages":packages,"original_rows_verified":len(rows),"actual_chunks_recomputed":len(selected),
            "source_files_unchanged":True,"new_training":False,"official_test_used":False,
            "reference_artifacts_directory":str(root/diagnostics_manifest["directory"]),
            "reference_artifact_files_verified":len(diagnostics_manifest["files"])}
    write_json(root/"verification.json",result)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir',required=True)
    parser.add_argument('--target-features',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--seed',type=int,default=None,help='Optional archived seed override; default reads resolved.yaml')
    parser.add_argument('--code-revision')
    args=parser.parse_args()
    print(json.dumps(package(args.experiment_dir,args.target_features,args.output_dir,seed=args.seed,code_revision=args.code_revision),ensure_ascii=False,indent=2))
