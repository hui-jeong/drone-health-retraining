"""Evaluate configured Target files against a fixed, disjoint PAPE Source."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd

from scripts.run_pape_pilot import _run_one, aggregate_records, aggregate_loss_sensitivity
from scripts.run_pape_real import _branch_info
from src.pape.contracts import PAPE_ESTIMATE_COLUMNS, validate_window_features
from src.pape.fixed_reference import load_fixed_features, sha256
from src.pape.handoff import iter_handoff_target_runs, validate_window_geometry
from src.pape.io import config_hash, load_config, utc_now, write_handoff_metadata, write_json

_SOURCE = _TARGET = None


def initialize_worker(config):
    global _SOURCE, _TARGET
    _SOURCE, _TARGET, _ = load_fixed_features(config)


def evaluate_group(config, seed, name, weighting):
    root = Path(config["fixed_reference"]["evaluation_root"]) / "trials" / str(seed) / name / weighting
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for chunk, target in iter_handoff_target_runs(_TARGET):
        record_id = int(target.record_id.iloc[0])
        path = root / f"record_{record_id:04d}.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["study_config_hash"] != config_hash(config):
                raise ValueError("Cannot resume with changed experiment settings")
            for kind, entry in record["trial_artifacts"].items():
                if sha256(root / entry["file"]) != entry["sha256"]:
                    raise ValueError(f"Changed trial artifact: {kind}")
        else:
            record, weights, probabilities = _run_one(_SOURCE, target, chunk, seed,
                "fixed_" + "_".join(config["fixed_reference"]["roles"]["pape_source"]) + "_reference", "validation", name, weighting, config)
            record.update(target_record_id=record_id, scenario_id=str(target.scenario_id.iloc[0]),
                target_data_class=str(target.data_class.iloc[0]), study_config_hash=config_hash(config),
                baseline_split="pape_validation", baseline_seed=config["fixed_reference"]["model_seed"])
            record["trial_artifacts"] = {}
            for kind, table in [("source_weights", weights), ("target_probabilities", probabilities)]:
                file = root / f"record_{record_id:04d}_{kind}.parquet"
                table.to_parquet(file, index=False)
                record["trial_artifacts"][kind] = {"file": file.name, "sha256": sha256(file)}
            temporary = path.with_suffix(".partial")
            write_json(temporary, record)
            temporary.replace(path)
        records.append(record)
        if len(records) % 6 == 0:
            print(f"Progress {seed}/{name}/{weighting}: {len(records)} target files", flush=True)
    return records


def seed_claims(records):
    """Report signed paired differences; no exact-winner or invented tolerance gate."""
    table = pd.DataFrame(aggregate_records(records, ["seed", "input_name", "weighting"]))
    claims = []
    for name, part in table.groupby("input_name"):
        baseline = part[part.weighting.eq("uniform")].set_index("seed")
        candidate = part[part.weighting.eq("dre")].set_index("seed")
        for metric in ["fn_mae", "recall_mae", "f1_mae", "loss_mae"]:
            delta = candidate[metric] - baseline[metric]
            claims.append({"comparison": "dre_minus_uniform", "input_name": name, "metric": metric,
                "seed_deltas": {str(k): float(v) for k, v in delta.items()},
                "all_seeds_strictly_improve": bool(delta.lt(0).all()),
                "all_seeds_non_worsening": bool(delta.le(0).all()),
                "direction_varies": bool(delta.lt(0).any() and delta.gt(0).any())})
    return claims


def write_report(summary, output):
    frozen = summary['protocol'] == 'frozen_export_source_v1'
    source_runs = ', '.join(summary['roles']['pape_source'])
    target_runs = ', '.join(summary['roles']['pape_validation'])
    evaluation_note = (f"{target_runs}은 PAPE Source에서 제외한 개발 평가 Target입니다. 원본 split=train은 보존하며 공식 Test로 재분류하지 않았습니다. 007 및 Test 008~009는 이번 실행에서 사용하지 않았습니다."
                       if summary.get('target_mode') == 'heldout_train_run' else
                       "007은 과거 개발 과정에서 이미 검토한 자료이므로, 이번에도 독립 최종 평가로 해석하지 않습니다. Test 008~009는 사용하지 않았습니다.")
    if summary.get('target_mode') == 'target005_with_007_source':
        evaluation_note = ("005 전체를 PAPE Source에서 제외하고 평가합니다. 007의 특징과 정답은 사용자가 승인한 개발 비교의 Source로 사용했습니다. "
                           "원본 005 split=train과 007 split=validation은 보존합니다. 독립 최종 평가 또는 시간순 미래 검증이 아니며, Test 008~009는 사용하지 않았습니다.")
    lines = ["# 고정 Source PAPE 재구현 결과", "",
        "역할 배분: " + json.dumps(summary["roles"], ensure_ascii=False),
        "전달된 기존 탐지기 출력을 고정하고 PAPE만 학습했습니다. LSTM-AE·Scaler·Residual·융합·임계값은 fit하지 않았습니다." if frozen else ("내부 검증이 없는 실행은 학습 출력으로 융합·임계값을 보정합니다. 이 보정 성능은 독립 검증 성능이 아닙니다." if not summary["roles"]["model_calibration"] else "별도 내부 검증 자료로 모델을 보정했습니다."),
        evaluation_note, "",
        f"{source_runs} 기존 출력과 정답 전체를 Source로 사용했습니다. 전달 export는 고정했지만 정확한 checkpoint 이력은 인계 문서만으로 인증할 수 없습니다. 과거 학습 출력 재사용의 편향 가능성을 포함한 개발 평가입니다." if frozen else "기존 Hybrid와 다른 체크포인트입니다. 이전 Pilot과의 차이는 Source 분리와 탐지기 재학습이 함께 바뀐 결과이며 PAPE 알고리즘만의 효과로 볼 수 없습니다.", "",
        "| 입력 | 가중치 | FN MAE | Recall MAE | 양성 Run Recall MAE | F1 MAE |", "|---|---|---:|---:|---:|---:|"]
    for row in summary["metrics"]:
        positive = row["recall_mae_defined_only"]
        positive_text = "해당 없음" if positive is None else f"{positive:.5f}"
        lines.append(f"| {row['input_name']} | {row['weighting']} | {row['fn_mae']:.4f} | {row['recall_mae']:.5f} | {positive_text} | {row['f1_mae']:.5f} |")
    lines += ["", "FN·Recall 오차를 함께 비교하며 순위 합으로 자동 선정하지 않습니다. 기본 Hybrid/OOF를 유지하고, 입력 선정 및 운영 승인은 별도 판단 사항입니다.",
        "정상 Run의 Recall=0 관례가 전체 MAE에 포함되므로 실제 양성 Run만의 오차도 병기했습니다.", "",
        "## Seed 해석", "", "Source·Target과 Hybrid 모델은 고정했습니다. 5개 seed는 PAPE DRE 내부 fold와 LGBM 난수만 바꿉니다. Hybrid 재학습 seed 안정성을 평가한 것은 아닙니다.",
        f"동일한 {summary['unique_target_records']}개 Target을 seed별로 반복 평가하므로 독립 표본 수가 늘어난 것으로 세지 않습니다. 아래 값은 DRE 오차에서 uniform 오차를 뺀 값이며, 음수이면 DRE가 개선한 것입니다.", "",
        "| 입력 | 지표 | seed 11 | seed 23 | seed 37 | seed 53 | seed 71 | 개선 방향 혼재 |",
        "|---|---|---:|---:|---:|---:|---:|---|"]
    for claim in summary["seed_claims"]:
        if claim["metric"] in {"fn_mae", "recall_mae"}:
            values = " | ".join(f"{claim['seed_deltas'][str(seed)]:+.5f}" for seed in summary["seeds"])
            lines.append(f"| {claim['input_name']} | {claim['metric']} | {values} | {'예' if claim['direction_varies'] else '아니오'} |")
    lines += ["",
        "## A1", "", "아래는 A1 합성 이상 Run의 실제 FN과 추정 FN입니다. 실제 FN 감소(탐지기)와 FN 추정오차 감소(PAPE)를 구분해야 합니다.",
        "| 입력 | 방식 | 실제 FN | 추정 FN 최소 | 추정 FN 최대 | FN MAE |", "|---|---|---:|---:|---:|---:|"]
    for row in summary["a1_abnormal"]:
        lines.append(f"| {row['input_name']} | {row['weighting']} | {row['actual_fn']} | {row['estimated_fn_min']:.4f} | {row['estimated_fn_max']:.4f} | {row['fn_mae']:.4f} |")
    lines += ["", "## 추적 자료", "", "공통 14개 컬럼은 pape_estimates.csv에 유지했습니다. seed·source_key·실행시각·config hash와 실제/추정 혼동행렬은 pape_summary.json에 있습니다.",
        "trial별 Source 가중치와 Target 확률은 trials/<seed>/<input>/<weighting>/record_* parquet에 보존합니다. 각 완료 trial의 JSON에 파일 해시가 있으며 재개 시 확인합니다.",
        "실제 Gain/RCCDA 연결 및 재학습 정책 판단은 하지 않았습니다.", ""]
    (output / "pape_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(config, config_paths):
    git = _branch_info(ROOT)
    source, target, provenance = load_fixed_features(config)
    for frame in (source, target):
        validate_window_features(frame)
        validate_window_geometry(frame, config, check_runs=True)
    study = config["experiment"]
    if set(study["input_names"]) != {"ae_score", "residual_probability", "hybrid_score", "combined_scores"}:
        raise ValueError("Use the four document-required inputs for this protocol")
    if len(set(study["seeds"])) != len(study["seeds"]) or not study["seeds"]:
        raise ValueError("Seeds must be nonempty and unique")
    output = Path(config["fixed_reference"]["evaluation_root"])
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "run_status.json"
    identity = {"config_hash": config_hash(config), "export_manifest_sha256": sha256(
        Path(config["fixed_reference"]["export_root"]) / "baseline_manifest.json")}
    if status_path.exists():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        if any(previous[k] != v for k, v in identity.items()):
            raise ValueError("Output belongs to another experiment")
        if previous["status"] == "complete":
            raise FileExistsError("Completed experiment must not be rerun")
    write_json(status_path, {**identity, "status": "running", "executed_at": utc_now(), "git": git})
    tasks = [(seed, name, weighting) for seed in study["seeds"]
             for name in study["input_names"] for weighting in study["weightings"]]
    records = []
    with ProcessPoolExecutor(max_workers=config["fixed_reference"]["pape_workers"],
                             initializer=initialize_worker, initargs=(config,)) as pool:
        futures = {pool.submit(evaluate_group, config, *task): task for task in tasks}
        for future in as_completed(futures):
            records.extend(future.result())
            print(f"Completed {futures[future]} | {len(records)} / {len(tasks) * target.record_id.nunique()} trials", flush=True)
    records.sort(key=lambda r: (r["seed"], r["input_name"], r["weighting"], r["target_record_id"]))
    table = pd.DataFrame(records)
    expected = len(tasks) * target.record_id.nunique()
    if len(records) != expected or table.duplicated(["seed", "input_name", "weighting", "target_record_id"]).any():
        raise ValueError("Trial coverage mismatch")
    a1 = table[table.scenario_id.eq("A1") & table.target_data_class.eq("abnormal")]
    a1_summary = [{"input_name": name, "weighting": weighting, "actual_fn": int(part.actual_fn.iloc[0]),
        "estimated_fn_min": float(part.estimated_fn.min()), "estimated_fn_max": float(part.estimated_fn.max()),
        "fn_mae": float(part.fn_absolute_error.mean())}
        for (name, weighting), part in a1.groupby(["input_name", "weighting"])]
    summary = {**identity, "executed_at": utc_now(), "model_version": config["model_version"],
        "protocol": config["fixed_reference"]["protocol"], "roles": config["fixed_reference"]["roles"],
        "target_mode": config['fixed_reference'].get('target_mode', 'validation_007'),
        "seeds": study["seeds"], "unique_target_records": int(target.record_id.nunique()),
        "source_keys": sorted(source.source_key.unique()), "target_keys": sorted(target.source_key.unique()),
        "model_provenance": provenance, "git": git, "official_test_used": False,
        "target_labels_used_in_estimation": False, "default_input_variant": config["pape"]["input_variant"],
        "production_input_selected": False, "metrics": aggregate_records(records, ["input_name", "weighting"]),
        "metrics_by_seed": aggregate_records(records, ["seed", "input_name", "weighting"]),
        "seed_claims": seed_claims(records), "a1_abnormal": a1_summary,
        "loss_sensitivity_metrics": aggregate_loss_sensitivity(records),
        "independent_final_evaluation": False,
        "stability_scope": study["stability_scope"],
    }
    write_handoff_metadata(output, records, summary, config, config_paths)
    write_report(summary, output)
    write_json(status_path, {**identity, "status": "complete", "executed_at": utc_now(), "records": len(records)})
    print(f"COMPLETE: {len(records)} trials", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", required=True)
    args = parser.parse_args()
    run(load_config(*args.config), args.config)
