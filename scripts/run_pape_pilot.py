"""PAPE dummy experiments and the week-two real model-adapter handoff Pilot."""
import argparse
from copy import deepcopy
from importlib.metadata import version
import platform
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.dummy_data import iter_target_runs, make_dummy_dataset
from src.pape.io import config_hash, load_config, save_handoff, utc_now, write_json
from src.common.metrics import actual_confusion, estimation_errors
from src.pape.contracts import hide_target_labels
from src.pape.estimator import estimate_target_performance


def aggregate_records(records, group_columns):
    df = pd.DataFrame(records)
    result = []
    for keys, group in df.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        entry = dict(zip(group_columns, keys))
        entry["n_target_chunks"] = len(group)
        for metric in ["fn", "recall", "f1", "loss"]:
            entry[f"{metric}_mae"] = float(group[f"{metric}_absolute_error"].mean())
        # Macro MAE above uses the documented zero_division convention. Also
        # report Recall MAE over chunks that actually contain positives.
        positives = group[group.actual_recall_defined]
        entry["recall_mae_defined_only"] = float(positives.recall_absolute_error.mean()) if len(positives) else None
        entry["n_actual_recall_defined"] = len(positives)
        result.append(entry)
    return result


def select_input(records, config):
    """Only the configured selection groups select DRE inputs; never Test or validation."""
    eligible = [r for r in records if r["split"] == "selection" and
                r["case"] == config["experiment"]["selection_case"] and r["weighting"] == "dre"]
    if not eligible:
        raise ValueError("No eligible selection records")
    priorities = config["experiment"]["selection_priority"]
    if priorities != ["fn_mae", "recall_mae"]:
        raise ValueError("The shared contract requires FN and Recall estimation errors as priorities")
    if config["experiment"]["selection_rule"] != "sum_of_fn_and_recall_mae_ranks":
        raise ValueError("Unknown selection rule")

    def ranked(rows):
        table = pd.DataFrame(aggregate_records(rows, ["input_name"]))
        table["selection_rank_sum"] = sum(table[m].rank(method="min") for m in priorities)
        order = ["selection_rank_sum", *config["experiment"]["tie_breakers"]]
        return table.sort_values(order).to_dict(orient="records")

    ranking = ranked(eligible)
    per_seed = {str(seed): ranked([r for r in eligible if r["seed"] == seed])
                for seed in config["experiment"]["seeds"]}
    winner = ranking[0]["input_name"]
    seed_winners = {seed: entries[0]["input_name"] for seed, entries in per_seed.items()}
    stable = all(name == winner for name in seed_winners.values())
    return {
        "selected_input_variant": winner,
        "selected_configuration": config["pape"]["variants"][winner],
        "status": ("exploratory_real_candidate_only" if config.get("data_origin") == "model_adapter_handoff"
                   else "dummy_candidate_only") if stable else "provisional_seed_unstable",
        "reason": "FN MAE와 Recall MAE의 순위 합을 최소화. 동률이면 설정된 오차 순서로 비교. DRE AUC는 선정에 사용하지 않음.",
        "ranking": ranking, "per_seed_ranking": per_seed, "seed_winners": seed_winners,
        "same_winner_all_seeds": stable, "default_input_variant": config["pape"]["input_variant"],
        "production_input_selected": False,
        "stability_scope": config["experiment"].get("stability_scope", "joint_data_split_dre_fold_and_model_seed"),
        "robustness_completion_passed": stable and len(config["experiment"]["seeds"]) >= 2
            and not any(v is not None for v in config.get("randomness", {}).values()),
        "selection_scope": "PAPE selection source_keys only; PAPE validation, diagnostics and official Test excluded",
    }


def aggregate_loss_sensitivity(records):
    rows = []
    for record in records:
        for result in record["loss_sensitivity"]:
            rows.append({**{key: record[key] for key in ["case", "split", "input_name", "weighting"]}, **result})
    if not rows:
        return []
    groups = ["case", "split", "input_name", "weighting", "name", "lambda_fn", "lambda_fp"]
    table = pd.DataFrame(rows)
    result = []
    for identity, group in table.groupby(groups, sort=True):
        result.append({**dict(zip(groups, identity)), "n_target_chunks": len(group),
                       "loss_mae": float(group.loss_absolute_error.mean())})
    return result


def summarize_diagnostics(records):
    table = pd.DataFrame([r for r in records if r["weighting"] == "dre"])
    if table.empty:
        return []
    result = []
    for identity, group in table.groupby(["case", "split", "input_name"], sort=True):
        result.append({
            **dict(zip(["case", "split", "input_name"], identity)),
            "n_target_chunks": len(group),
            "min_ess_ratio": float(group.ess_ratio.min()),
            "median_ess_ratio": float(group.ess_ratio.median()),
            "max_raw_weight": float(group.max_raw_weight.max()),
            "max_weight": float(group.max_weight.max()),
            "mean_clip_fraction": float(group.clip_fraction.mean()),
            "warning_counts": {flag: sum(flag in flags for flags in group.warnings) for flag in [
                "low_effective_sample_size", "excessive_weight_clipping", "target_outside_source_range",
                "small_target_sample", "too_few_expected_positives_for_stable_recall",
                "target_scenario_absent_from_source"]},
        })
    return result


def write_pilot_report(summary, output):
    """Human-readable handoff derived from exactly the saved JSON values."""
    chosen = summary["selection"]
    real = summary.get("data_origin") == "model_adapter_handoff"
    lines = [
        ("# 3주차 PAPE 입력·보정 비교 (개발용 탐색 Pilot)" if summary.get("study_stage") == "stage3_development_comparison"
         else "# 2주차 실제 데이터 PAPE 탐색 Pilot") if real else "# 1주차 PAPE 더미 실험 결과", "",
        ("Hybrid LSTM-AE v2.3 연동 담당자가 인계한 실제 Window Feature로 계산했습니다. 기준 모델의 Validation을 다시 나눈 탐색 실험이며 독립 최종 평가가 아닙니다. PAPE가 기준 모델을 재학습하거나 재실행하지 않았습니다."
         if real else "이 결과는 합성 Window Feature 단위 테스트입니다. 실제 Hybrid LSTM-AE v2.3 모델을 실행한 결과가 아닙니다."), "",
        f"- 실험 seed: {summary['seeds']}",
        f"- 기본 입력: `{chosen['default_input_variant']}`",
        f"- {'실제 탐색 Pilot' if real else '더미 데이터'}의 선택 후보: `{chosen['selected_input_variant']}`",
        f"- 모든 seed의 1위가 동일함: `{chosen['same_winner_all_seeds']}`",
        f"- 설정 hash: `{summary['config_hash']}`", "",
        chosen["reason"], "",
        ("각 seed의 선택용 source_key로 후보를 고정하고 같은 seed의 별도 검증 source_key에서 확인했습니다. 전체 seed의 합산 순위는 요약용이며 검증 후보 결정에 사용하지 않습니다. 실제 입력의 최종 선정은 후속 검증 이후 진행합니다."
         if real else "선택용 source_key로 후보를 고른 후 고정하고, 별도 검증 source_key에서 확인했습니다. 실제 입력의 최종 선정은 후속 검증 이후 진행합니다."), "",
        "## 선택용 데이터의 DRE 비교", "",
        "| 입력 구성 | FN MAE | Recall MAE | F1 MAE | Loss MAE | FN·Recall 순위 합 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in chosen["ranking"]:
        lines.append(f"| {row['input_name']} | {row['fn_mae']:.4f} | {row['recall_mae']:.4f} | {row['f1_mae']:.4f} | {row['loss_mae']:.4f} | {row['selection_rank_sum']:.1f} |")
    lines += ["", "## 별도 검증 데이터: 같은 split에서 동일 가중치와 DRE 비교", "",
              "| 입력 구성 | 방식 | Run 평가 수 | FN MAE | Recall MAE | F1 MAE | Loss MAE |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary["metrics"]:
        if row["split"] == "validation":
            lines.append(f"| {row['input_name']} | {row['weighting']} | {row['n_target_chunks']} | {row['fn_mae']:.4f} | {row['recall_mae']:.4f} | {row['f1_mae']:.4f} | {row['loss_mae']:.4f} |")
    lines += ["", "Recall 분모가 0인 정상 Run은 설정에 따라 0으로 기록합니다. 전체 Run MAE에는 이 관례의 영향이 있으므로 실제 양성이 있는 Run만의 Recall MAE도 함께 확인해야 합니다.", "",
              "| 입력 구성 | 방식 | 실제 양성 Run 수 | 실제 양성 Run만의 Recall MAE |",
              "|---|---|---:|---:|"]
    for row in summary["metrics"]:
        if row["split"] == "validation" and row["recall_mae_defined_only"] is not None:
            lines.append(f"| {row['input_name']} | {row['weighting']} | {row['n_actual_recall_defined']} | {row['recall_mae_defined_only']:.4f} |")
    lines += ["", "DRE 적용이 모든 지표의 개선을 보장하지 않습니다. 위의 동일 가중치 결과와 지표별로 비교하고, 입력 선택의 seed 안정성과 DRE의 지표별 개선 여부를 구분해 해석합니다.", "",
              "## 손실비율 민감도: 10:1 참고값", "",
              "기본 손실은 FN:FP = 1:1이며 위 표에 표시했습니다. 아래는 같은 추정 FN/FP를 10:1로 환산한 별도 검증 결과입니다. 기업 확정 비용은 아닙니다.", "",
              "| 입력 구성 | 방식 | FN 비용 | FP 비용 | Loss MAE |",
              "|---|---|---:|---:|---:|"]
    for row in summary["loss_sensitivity_metrics"]:
        if row["split"] == "validation":
            lines.append(f"| {row['input_name']} | {row['weighting']} | {row['lambda_fn']:g} | {row['lambda_fp']:g} | {row['loss_mae']:.4f} |")
    lines += ["", "## seed별 선택", ""]
    lines += [f"- seed {seed}: `{name}`" for seed, name in chosen["seed_winners"].items()]
    lines += ["", "## 별도 검증 데이터의 DRE 가중치 진단", "",
              "| 입력 구성 | Run 수 | 최소 ESS 비율 | 중앙 ESS 비율 | 낮은 ESS 경고 수 | clipping 경고 수 |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in summary["diagnostics_by_case"]:
        if row["split"] == "validation":
            warnings = row["warning_counts"]
            lines.append(f"| {row['input_name']} | {row['n_target_chunks']} | {row['min_ess_ratio']:.4f} | {row['median_ess_ratio']:.4f} | {warnings['low_effective_sample_size']} | {warnings['excessive_weight_clipping']} |")
    lines += ["", "후보 선정은 오차 비교 결과입니다. 낮은 ESS나 많은 clipping은 가중치 안정성을 다시 확인해야 하는 근거이며, 단위 테스트 통과가 이 경고를 해소하지 않습니다.", ""]
    if not real:
        lines += ["## 실패 조건 진단 실험", "",
                  "아래 조건은 입력 선택에서 제외했고, 각 seed의 검증 source_key 하나에서 정상·합성 Run을 평가했습니다. support 밖에서 일부 오차가 작게 나와도 support 문제가 해결되었다는 뜻은 아닙니다.", "",
                  "| 조건 | 입력 구성 | 방식 | Run 수 | FN MAE | Recall MAE | F1 MAE | Loss MAE |",
                  "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in summary["metrics"]:
        if row["split"] == "diagnostic" and row["input_name"] == chosen["selected_input_variant"]:
            lines.append(f"| {row['case']} | {row['input_name']} | {row['weighting']} | {row['n_target_chunks']} | {row['fn_mae']:.4f} | {row['recall_mae']:.4f} | {row['f1_mae']:.4f} | {row['loss_mae']:.4f} |")
    lines += ["", "## 실패 조건과 해석 한계", ""]
    lines += [f"- {condition}" for condition in summary["failure_conditions"]]
    if real:
        cv = chosen.get("selection_protocol") == "grouped_cv"
        lines += ["", "## seed별 외부 예약 구분" if cv else "## seed별 Source와 Target의 구성", "",
                  ("아래 source+selection을 합친 개발 풀에서 그룹 교차검증으로 후보를 고릅니다. validation은 그 seed의 선정에 사용하지 않습니다. 최종 보정은 개발 풀 전체로 학습합니다."
                   if cv else "원본 그룹 수이며 정상·파생 비정상 파일은 함께 이동합니다. seed 간에는 같은 원본이 재사용됩니다."), "",
                  "| seed | 용도 | 원본 그룹 | Run 파일 | Window | 시나리오별 원본 그룹 |",
                  "|---:|---|---:|---:|---:|---|"]
        for profile in summary.get("split_profiles", []):
            for role, counts in profile["roles"].items():
                scenarios = ", ".join(f"{k}:{v}" for k, v in counts["source_keys_by_scenario"].items())
                lines.append(f"| {profile['seed']} | {role} | {counts['source_key_count']} | {counts['record_count']} | {counts['window_count']} | {scenarios} |")
        if cv:
            lines += ["", "### 선정용 내부 그룹 교차검증", "",
                      "원본 그룹마다 정상·합성 짝을 함께 옮기며 각 seed에서 모든 개발 그룹을 정확히 한 번 Target으로 평가합니다. 다른 fold에서는 Source가 될 수 있으나 동일 추정의 Source와 Target에는 중복이 없습니다.", "",
                      "| seed | fold | Source 그룹 | 선정 Target 그룹 | 외부 검증 그룹 |",
                      "|---:|---:|---:|---:|---:|"]
            for fold in summary["selection_cv_manifest"]:
                lines.append(f"| {fold['seed']} | {fold['fold']} | {len(fold['source_keys'])} | {len(fold['target_keys'])} | {len(fold['outer_validation_keys'])} |")
    lines += ["", "## 파일 읽는 법", "",
              "`pape_estimates.csv`는 문서 5.2의 컬럼을 유지합니다. `input_variant`의 `__uniform` / `__dre`로 비교군을 구분합니다.", "",
              "추가 TP·TN, 실제 손실, 오차, seed, source_key, 실행시각, 유효 설정은 `pape_summary.json`의 `records`에 있습니다. `records[i]`는 CSV의 헤더를 제외한 0부터 시작하는 i번째 데이터 행입니다.", "",
              f"`source_weights.csv.gz`에 원래 가중치·clipping 결과·검증 fold를 압축 저장하고, `{'target_probabilities.csv.gz' if real else 'target_probabilities.csv'}`에 기대 혼동행렬을 재계산할 보정 확률과 고정 판정을 저장합니다.", "",
              "손실 비율의 민감도 결과는 `records[].loss_sensitivity`에 있습니다. 비용 비율은 기업 확정 기준이 아닙니다.", ""]
    (Path(output) / "pape_report.md").write_text("\n".join(lines), encoding="utf-8")


def _run_one(source, target, chunk_id, seed, case, split, input_name, weighting, config):
    local = deepcopy(config)
    local["seed"] = seed
    local["target_chunk_id"] = f"{case}::{chunk_id}"
    local["pape"]["input_variant"] = input_name
    local["pape"]["weighting"] = weighting
    start = time.perf_counter()
    hidden = hide_target_labels(target)
    result = estimate_target_performance(source, hidden, local)
    # This is the first evaluation-label access. The estimator has already finished.
    actual = actual_confusion(target.y_true, target.final_prediction,
                              local["loss"]["lambda_fn"], local["loss"]["lambda_fp"],
                              local["metrics"]["zero_division"])
    estimated = {k.removeprefix("estimated_"): v for k, v in result.estimates.items()
                 if k.startswith("estimated_")}
    record = {**result.estimates, **{f"actual_{k}": v for k, v in actual.items()},
              **estimation_errors(estimated, actual), **result.diagnostics,
              "case": case, "split": split, "duration_seconds": time.perf_counter() - start,
              "data_origin": config.get("data_origin", "synthetic_window_feature_fixture"),
              "effective_config": local}
    record["loss_sensitivity"] = [
        {"name": cost["name"], "lambda_fn": cost["lambda_fn"], "lambda_fp": cost["lambda_fp"],
         "estimated_loss": cost["lambda_fn"] * estimated["fn"] + cost["lambda_fp"] * estimated["fp"],
         "actual_loss": cost["lambda_fn"] * actual["fn"] + cost["lambda_fp"] * actual["fp"],
         "loss_absolute_error": abs(cost["lambda_fn"] * (estimated["fn"] - actual["fn"]) +
                                    cost["lambda_fp"] * (estimated["fp"] - actual["fp"]))}
        for cost in local["loss"]["sensitivity"]]
    identity = {"target_chunk_id": result.estimates["target_chunk_id"],
                "input_variant": result.estimates["input_variant"], "case": case, "split": split,
                **{k: record[k] for k in ["seed", "config_hash", "model_version", "executed_at"]}}
    for table in [result.source_weights, result.target_probabilities]:
        for key, value in identity.items():
            table[key] = value
    return record, result.source_weights, result.target_probabilities


def run_pilot(config, config_paths, output_dir):
    if config["loss"]["lambda_fn"] is None or config["loss"]["lambda_fp"] is None:
        raise ValueError("FN/FP dummy loss weights must be set in config/pape.yaml")
    if Path(output_dir).exists():
        raise FileExistsError(f"Output directory exists; choose a new --output-dir: {output_dir}")
    if len(set(config["experiment"]["seeds"])) != len(config["experiment"]["seeds"]):
        raise ValueError("Experiment seeds must be unique")
    start = time.perf_counter()
    records, weight_tables, probability_tables, manifests = [], [], [], []
    seed_datasets = {}
    selection_case = config["experiment"]["selection_case"]
    if selection_case != "covariate_shift":
        raise ValueError("Week-one selection is limited to the covariate_shift fixture")

    def append_result(result):
        record, weights, probabilities = result
        records.append(record)
        weight_tables.append(weights)
        probability_tables.append(probabilities)

    # All predictions for each trial are produced before its labels are revealed.
    # Only dedicated selection groups influence the choice of input configuration.
    for seed in config["experiment"]["seeds"]:
        datasets, manifest = make_dummy_dataset(config, seed, selection_case)
        seed_datasets[seed] = datasets
        manifests.extend(manifest.to_dict(orient="records"))
        for input_name in config["pape"]["variants"]:
            for chunk_id, target in iter_target_runs(datasets["selection"]):
                for weighting in config["experiment"]["weightings"]:
                    append_result(_run_one(datasets["source"], target, chunk_id, seed, selection_case,
                                           "selection", input_name, weighting, config))
            print(f"selection seed={seed} input={input_name} completed", flush=True)
    selection = select_input(records, config)
    print(f"Frozen dummy candidate: {selection['selected_input_variant']}; seed stable={selection['same_winner_all_seeds']}", flush=True)
    # A frozen choice is now checked on distinct validation keys; no re-selection.
    validation_variants = list(dict.fromkeys([config["pape"]["input_variant"], selection["selected_input_variant"]]))
    for seed, datasets in seed_datasets.items():
        for input_name in validation_variants:
            for chunk_id, target in iter_target_runs(datasets["validation"]):
                for weighting in config["experiment"]["weightings"]:
                    append_result(_run_one(datasets["source"], target, chunk_id, seed, selection_case,
                                           "validation", input_name, weighting, config))
        print(f"validation seed={seed} completed", flush=True)
        # Diagnostic cases are deliberately excluded from model/input selection.
        # Keep one normal + one abnormal file from the first validation source_key.
        for case in config["experiment"]["diagnostic_cases"]:
            diagnostic_data, _ = make_dummy_dataset(config, seed, case)
            first_key = sorted(diagnostic_data["validation"].source_key.unique())[0]
            subset = diagnostic_data["validation"].query("source_key == @first_key")
            for input_name in validation_variants:
                for chunk_id, target in iter_target_runs(subset):
                    for weighting in config["experiment"]["weightings"]:
                        append_result(_run_one(diagnostic_data["source"], target, chunk_id, seed, case,
                                               "diagnostic", input_name, weighting, config))
        print(f"diagnostics seed={seed} completed", flush=True)

    summary = {
        "schema_reference": "엔스퀘어 과제 구현 및 업무 분담.pdf v1.0, sections 4.2 / 5.1 / 5.2 / 6",
        "model_version": config["model_version"], "executed_at": utc_now(),
        "config_hash": config_hash(config), "seeds": config["experiment"]["seeds"],
        "data_origin": "synthetic_window_feature_fixture", "real_hybrid_model_executed": False,
        "baseline_f1_reproduced": False, "official_test_used": False,
        "target_labels_used_during_estimation": False,
        "selection": selection,
        "metrics": aggregate_records(records, ["case", "split", "input_name", "weighting"]),
        "metrics_by_seed": aggregate_records(records, ["case", "split", "seed", "input_name", "weighting"]),
        "loss_sensitivity_metrics": aggregate_loss_sensitivity(records),
        "diagnostics_by_case": summarize_diagnostics(records),
        "split_manifest": manifests,
        "loss_definition": config["loss"],
        "csv_contract": {"schema_unchanged": True,
                         "input_variant_encoding": "{input_name}__{uniform|dre}",
                         "record_mapping": "records[i] corresponds to zero-based CSV data row i (header excluded)",
                         "uniform_dre_auc": "null in JSON; empty in CSV (no DRE fitted)"},
        "failure_conditions": [
            "Concept shift: P_source(y|x) != P_target(y|x); cannot be identified reliably without labels.",
            "Target outside Source support: range checks are warnings, not proof of multivariate support overlap.",
            "Low ESS or heavy clipping: density estimates may be unstable or biased. No automatic claim of valid loss.",
            "Too few Source groups, one-class Source labels, or too few non-overlapping Target windows: fail explicitly.",
            "Marginal score calibration alone need not calibrate subgroups selected by context thresholds or hysteresis.",
            "Recall is undefined without positives; zero_division=0 is logged alongside defined-only MAE.",
            "Overlapping Windows are dependent; this pilot does not produce IID confidence intervals.",
            "Dummy winners and diagnostic thresholds do not establish real Hybrid LSTM-AE v2.3 performance.",
        ],
        "implementation_notes": [
            "PAPE uses DRE-weighted Source calibration, then Target expected confusion counts; not direct IW counts.",
            "DRE weights are out-of-fold by Source source_key with fold training prior correction.",
            "Single-Run Target DRE validation uses contiguous temporal blocks and purges overlapping intervals.",
            "combined_scores is a multivariate calibration extension. hybrid_score_dre_combined changes only DRE input.",
            "No scaler is fitted: input is already-exported feature fixtures. Real model scaler must be normal-Train-only.",
            "Expected Recall/F1 are ratios of expected counts, not exact expectations of nonlinear metric distributions.",
            "window_id remains source_key + window_start. Dummy paired files use disjoint start ranges by user decision.",
        ],
        "environment": {"python": platform.python_version(), "platform": platform.system(),
                        "packages": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "lightgbm", "PyYAML", "pytest"]}},
        "duration_seconds": time.perf_counter() - start,
    }
    output = save_handoff(output_dir, records, summary, pd.concat(weight_tables, ignore_index=True),
                          pd.concat(probability_tables, ignore_index=True), config, config_paths)
    write_json(output / "split_manifest.json", {"config_hash": config_hash(config), "assignments": manifests})
    write_pilot_report(summary, output)
    print(f"Saved {len(records)} estimates to {output}", flush=True)
    return summary, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/pape.yaml")
    parser.add_argument("--base-config", default="config/base.yaml")
    parser.add_argument("--local-config", default=None)
    parser.add_argument("--output-dir", default=None, help="Must not already exist")
    parser.add_argument("--mode", choices=["dummy", "audit", "real"], default="dummy")
    parser.add_argument("--profile", default=None, help="Additional experiment YAML (e.g. config/pape_real.yaml)")
    parser.add_argument("--overlay", action="append", default=[], help="Additional YAML, merged after profile; repeatable")
    args = parser.parse_args()
    paths = [PROJECT_ROOT / args.base_config, PROJECT_ROOT / args.config]
    if args.profile:
        paths.append(PROJECT_ROOT / args.profile)
    for overlay in args.overlay:
        paths.append(PROJECT_ROOT / overlay)
    if args.local_config:
        paths.append(PROJECT_ROOT / args.local_config)
    config = load_config(*paths)
    if args.output_dir:
        output = PROJECT_ROOT / args.output_dir
    else:
        stamp = utc_now().replace(":", "").replace("+", "_")
        output = PROJECT_ROOT / config["paths"]["output_root"] / f"{args.mode}_{stamp}"
    if args.mode == "dummy":
        if config.get("data_origin") != "synthetic_window_feature_fixture":
            raise ValueError("Dummy mode requires the dummy profile")
        run_pilot(config, paths, output)
    else:
        from scripts.run_pape_real import run_real_pilot
        run_real_pilot(config, paths, output, project_root=PROJECT_ROOT, audit_only=args.mode == "audit")


if __name__ == "__main__":
    main()
