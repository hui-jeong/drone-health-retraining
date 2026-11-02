# RISE2 앱 출력과 PAPE 계약

스키마 버전: `rise2_apigraph_v2`. 앱 한 행이 분류 표본입니다. PAPE 실행 묶음은 별도 소속표로 연결합니다.

## app_features.parquet: Source/Validation 공통 13컬럼

| 컬럼 | 의미 |
|---|---|
| sample_id | NPZ 파일 SHA256 + 원본 행 인덱스. 원본 APK MD5가 아님 |
| origin_file | train/val/test 원파일 식별 (현재 export는 val/test) |
| original_row | 해당 NPZ 원행 인덱스, 0부터 시작 |
| feature_group | 원 uint8 특징 벡터 SHA256. 동일 APK 증명이 아님 |
| observed_month | 배포본 월 메타데이터 YYYY-MM. 실제 유입시각 아님 |
| project_role | source / validation |
| xgb_score | 고정 최종 XGBoost의 class 1 악성 확률, [0,1] |
| final_prediction | uint8, `int(xgb_score > 0.5)` |
| decision_threshold | 0.5 |
| model_hash | model.json 파일 SHA256 |
| threshold_policy_hash | 모델과 연결된 판정 정책의 canonical JSON SHA256 |
| schema_version | rise2_apigraph_v2 |
| feature_schema_hash | 특징 차원·순서·표현 계약 SHA256 |

별도 `labels_for_evaluation.parquet`: `sample_id`, `y_true` (1=악성, 0=정상). Source에서는 PAPE 보정 학습 라벨로 결합합니다. Validation에서는 PAPE 호출 완료 후 실제값 계산에만 결합합니다. 공통 앱 입력에는 정답 컬럼을 넣지 않습니다.

## chunk_membership.parquet

`target_chunk_id`, `mode`, `ordinal`, `sample_id`, `position`, `member_hash`, `is_partial`.

회의 기준은 monthly만 허용합니다. 역할 안의 모든 앱은 정확히 한 월 소속을 갖습니다. Chunk ID는 역할·방식·정렬된 앱 목록·설정 hash로 연결됩니다. 모델/판정 정책 hash를 별도 기록합니다. 과거 count_window 실험은 이번 범위에 포함되지 않습니다.

## PAPE CSV: 원 업무분담과 동일한 14컬럼

```text
target_chunk_id,input_variant,estimated_fn,estimated_fp,estimated_recall,
estimated_f1,estimated_loss,actual_fn,actual_fp,actual_recall,actual_f1,
dre_auc,ess_ratio,max_weight
```

`input_variant`: `all_binary__uniform`, `all_binary__dre`. uniform의 DRE AUC는 빈칸(null)입니다. uniform의 ESS ratio와 최대 가중치는 1입니다. 추정 FN/FP는 확률 합이므로 실수입니다. Actual FN/FP는 실제 건수이며 CSV에서 수치형으로 저장합니다.

`pape_summary.json.records[]`에는 `csv_row_index`(0 시작), seed, mode, 모든 기대/실제 혼동행렬, actual_loss, 지표 정의 여부, 절대 오차, 비용, 모델/정책/Source/설정 hash, 실행시각, 코드 revision, 로그 경로/hash, 진단을 둡니다. 비용 정의는 summary 최상위 `loss_definition`, 실행 전체 설정은 `resolved.yaml`에 있습니다.

PAPE 표는 여러 seed의 개별 행을 유지합니다. 이 표에 seed 컬럼을 추가하지 않았으므로 CSV 행 순서와 summary 매핑을 함께 인계해야 합니다. `actual_*`와 True 표는 평가 전용이며 운영 Gain/RCCDA 특징에 넣으면 안 됩니다.

기대 혼동행렬 정의: p=보정 악성 확률, d=고정 판정.

```text
TP = sum(p[d=1])       FP = sum(1-p[d=1])
FN = sum(p[d=0])       TN = sum(1-p[d=0])
Recall = TP/(TP+FN)    F1 = 2TP/(2TP+FP+FN)
Loss = lambda_fn*FN + lambda_fp*FP
TP+FP = count(d=1)     FN+TN = count(d=0)
```

Recall/F1은 기대 혼동행렬에 지표 공식을 대입한 plug-in 값입니다. 지표 자체의 확률론적 기댓값과 같다는 주장을 하지 않습니다. 분모 0 처리는 `zero_division`, 정의 여부 플래그로 추적합니다.

## 회의 기준 실행과 확장 계약

2012 Train / 2013-2014 초기 Source / 2015 Validation / 2016-2018 Test 보류.
원본 APIGraph 320,315행과 이진 1,159개 피처 순서를 유지합니다. 이번 실행은 Validation 12개월만 허용합니다.
Source/Target의 binary_features.parquet는 sample_id + binary_0000..binary_1158 (uint8)입니다.
13컬럼과 14컬럼을 유지하고 이진 피처를 sidecar로 전달합니다. DRE/보정기 모두 이진 피처 전체를 사용합니다.
원본 schema_version v1 대비 v2는 이 sidecar와 월별 통합 계약을 명시하며, 13컬럼의 순서와 의미를 유지합니다.

confusion_matrix.csv: TP/FP/FN/TN, F1/Accuracy/Recall/Loss의 추정값과 실제값을 함께 제공합니다.
행=true, 열=predicted, label_order=[0,1], [[TN,FP],[FN,TP]]. 실제값은 평가용입니다.
월별 actual 값은 seed/방법별 반복이므로 연간 집계에 한 번만 사용합니다.
추정 연간 지표는 seed별 월 행렬 합계에서 계산한 뒤 평균하며 seed를 서로 다른 앱으로 합산하지 않습니다.

gain_inputs.parquet / rccda_inputs.json: schema_version=pape_gain_rccda_monthly_v1.
추정 특징은 estimated_tp,estimated_fp,estimated_fn,estimated_tn,estimated_recall,estimated_f1,
estimated_accuracy,estimated_loss,estimated_loss_per_app,dre_auc,ess_ratio,max_weight,clip_fraction입니다.
provenance에 월·seed·방법·모델/정책/Source hash·Source 기간·사용 가능 시점을 보존합니다.
운영 입력에는 y_true,actual_*,평가 오차,realized_gain을 넣지 않습니다. RCCDA JSON에는 readiness와 경고도 있습니다.
운영 loader에서 seed 하나와 방법 하나를 명시해 월별 1행을 얻습니다. 진단은 RCCDA readiness와 함께 확인합니다.
자료의 사용 가능 시점은 해당 월 종료 후 다음 달입니다. GAIN 정답 생성과 RCCDA 정책 본체는 후속 통합 범위입니다.

2015 Validation 설정 확정 후 2016 전 Source에 2015를 추가하고, 매년 1월 직전 연도를 누적합니다.
2016 Source=2013-2015 / 2017 Source=2013-2016 / 2018 Source=2013-2017.
직전 연도 전체 라벨이 다음 1월에 제공된다는 가정입니다. 미확보 시 자동 추가하면 안 됩니다.
Source 갱신은 탐지기 재학습과 별개이며, 모델 교체 시 Source/Target 재채점과 보정 재학습이 필요합니다.
source_schedule.py는 일정 검증 함수이고 Test 실행기가 아닙니다. Test는 통합 담당자가 후속 수행합니다.

## 업무분담 출력 대응

모델 담당 window_features.parquet는 공통 13컬럼 app_features의 호환 파일명입니다.
분류 표본은 Android 앱이며 의사결정 Window는 chunk_membership의 월입니다.
baseline_manifest.json은 학습·환경·기준 결과·재현 명령을 담습니다.
드론 시계열 전용 AE/Hybrid/latent 값은 APIGraph에서 생성하지 않습니다.
데이터·모델·Parquet·결과 ZIP은 Git 밖에서 별도 전달합니다.
