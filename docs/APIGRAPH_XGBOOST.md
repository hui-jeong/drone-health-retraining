# APIGraph XGBoost 모델 담당 인계

## 범위

엔스퀘어 업무분담문서 4.1·5.1·6·8.1절의 모델 연동/출력 책임을 APIGraph 앱 데이터에 맞게 구현한다.
Train 2012, PAPE Source 2013-2014, Validation 2015이며 이번 검증에서 Test는 사용하지 않는다.
2016-2018 Test는 통합 후 다른 담당자가 실행한다. 2016 전 Source에 2015를 추가하고 매년 직전 연도를 누적한다.
원 APIGraph 320,315행과 1,159개 이진 피처, 원행 유지 정책을 사용한다.
분류 표본은 앱 1행, 추정/정책의 Window는 달력 월이다. 드론의 5초 시계열이나 latent를 만들어 넣지 않는다.

## 실행

```powershell
python -m pip install -r requirements-apigraph.lock.txt
python scripts/run_xgboost_apigraph.py train --config config/apigraph.yaml
python scripts/run_xgboost_apigraph.py export --config config/apigraph.yaml
python -m unittest discover -s tests/apigraph -v
```

개인 데이터 위치를 바꿀 때는 무시되는 local.yaml에 설정을 복사하고 --config로 지정한다. 개인 절대경로를 커밋하지 않는다.
학습은 2012년 내부 그룹 분리로 트리 수를 정한 뒤 2012년 전체로 새 모델을 학습한다.
기존 checkpoint를 사용하지 않는다. 기본 판정은 xgb_score > 0.5다.

## 업무분담 출력

- window_features.parquet: 앱 단위 13컬럼. 전체 Source/Validation 합본과 역할별 파일을 제공한다.
- baseline_manifest.json: 학습/모델/정책/특징 hash, seed, CPU 사용, 라이브러리, 시간, 기준 성능, 재현 명령.
- binary_features.parquet: sample_id + binary_0000..binary_1158. 원 이진 피처 순서를 유지한다.
- labels_for_evaluation.parquet: sample_id,y_true. Target 정답을 별도 보관한다.
- chunk_membership.parquet: 달력 월 소속표.
- model_monthly_metrics.csv 및 confusion_matrix.csv: 월별 실제 성능과 전체 행렬. 평가 전용.
- model_results.csv: Source/Validation 전체 성능.
- training_baseline_manifest.json: 기존 실제 학습 manifest 원문 보존.
- model_output_manifest.json 및 verification.json: 인계 파일 hash와 검증 결과.

앱 13컬럼은 sample_id,origin_file,original_row,feature_group,observed_month,project_role,xgb_score,
final_prediction,decision_threshold,model_hash,threshold_policy_hash,schema_version,feature_schema_hash다.
sample_id는 NPZ SHA256+원행 번호이며 APK MD5가 아니다. feature_group은 원 이진 벡터 hash다.
Window Feature라는 기존 파일명은 유지하되 행 단위가 앱임을 명시한다. Target y_true와 AE/Hybrid/latent 컬럼을 공통 13컬럼에 추가하지 않는다.
전체 행렬의 방향은 행=true, 열=predicted, label_order=[0,1], [[TN,FP],[FN,TP]]다.

인계 생성은 다음 인터페이스로 수행한다. 대문자 인자는 수신자의 로컬 경로 값이며 커밋하지 않는다.

```powershell
python scripts/run_xgboost_apigraph.py handoff --model-run MODEL_RUN --export-run EXPORT_RUN --output HANDOFF_DIR
python scripts/run_xgboost_apigraph.py verify-handoff --output HANDOFF_DIR
```

원 데이터 NPZ는 인계 ZIP에 포함하지 않는다. 모델·Parquet·출력 ZIP은 Git 밖에서 별도 전달한다.

## PAPE 연계와 Git 규칙

PAPE는 app_features.parquet, 이진 sidecar, 별도 Source 라벨과 월 소속표를 읽는다.
모델/정책/Source hash를 유지하고 PAPE 담당자의 스키마를 임의로 변경하지 않는다.
공통 APIGraph 코드는 두 브랜치에서 동일하게 보존하고 기능 변경은 해당 feature 브랜치에 코드·검증과 함께 커밋한다.
main 병합은 업무분담문서의 PR·리뷰 절차를 따른다. 이번 반영은 feature/modelXGBoost에 한정한다.
