# APIGraph 월별 PAPE 인계

## 범위와 데이터

엔스퀘어 업무분담문서 4.2·5.2·6·8.1절을 APIGraph 데이터에 적용한 신규 어댑터다.
기존 드론 PAPE version_1.0과 별도 이름공간이며 기존 Run/Window/latent 스키마를 바꾸지 않는다.
2012년 새 XGBoost, PAPE Source 2013-2014, Validation 2015의 월별 12개를 사용한다.
Test는 비활성화한다. 원 320,315행과 이진 피처 1,159개의 원 순서를 유지한다.
2016-2018 Test는 통합 담당자가 추후 실행한다. 이 브랜치의 Test 실행 경로는 차단되어 있다.
2015 Validation에서 설정을 확정한 뒤, 2016 시작 전 2015 라벨을 Source에 추가한다.
이후 매년 1월 직전 연도까지 누적한다(2017: 2013-2016, 2018: 2013-2017).
직전 연도 전체 라벨이 다음 1월에 확보된다는 가정이며, 확보되지 않으면 갱신을 지연해야 한다.
source_schedule.py는 설정 확정/직전 연도 완료를 검사하는 인계용 계획 함수이며 Test를 실행하지 않는다.

## 입력 및 실행

모델 담당의 완료된 export를 명시적으로 전달한다. XGBoost 학습 코드 없이 PAPE 브랜치 단독으로 실행할 수 있다.
모델 담당이 만든 인계 디렉터리에도 동일한 역할별 export 파일이 포함되어 있어 입력으로 사용할 수 있다.

```powershell
python -m pip install -r requirements-apigraph.lock.txt
python scripts/run_pape_apigraph.py run --config config/apigraph.yaml --export-run MODEL_EXPORT
python scripts/run_pape_apigraph.py verify --config config/apigraph.yaml
python -m unittest discover -s tests/apigraph -v
```

MODEL_EXPORT는 로컬 입력값이다. 개인 PC의 경로는 local.yaml에만 보관하고 커밋하지 않는다.
데이터/출력 경로를 제외한 의미 설정이 export 설정과 같은지 확인한다.
서로 다른 모델 담당/PAPE 코드 fingerprint를 동일하다고 가정하지 않고, 모델 hash와 앱·파일 계약으로 검증한다.

## 구현

DRE와 LGBMRegressor 보정기 모두 원 이진 피처 1,159개를 사용한다. xgb_score는 고정 판정·진단에만 쓴다.
3-fold OOF에서 동일 피처 패턴은 양 도메인에 걸쳐 같은 holdout으로 묶는다.
도메인 학습 사전비율을 보정한 Source 가중치를 0.05-20으로 제한하고 uniform과 비교한다.
월별 Target 정답은 추정 호출 전에 제거하며, 저장한 추정 결과의 평가 단계에서만 정답을 결합한다.
seed 11/23/37 × uniform/dre × 12개월 = 72회다. 탐지 모델 판정은 바꾸지 않는다.

## 기존 14컬럼 출력 유지

```text
target_chunk_id,input_variant,estimated_fn,estimated_fp,estimated_recall,
estimated_f1,estimated_loss,actual_fn,actual_fp,actual_recall,actual_f1,
dre_auc,ess_ratio,max_weight
```

input_variant는 all_binary__uniform 또는 all_binary__dre다. csv_row_index로 summary의 seed를 연결한다.
전체 TP/FP/FN/TN과 Accuracy는 summary 및 confusion_matrix.csv로 별도 제공한다.
GAIN/RCCDA 입력은 gain_inputs.parquet와 rccda_inputs.json이며 실제값·정답·오차를 포함하지 않는다.
월 종료 후 다음 달부터 판단에 쓰고, 여러 seed/방법을 같은 달의 독립 관측으로 합산하지 않는다.
Source/모델 버전이 바뀌면 재채점·재보정해야 한다. 정책 본체와 Gain 정답 생성은 후속 모듈이다.

## 검증·실패 조건

파일 무결성, 13/14컬럼, 전체 행렬, 라벨 프록시 제거, 그룹 OOF, 월별 소속, Test 차단을 검사한다.
ESS 부족과 clipping 경고는 생성 검증 PASS와 구분한다. DRE AUC가 높다고 좋은 추정으로 선택하지 않는다.
ESS 부족과 clipping 경고는 실제 2015 Validation 실행 결과와 함께 검토한다.
이전 임시 2016 Validation 수치는 회의 기준 결과로 사용하지 않는다.
기본 workers=4로 독립 실험을 병렬 계산하고 월/seed/방법 순서로 저장한다.
직렬/병렬 확률과 가중치가 동일한지 회귀 검증한다. 각 LightGBM은 n_jobs=2다.

## 저장소·공통 계약

데이터·모델·Parquet·ZIP과 outputs는 Git에서 제외한다. 실행 YAML에는 저장소 상대경로를 사용한다.
기존 드론 테스트에 필요한 공통 네 파일은 README에 명시된 고정 모델-adapter 커밋에서 그대로 가져왔다.
새 APIGraph 공통 모듈은 feature/modelXGBoost와 동일하다. 기존 드론 common 스키마를 APIGraph 스키마로 대체하지 않았다.
main 병합은 PR과 리뷰를 통해 수행한다. 이번 업로드는 feature/pape에 한정한다.
