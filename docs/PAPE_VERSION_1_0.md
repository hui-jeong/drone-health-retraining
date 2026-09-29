# PAPE version_1.0 - 007 validation / RCCDA 인계

이 버전은 기존 Hybrid LSTM-AE v2.3의 전달 출력을 고정하고, Source 000~006으로 PAPE를 학습해 Target 007을 평가한 버전이다. 원격 `feature/pape`의 초기 커밋 위에 PAPE 변경만 올린다. 모델 담당자의 model-adapter 커밋 이력과 feature export 구현은 포함하지 않는다. 새 커밋의 범위는 PAPE 모듈, PAPE가 사용하는 지표 계산, 실행·검산·인계 도구, 설정과 관련 테스트다.

실행할 때는 모델 담당자의 공통 인터페이스가 필요하다. 호환성을 검증한 버전은 `feature/model-adapter`의 `caa590cad36fa56aa3351ff3fbe417499612ca46`이며, 필요한 파일은 `config/base.yaml`, `src/common/__init__.py`, `src/common/io.py`, `src/common/schemas.py` 네 개다. 이 파일들은 PAPE 소유 코드로 복제하거나 수정하지 않으며 별도 의존성으로 사용한다. `version_1.0`만 체크아웃한 상태에는 이 의존성이 없으므로 아래 실행 준비가 필요하다.

공개 인터페이스는 `src.pape.estimator.estimate_target_performance(source_df, target_df, config) -> PapeResult`다. 전달된 Target의 정답은 보정/DRE 입력에서 제거되고, 평가 후 실제값과 비교할 때만 사용된다. Scaler/LSTM-AE/Residual/융합/임계값 재학습 도구는 이 릴리스에 포함하지 않는다.

## 범위와 기본값

- PAPE Source: 000~006, 187,428 Window, 원본 126그룹.
- PAPE validation Target: 007, 26,756 Window, 정상·합성 36파일, 원본 18그룹.
- 공식 Test 008~009 제외. 007은 반복 사용한 개발 평가이며 독립 최종 평가가 아니다.
- 기본 연결 입력: Hybrid + DRE, seed 11. 문서 기본값과 구성 seed이며, validation에서 우승한 방법/seed를 골랐다는 뜻이 아니다.
- 비교: AE / Residual / Hybrid / 결합, uniform / DRE, seed 11·23·37·53·71, 총 1,440건.
- DRE와 보정의 특징은 각 입력 구성에서 동일하다. 기본은 `hybrid_score` 하나. 현재 007의 점수+latent 확장 개발은 version_1.0에 포함하지 않는다.
- DRE Source 그룹 OOF 3분할, Target 시간 블록 중첩 제거, weight clipping 0.05~20.
- 손실: `FN + FP`, 실험용 1:1 비용. 회사 확정 비용이 아니다.

## RCCDA 담당자에게 제공하는 두 묶음

| 묶음 | 주 파일의 의미 | 행 수 |
|---|---|---:|
| PAPE_version_1.0_007_validation | Hybrid+DRE, seed 11의 실제 PAPE 추정 | 36 |
| LSTM_AE_true_data_version_1.0_007 | 같은 Hybrid 판정의 실제값, Target 정답을 사용한 oracle 비교 자료 | 36 |

각 묶음의 `pape_estimates.csv`는 업무분담 §5.2의 기존 14컬럼을 유지한다. `pape_summary.json`에는 TP/FP/FN/TN, 실제값·오차, 선택 이유, 진단, 비용, 원본 실험 hash와 실패 조건을 담는다. `chunk_index.csv`의 `target_chunk_id`로 두 묶음을 일대일 결합한다.

PAPE 묶음의 `comparisons/`에는 1,440건 전체 비교를 보존한다. 공유 CSV 자체에는 seed 컬럼이 없으므로, `pape_summary.json.records[].csv_row_index`로 seed·구성을 연결한다. 여러 seed나 정상·합성 짝을 임의의 운영 시간축으로 이어 붙이면 안 된다.

true_data 묶음은 `lstm_ae_true_data.csv`에 실제 혼동행렬을 직접 제공한다. 인터페이스 호환용 `pape_estimates.csv`의 estimated_*도 actual_*와 같게 놓되 `input_variant=lstm_ae_true_data__oracle`로 구별한다. 이는 PAPE 예측이 아니다. oracle 오차 0은 실제값을 복사한 정의의 결과이며 추정기 성능 증거가 아니다. DRE AUC/ESS/weight는 적용되지 않으므로 null이다.

```python
from src.pape.release import load_release_inputs

# 평가용 actual_*를 제거하고 허용된 현재 추정 특징만 읽는다.
# allow_review=True는 알려진 진단 경고를 확인한 오프라인 개발용이다.
pape = load_release_inputs('shared/PAPE_version_1.0_007_validation', allow_review=True)
truth = load_release_inputs('shared/LSTM_AE_true_data_version_1.0_007',
                            allow_review=True, allow_oracle=True)
```

normal loader는 oracle 자료를 기본 거부한다. 두 자료 모두 `chronology_verified=False`다. 36개 파일의 record_id 순서가 실제 비행 수집 시간순이라는 근거가 없으므로, 이 자료만으로 미래 H개 Chunk의 Gain 정답이나 정책의 장기 성능을 구성할 수 없다. 이번 인계는 현재 손실 입력 연결 및 같은 Chunk의 추정/실제값 비교용이다. Gain·Candidate Gate·RCCDA 정책 자체는 구현하지 않는다.

## 실패 조건과 선택 근거

실제 007 전체는 TP=923, FP=24, FN=43, TN=25,766이다. A1 이상 주입 파일은 TP=0, FP=0, FN=39, TN=703인데 기본 PAPE의 FN 추정은 seed 11에서 약 0.00003669다. 다섯 seed 평균은 약 0.00003682다. A1의 raw_prediction도 이상을 모두 놓쳤고 후처리 전후 판정은 같다.

전체 FN MAE의 다섯 seed 평균은 AE 11.193458, Residual 1.412044, Hybrid 1.671985, 결합 1.330887이다. 결합의 평균 오차가 더 작지만 “항상 FN=0” 기준선 1.194444보다 크며, A1 실패도 해결하지 못한다. 기본 Hybrid는 문서상 연결 기준으로 유지한다. 최적 입력 최종 선정이나 운영 승인을 주장하지 않는다.

seed 11의 기본 36개 Chunk 중 22개에 기존 진단 경고가 있다. 경고 없이 반환되더라도 정확도 보장이 아니다. `load_release_inputs`는 경고 있는 자료를 기본 거부하며, 연구 소비에 한해 `allow_review=True`를 명시한다. 입력에서 actual_*를 사용하지 말고, 정책 결과 평가와 oracle 비교를 분리한다.

전달 export와 저장 trial hash는 검산했지만, 정확한 baseline checkpoint/학습 source_key의 계보는 인증되지 않았다. PAPE가 계산한 낮은 손실만으로 KEEP이나 재학습 불필요를 확정하지 않는다.

## 환경과 실행

Python 3.10 기준 CPU 환경은 `requirements-pape.lock.txt`를 사용한다. PAPE 자체는 GPU 모델 실행을 요구하지 않는다. 다음 명령은 새 실행용 폴더에서만 수행한다. 다른 담당자의 브랜치를 수정하거나 푸시하지 않고, 고정한 공통 인터페이스 네 파일만 실행용 작업 트리에 가져온다. 해당 파일을 PAPE 커밋에 추가하지 않는다.

```powershell
git clone --branch version_1.0 https://github.com/hui-jeong/drone-health-retraining.git pape-runtime
Set-Location pape-runtime
git fetch origin feature/model-adapter
git restore --source=caa590cad36fa56aa3351ff3fbe417499612ca46 --worktree -- config/base.yaml src/common/__init__.py src/common/io.py src/common/schemas.py
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-pape.lock.txt
.\.venv\Scripts\python.exe -m pytest -q
```

이 의존성에는 모델 추론·재학습 코드가 포함되지 않는다. 이미 공통 인터페이스가 통합된 팀 실행 환경에서는 별도 가져오기 없이 PAPE 변경을 적용하면 된다. `src/common/metrics.py`는 이번 PAPE가 제공하는 지표 계산 코드다. 다른 모듈 담당자의 Git 브랜치와 기존 파일을 변경하지 않는다.

완료된 원본 실험을 다시 학습하지 않고 새 인계 묶음을 만드는 명령:

```powershell
python scripts/export_pape_release.py --experiment-dir outputs/pape/frozen_handoff_000006_20260921 --target-features outputs/baseline/frozen_handoff_000006_20260921/pape_validation/window_features.parquet --output-dir deliverables/version_1_0
```

exporter는 해당 007 Parquet의 hash를 먼저 검증하고, 실제 혼동행렬을 다시 계산하며, 기존 CSV/JSON 1,440행을 대조한다. 출력 경로가 이미 있으면 덮어쓰지 않는다. 원본 자료, 출력 ZIP, Parquet, trial weight/probability 대용량 자료는 Git에 넣지 않는다. 필요 파일은 별도 인계 묶음으로 공유한다.

새 환경에서 같은 방법을 재현해야 하는 경우에만 경로 예제를 `config/pape_version_1_0.local.yaml`로 복사하고 다음을 사용한다. 기존 완료 결과 폴더에 재실행하지 않는다.

```powershell
python scripts/prepare_pape_frozen_handoff.py --config config/pape.yaml config/pape_version_1_0.yaml config/pape_version_1_0.local.yaml
python scripts/run_pape_fixed_reference.py --config config/pape.yaml config/pape_version_1_0.yaml config/pape_version_1_0.local.yaml
python scripts/verify_pape_fixed_reference.py --config config/pape.yaml config/pape_version_1_0.yaml config/pape_version_1_0.local.yaml
```

원본 실험의 개인 경로와 새 환경의 경로가 다르면 설정 hash는 달라진다. 릴리스 인계는 원본 `original_study_config_hash`를 보존한다. Git은 `feature/pape`, 배포 표시는 `version_1.0` 태그를 사용한다. 업무분담 문서 §7은 main, 담당별 feature 브랜치와 PR·1명 이상 리뷰 후 병합을 정한다. develop은 원문에 명시되지 않았다. 이 릴리스는 PAPE 브랜치와 태그 게시이며, 다른 담당자 브랜치나 통합 브랜치로 병합하지 않는다.

## 입력 계약과 Chunk 식별

공개 estimator는 문서 §5.1의 공통 50컬럼을 받는다. Target 정답 관련 4컬럼은 제거한 후 계산하므로 라벨 없는 46컬럼으로도 호출할 수 있다. `record_id`, `start_idx`, `split`은 공개 estimator의 필수 컬럼이 아니다. `source_key`는 정상 원본과 파생 합성 파일이 공유하는 분할 그룹이며, 파일 식별자로 사용하지 않는다.

명시한 `config.target_chunk_id`를 우선 사용한다. 없으면 해당 Run의 고유 `window_id` 목록을 정렬해 SHA256으로 식별한다. 라벨이나 이상 유형은 ID 생성·모델 특징에 사용하지 않는다. 파일별 Window ID는 전역 고유해야 한다. 일부 Window만 전달하면 ID가 달라지므로 동일 파일의 안정적인 외부 식별자가 있으면 명시적으로 넘긴다. 기존 007 인계는 명시한 파일 ID를 그대로 사용한다.

`inspect_handoff`, `load_reference_pool`, frozen export 실행기는 특정 모델 인계 형식 전용 어댑터다. 해당 형식의 `record_id`는 파일 단위 정수, `start_idx`는 0부터 시작하는 샘플 인덱스, `split`은 기존 모델의 train/validation/test 구분이다. `record_idx`는 있으면 읽는 선택 항목이다. 이 추가 필드는 공통 50컬럼 스키마의 변경이 아니다. 전용 입력의 조건을 공통 스키마 구현자에게 암묵적으로 요구하지 않는다. 공통 형식만 받는 경우 estimator에 단일 파일을 전달한다. 공통 명세를 바꾸는 작업은 담당자 간 합의가 필요하다.

## 2026-09-29 인계 수정

- 비용, 기본 seed, 정의 불가능한 비율의 대체값은 원본 `config/resolved.yaml`에서 읽고 설정 hash를 검증한다. `--seed`를 사용하면 override를 별도로 기록한다.
- `config/resolved_portable.yaml`은 모든 설정을 보존하며 개인 절대경로만 교체한다. `effective_configs_portable.json`은 행별 원본 config hash와 이식한 전체 설정·새 hash를 연결한다. 두 hash를 같다고 보고하지 않는다.
- 두 ZIP과 함께 생성하는 `PAPE_reference_artifacts_version_1.0/`에는 원본 YAML, 행별 원본 설정, 원시·clipping 가중치, Target 확률을 보관한다. `diagnostics_manifest.json`은 상대경로·SHA256·행 식별자를 제공한다. 원본 설정의 개인 경로가 있으므로 Git 밖에서 전달한다. ZIP 두 개만 전달하면 대형 진단 자료는 포함되지 않는다.
- PAPE ZIP의 `comparisons/loss_sensitivity.csv`와 집계 JSON에 원본 비용 민감도 결과를 제공한다. 기존 FN/FP 및 추정 수치는 변경하지 않는다.
- RCCDA용 입력의 provenance는 config hash, 실행시각, 코드 revision, 비용, bundle ID를 보존한다. 실제 정답은 추정 특징에서 제외하며, oracle은 별도의 명시적 허용이 필요하다.
- 기본 입력은 `reference_default_final_selection_pending`으로 표시한다. 최종 입력 선정 및 A1 미탐 문제 해결을 완료했다고 주장하지 않는다.
- NPZ와 개인 local YAML을 Git에서 제외한다. 실험 실행 시 브랜치·커밋·변경 유무를 기록하되 태그나 통합 브랜치에서 실행하는 것을 막지 않는다.
