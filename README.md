# Drone Health Retraining

## 기존 모델 연동

기업에서 제공한 Hybrid LSTM-AE v2.3 모델을 이후 재학습 모듈에서 사용할 수 있도록 연동했습니다.

기업 모델 코드는 수정하지 않고, 외부 경로의 기존 모델과 학습 결과를 불러와 Window 단위의 정보를 추출하도록 구성했습니다.

추출하는 정보는 다음과 같습니다.

- AE reconstruction score
- AE percentile score
- Residual probability
- Hybrid score
- Context threshold
- Raw prediction
- Final prediction
- 32차원 latent vector

추출한 결과는 `window_features.parquet` 형태로 저장하며, 이후 PAPE 및 재학습 의사결정 모듈에서 사용할 수 있습니다.

### 주요 파일

- `src/feature_export/model_adapter.py`
  - Hybrid LSTM-AE v2.3 모델 및 학습 artifact 로드
  - Window 단위 score, prediction, latent 추출

- `src/feature_export/export_window_features.py`
  - 추출 결과를 공통 Window Feature 형식으로 변환
  - `window_features.parquet` 생성

- `scripts/validate_dataset.py`
  - 데이터 파일 수, Source Run 수, Feature 수 확인

- `scripts/reproduce_baseline.py`
  - 기존 모델 성능 및 실행 환경 기록

- `scripts/export_window_features.py`
  - 전체 Window Feature 생성

- `scripts/verify_test_export.py`
  - 기존 Test report와 결과 비교

### 데이터 조건

검증에 사용한 데이터 조건은 다음과 같습니다.

- 전체 파일: 360개
- Normal: 180개
- Abnormal: 180개
- Source Run: 180개
- 입력 Feature: 106개
- Window size: 50 timestep
- Stride: 5 timestep
- Train Run: 000~006
- Validation Run: 007
- Test Run: 008~009

### 검증 결과

전체 데이터에 대해 Window Feature를 생성하고 다음 항목을 확인했습니다.

- 전체 Window 수: 267,698
- 전체 컬럼 수: 56
- 32차원 latent 정상 추출
- `window_id` 중복: 0
- 필수 컬럼 결측: 0
- Schema validation: PASS

기존 Test report와 비교한 결과는 다음과 같습니다.

- Test Window 수: 53,514
- `y_true` mismatch: 0
- `raw_prediction` mismatch: 0
- `final_prediction` mismatch: 0

현재 제공된 Hybrid LSTM-AE v2.3 코드와 데이터로 재실행한 성능은 다음과 같습니다.

- Window F1: 0.96598
- Run F1: 0.98592
- 실행 시간: 약 664초
- GPU: NVIDIA GeForce RTX 5060

기존 업무 문서에 기록된 기준값과 재실행 결과에 차이가 있으나, 당시 사용된 정확한 코드/데이터/환경을 확인할 수 없어 차이의 원인은 별도로 특정하지 않았습니다.

### 로컬 경로 설정

기업 모델과 데이터 경로는 사용자 환경마다 다르므로 Git에 포함하지 않습니다.

`config/local.example.yaml`을 참고하여 개인 환경에 맞는 `config/local.yaml`을 작성한 뒤 사용합니다.

`config/local.yaml`은 `.gitignore`에 포함되어 있습니다.

### Git에 포함하지 않는 파일

다음 항목은 저장소에 올리지 않습니다.

- 기업 원본 데이터셋
- 기업 모델 및 checkpoint
- `*.pt`, `*.pth`, `*.joblib`
- `*.npz`
- `*.parquet`
- `outputs/`
- `config/local.yaml`

실제 `window_features.parquet`과 `baseline_manifest.json`은 별도 공유 경로를 통해 전달합니다.