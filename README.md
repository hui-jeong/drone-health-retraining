# Drone Health Retraining

## PAPE version_1.0 (007 validation)

Source 000~006과 고정 Hybrid 출력을 사용한 PAPE 릴리스입니다. 기존 탐지기는 재학습하지 않습니다. 실행 방법, RCCDA 연동, 두 인계 자료의 차이와 알려진 실패는 [PAPE version_1.0 인계 안내](docs/PAPE_VERSION_1_0.md)를 확인하세요. PAPE CPU 의존성은 `requirements-pape.lock.txt`로 설치합니다. 실제 결과 데이터는 Git 밖에서 별도 전달합니다.

이 브랜치는 PAPE 변경만 포함합니다. 실행에 필요한 공통 설정·스키마·I/O 네 파일은 안내된 모델 담당자의 고정 커밋과 동일한 내용을 포함합니다.

## APIGraph monthly PAPE

APIGraph 전체 이진 피처·월별 Validation·GAIN/RCCDA 출력은 [PAPE 인계 안내](docs/APIGRAPH_PAPE.md)를 확인하세요. 모델 export를 입력받으며 XGBoost 학습 코드는 모델 브랜치에 분리했습니다.

개인 PC 절대경로, 데이터·모델·Parquet·실험 ZIP은 커밋하지 않습니다. 공통 앱/PAPE 스키마와 기존 드론 버전은 구분해 유지합니다.
