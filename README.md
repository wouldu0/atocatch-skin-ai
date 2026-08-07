# 🌿 AtoCatch

**AI 기반 피부 상태 분류 및 아토피 가능성 스크리닝 서비스**

> 피부 사진 업로드 → 5종 피부 상태 분류 → 아토피로 분류된 경우 건강설문 → 스크리닝 결과 및 종합 보고서 제공

![AtoCatch 시연](demo.gif)

---

## 📌 프로젝트 소개

AtoCatch는 피부 이미지를 분석하는 이미지 분류 모델과 KNHANES 건강설문 데이터를 활용한 아토피 가능성 스크리닝 모델을 결합한 AI 헬스케어 프로젝트입니다.

| 구분 | 내용 |
|---|---|
| 개발 기간 | 2026.04 · 약 1개월 |
| 팀 구성 | 4인 |
| 이미지 데이터 | [AI Hub](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71863) 4,500장 + [DermNet](https://www.kaggle.com/datasets/shubhamgoel27/dermnet) 2,167장 |
| 설문 데이터 | [KNHANES](https://knhanes.kdca.go.kr) 7,409명 |
| 이미지 모델 | EfficientNetV2-S + Optuna |
| 설문 모델 | Logistic Regression + Isotonic Calibration |
| 서비스 | Streamlit 멀티페이지 웹앱 |

**핵심 기능**

- **피부 상태 분류** — 정상·아토피·건선·여드름·주사 5종 분류
- **아토피 가능성 스크리닝** — 건강·생활습관 정보를 기반으로 아토피 진단군과 유사한 패턴을 스크리닝
- **종합 분석 보고서** — 이미지 결과, 스크리닝 결과, 예측에 영향을 준 주요 특성 제공

> **문진 모델 해석 범위**: 학습 타깃은 KNHANES의 아토피피부염 의사진단 이력 여부(`DL1_dg`)입니다. 따라서 미래 발병 위험을 예측하는 모델이 아니라, 아토피 진단군에서 관찰된 특성을 바탕으로 현재 사용자의 아토피 가능성을 선별하는 스크리닝 모델입니다.

---

## 👩‍💻 담당 역할

4인 팀(기획 1 · 데이터분석 1 · 기술 2)에서 기술 파트로 참여했습니다.

| 구분 | 담당 내용 |
|---|---|
| 이미지 모델링 | 데이터 전처리, 후보 모델 비교, Optuna 튜닝, 최종 학습·평가 |
| 웹 서비스 | Streamlit 멀티페이지 앱 전체 구현, 로그인·DB, 두 모델 연동 |

> KNHANES 기반 설문 모델링은 팀 내 데이터분석 파트가 수행했으며, 완성된 모델을 웹 서비스에 연동했습니다.

---

## 📊 주요 결과

### 이미지 분류 모델

| 평가 데이터 | Accuracy | Macro F1 |
|---|---:|---:|
| AI Hub hold-out (합성, 500장) | 99.60% | 0.9960 |
| DermNet 재분할 hold-out (실사, 약 323장) | 94.43% | 0.9486 |
| 외부 실사 이미지 스팟체크 (25장) | 52.0% | - |

합성·정제된 데이터에서는 높은 성능을 보였지만, 통제되지 않은 실제 촬영 이미지에서는 성능이 크게 하락해 도메인 갭을 확인했습니다.

<details>
<summary>🔬 왜 EfficientNetV2-S를 선택했는지 보기 — 모델 후보 비교</summary>

5개 후보 모델을 같은 데이터·같은 학습 조건으로 비교했습니다 (`image_model/02_image_model_compare.py`).

| 모델 | 파라미터(M) | 입력크기 | Val Acc | Macro F1 | 학습시간 |
|---|---:|---:|---:|---:|---:|
| **EfficientNetV2-S** | 20.2 | 300 | **0.9745** | **0.9774** | 9.8분 |
| EfficientNet-B2 | 7.7 | 260 | 0.9708 | 0.9751 | 7.4분 |
| ConvNeXt-Tiny | 27.8 | 224 | 0.9684 | 0.9725 | 6.6분 |
| EfficientNet-B4 | 17.6 | 300 | 0.9405 | 0.9487 | 11.7분 |
| MobileNetV3 | 4.2 | 224 | 0.9405 | 0.9474 | 5.8분 |

EfficientNet-B2가 파라미터 수(7.7M)와 학습 시간(7.4분) 면에서는 더 가볍지만, 동일 조건에서 Accuracy·Macro F1 모두 EfficientNetV2-S가 근소하게 앞서 최종 후보로 선택했습니다. 배포 효율을 우선한다면 EfficientNet-B2도 충분히 경쟁력 있는 대안입니다.

이후 Optuna로 30 trials 탐색해 Best Val Macro F1 **0.9844**(7-epoch 기준 탐색)의 하이퍼파라미터를 찾아 최종 학습에 적용했습니다.

> 위 표의 공개 성능은 당시 **Accuracy 기준**으로 선정된 checkpoint의 결과입니다. 이후 코드의 모델 선정 기준은 클래스 불균형을 고려해 **Macro F1**으로 통일했으며, 해당 기준으로 재학습할 경우 결과가 소폭 달라질 수 있습니다.

</details>

### 설문 스크리닝 모델

| 지표 | 결과 |
|---|---:|
| ROC-AUC | 0.786 |
| PR-AUC | 0.3275 |
| 운영 Threshold | 0.11 (Youden's J) |
| Test Recall | 0.67 |
| Test Precision | 0.28 |
| Test F1 | 0.40 |

Threshold는 calibration holdout에서 선정한 뒤 고정하여 최종 test set에 1회 적용했습니다.

---

## 🌐 서비스 흐름

```
피부 사진 업로드
      ↓
EfficientNetV2-S
      ↓
5종 피부 상태 분류
      ↓
아토피가 1위로 예측된 경우
      ↓
7문항 건강설문
      ↓
Logistic Regression + Calibration
      ↓
스크리닝 기준치 이상 / 미만
      ↓
종합 분석 보고서
```

이미지 모델과 설문 모델은 서로 다른 정보를 측정하기 때문에 확률을 합산하지 않고 독립적으로 해석합니다.

---

## 🔍 모델 검증과 의사결정

### 1. 실사 이미지 검증에서 확인한 도메인 갭

AI Hub 합성 이미지에서는 99.60%의 정확도를 보였지만 외부 실사 이미지 25장에서는 52%까지 하락했습니다. 오분류 사례를 확인한 결과 조명·배경·피부톤 등 학습 데이터와 실제 촬영 환경의 차이가 주요 원인으로 판단됐습니다.

이를 개선하기 위해 DermNet "아토피" 라벨 이미지 중 keratosis·ichthyosis 등 다른 질환이 섞여 있던 것을 파일명 기준으로 걸러내 더 정제된 데이터로 재학습했지만, 실사 정확도가 52% → 44%로 오히려 감소했습니다. 해당 버전은 채택하지 않고, 실제 촬영 환경에 대한 일반화 성능을 모델의 한계로 남겼습니다.

### 2. 두 모델의 확률 합산 설계 폐기

초기에는 이미지 모델과 설문 모델의 확률을 합산해 하나의 위험도 점수를 제공하려 했습니다. 그러나 이미지 모델은 사진의 시각적 피부 상태, 설문 모델은 아토피 진단군에서 관찰된 건강 특성을 학습한다는 점에서 두 확률의 의미가 다르다고 판단했습니다.

이에 확률 합산을 폐기하고 두 모델을 독립적으로 해석하는 구조로 변경했습니다.

---

## 🛠️ 기술 스택

Python 3.11 · PyTorch · timm · scikit-learn · Optuna · pandas · Streamlit · SQLite · bcrypt

---

## 📁 데이터

| 데이터 | 출처 | 구성 |
|---|---|---|
| 피부 이미지 | [AI Hub](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71863) | 정상·아토피·건선·여드름·주사 / 4,500장 |
| 피부 이미지 | [DermNet](https://www.kaggle.com/datasets/shubhamgoel27/dermnet) | 아토피·건선·여드름·주사 / 2,167장 |
| 건강설문 | [KNHANES](https://knhanes.kdca.go.kr) 2022–2024 | 아토피 의사진단 이력 기반 / 7,409명 |

AI Hub 데이터는 합성 이미지이며, DermNet은 원본 train/test를 통합한 뒤 이미지 단위로 재분할했습니다.

<details>
<summary><b>🗂️ 레포지토리 구조</b></summary>

```
atocatch-skin-ai/
├── image_model/    # 이미지 전처리·모델 비교·튜닝·학습·평가
├── survey_model/   # KNHANES 전처리·튜닝·캘리브레이션·threshold 분석
├── app/            # Streamlit 서비스
├── models/         # 이미지 모델 산출물
├── requirements.txt
└── README.md
```

</details>

<details>
<summary><b>⚙️ 실행 방법</b></summary>

**환경 설치**
```bash
pip install -r requirements.txt
```

**이미지 모델**
```bash
python image_model/01_image_preprocess.py
python image_model/02_image_model_compare.py
python image_model/03_image_tune_optuna.py
python image_model/04_image_train.py
python image_model/05_image_evaluate.py
```

**설문 모델**
```bash
python survey_model/01_survey_eda_modeling.py
python survey_model/02_prepare_features.py
python survey_model/03_survey_tune_cv.py
python survey_model/04_survey_select_retrain.py
python survey_model/05_threshold_analysis.py
```

**Streamlit**
```bash
streamlit run app/main.py
```

기본 경로는 저장소 루트 기준 상대경로를 사용합니다.

</details>

<details>
<summary><b>🧹 프로젝트 종료 후 리팩터링</b></summary>

포트폴리오 정리 과정에서 저장소와 실행 파이프라인을 다시 검증해 다음 문제를 수정했습니다.

- 누락됐던 설문 피처 생성 스크립트 복원
- 개발 PC 절대경로 → 상대경로 전환
- 근거가 불명확했던 위험 등급(0.10·0.25 기준 낮음/보통/높음)을, calibration holdout에서 재검토한 Youden's J 기준 threshold(0.11)로 교체하고 기준치 이상/미만의 이진 스크리닝으로 단순화
- 이미지 모델 선정 지표를 Macro F1으로 통일
- SHA-256 비밀번호 저장 → bcrypt 전환
- 사용자 데이터·업로드 이미지의 Git 추적 제거
- 실제 사용 패키지 기준 requirements 재정리

</details>

---

## ⚠️ 한계

- 이미지 학습 데이터와 실제 촬영 환경 사이에 도메인 갭이 존재합니다.
- DermNet은 환자 ID가 없어 환자 단위 train/test 분리를 수행하지 못했습니다.
- 설문 모델은 미래 발병 예측이 아닌 현재 아토피 가능성을 선별하는 모델입니다.
- 설문 모델의 PR-AUC가 약 0.33으로 불균형 데이터에 따른 성능 한계가 있습니다.
- Grad-CAM 등 이미지 모델의 설명 가능성 검증이 추가로 필요합니다.

---

## ⚠️ 사용 안내

본 프로젝트는 학습·연구 목적으로 제작되었습니다. 의료 진단을 대체하지 않으며, 실제 피부 질환에 대한 판단은 의료진의 진료가 필요합니다.
