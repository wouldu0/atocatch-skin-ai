# 🌿 AtoCatch

**AI 기반 피부 이미지 아토피 스크리닝 및 케어 솔루션**

> 피부 사진 업로드 + 생활습관 문진 → 5종 피부 질환 분류 & 아토피 관련 특성 분석 → 맞춤형 보고서 제공

![AtoCatch 시연](demo.gif)

---

## 📌 프로젝트 한눈에 보기

아토피 피부염은 국내 진료 환자 수 약 97만 명(2024, [건강보험심사평가원 국민관심질병통계](https://www.hira.or.kr))에 달하는 만성 질환입니다. 19세 이상 성인 유병률은 질병관리청 국민건강영양조사 기준 2013년 3.4%에서 2022년 6.3%로 10년간 약 2배 증가했습니다. 기존 진단은 병원 방문과 전문의 대면 평가에 의존하여 일상적 스크리닝이 어렵다는 한계가 있습니다.

AtoCatch는 두 가지 AI 모델을 결합해 이 문제를 해결합니다.

1. **이미지 분류 모델** — 피부 사진으로 5종 질환(정상/아토피/건선/여드름/주사)을 분류
2. **문진형 관련도 분석 모델** — KNHANES 건강 설문 데이터 기반으로, 아토피 진단 이력과 통계적으로 관련된 생활습관 특성을 분석

> 문진 모델의 타깃은 KNHANES의 `DL1_dg`(아토피피부염 **의사진단 이력** 여부, 0/1)입니다. 즉 "미래에 아토피가 생길 확률"을 예측하는 모델이 아니라, 현재 특성과 과거 진단 이력 사이의 통계적 관련성을 학습한 분류 모델입니다.

| 항목 | 내용 |
|---|---|
| 데이터 | AI Hub 피부질환 이미지 4,500장 + Kaggle DermNet 2,167장, KNHANES 건강설문 7,409명 |
| 개발 기간 | 2026.04 (약 1개월), 4인 팀 |
| 이미지 모델 | EfficientNetV2-S (timm), Optuna 튜닝 |
| 문진 모델 | Logistic Regression (KNHANES 기반, 진단 이력 관련도 분류) |
| Test Accuracy (전체) | 97.57% (Macro F1 0.9794) |
| 문진 모델 ROC-AUC | 0.786 |
| 서비스 | Streamlit 4페이지 웹앱 |

---

## 👩‍💻 담당 역할

4인 팀(기획 1 · 데이터분석 1 · 기술 2) 중 기술 파트 2명 중 1명으로 참여했습니다.

| 구분 | 담당 내용 |
|---|---|
| 이미지 모델링 | 데이터 전처리, 후보 모델 비교, Optuna 튜닝, 최종 학습·평가 (`image_model/`) |
| 웹 서비스 | Streamlit 멀티페이지 앱 전체 구현 — 로그인/DB 설계, 4개 페이지, 두 모델 연동 (`app/`) |

> 문진 관련도 모델(`survey_model/`)은 팀 내 데이터분석 파트가 KNHANES 데이터를 전처리·모델링한 결과를 넘겨받아 웹 서비스에 연동했습니다.

---

## 📊 주요 성능

| 평가셋 | Accuracy | Macro F1 |
|--------|----------|----------|
| AI Hub hold-out (합성 이미지, 500장) | 99.60% | 0.9960 |
| DermNet 재분할 hold-out (실제 이미지, ~323장) | 94.43% | 0.9486 |
| 전체 합산 | 97.57% | 0.9794 |

| 문진 모델 | ROC-AUC | PR-AUC |
|----------|---------|--------|
| Logistic Regression (class_weight=balanced) | 0.786 | 0.3275 |

> 실제 촬영 환경의 이미지에서는 성능이 더 떨어집니다 — 자세한 원인 분석은 [트러블슈팅](#-트러블슈팅) 참고.

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

이후 Optuna로 30 trials 탐색한 결과 Best Val Macro F1 **0.9844**(7-epoch 기준 탐색)의 하이퍼파라미터(`lr=2.71e-4, weight_decay=2.52e-3, drop_rate=0.101, label_smoothing=0.103, batch_size=64`)를 찾아 최종 학습(`04_image_train.py`)에 적용했습니다.

</details>

---

## 🗂️ 레포지토리 구조

```
AtoCatch/
│
├── image_model/                     # 이미지 분류 파이프라인
│   ├── 01_image_preprocess.py       # AI Hub + DermNet 데이터 전처리 및 300px 리사이즈
│   ├── 02_image_model_compare.py    # 후보 모델 비교 (EfficientNetV2-S 외 4종, 10 epoch)
│   ├── 03_image_tune_optuna.py      # Optuna 하이퍼파라미터 자동 탐색 (30 trial)
│   ├── 04_image_train.py            # 최적 HP 적용 풀학습 (20 epoch, warmup + cosine)
│   ├── 05_image_evaluate.py         # 최종 테스트 평가 (AI Hub / DermNet / 전체)
│   └── experiments/                 # DermNet 아토피 정제 실험 (v4, 미채택)
│       ├── preprocess_v2.py         # DermNet 아토피 클래스 정제 전처리
│       ├── train_mixed_v4.py        # v4 재학습
│       ├── evaluate_test_mixed_v4.py
│       ├── test_real.py             # 실사 이미지 스팟체크 (v3 기준)
│       └── test_real_v4.py          # 실사 이미지 스팟체크 (v4)
│
├── survey_model/                    # 문진형 관련도 분석 파이프라인
│   ├── 01_survey_eda_modeling.py    # KNHANES EDA, 전처리, 로지스틱 회귀 분석
│   ├── 02_prepare_features.py       # 01의 출력 → 모델 입력 피처(model_dataset.csv) 생성
│   ├── 03_survey_tune_cv.py         # 5-Fold CV 하이퍼파라미터 튜닝 (LogReg/RF/LightGBM/MLP)
│   ├── 04_survey_select_retrain.py  # Best 모델 선정 → 재학습 → 테스트 평가 + 캘리브레이션
│   └── 05_threshold_analysis.py     # Calibration holdout 기준 운영 threshold 산출
│
├── app/                             # Streamlit 웹 애플리케이션
│   ├── main.py                      # 진입점 (로그인 / 회원가입)
│   ├── pages/
│   │   ├── 1_피부질환_예측.py        # 이미지 업로드 및 질환 예측
│   │   ├── 2_아토피_설문.py          # 7문항 생활습관 설문
│   │   ├── 3_위험도_보고서.py        # 종합 보고서 (HTML 다운로드 포함)
│   │   └── 4_내_기록.py             # 기록 히스토리 및 추이 그래프
│   └── utils/
│       ├── predict.py               # EfficientNetV2-S 추론 래퍼
│       ├── atopy_model.py           # 문진 모델 추론 + 예측 기여 특성 산출
│       ├── db.py                    # SQLite CRUD (users / records)
│       └── style.py                 # 전역 CSS 스타일
│
└── README.md
```

---

## 🛠️ 기술 스택

| 분류 | 라이브러리 |
|------|-----------|
| 딥러닝 | PyTorch, timm (EfficientNetV2-S) |
| 머신러닝 | scikit-learn, LightGBM, imbalanced-learn |
| 하이퍼파라미터 튜닝 | Optuna |
| 데이터 처리 | pandas, numpy, Pillow |
| 시각화 | matplotlib, seaborn |
| 웹 앱 | Streamlit |
| DB | SQLite |
| 언어 | Python 3.11 |

---

## 📁 데이터셋

| 데이터 | 출처 | 클래스 | 장수 |
|--------|------|--------|------|
| 피부 질환 이미지 | [AI Hub 안면부 피부질환 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71863) | 정상/아토피/건선/여드름/주사 | 4,500 |
| 피부 질환 이미지 | [Kaggle DermNet](https://www.kaggle.com/datasets/shubhamgoel27/dermnet) | 아토피/건선/여드름/주사 | 2,167 |
| 건강 설문 | [KNHANES](https://knhanes.kdca.go.kr) (2022~2024) | 아토피 진단 여부 (binary) | 7,409명 |

> **주의:** AI Hub 데이터는 합성(GAN) 이미지입니다. 실제 이미지(DermNet) 대비 약 5%p 성능 차이가 발생하며, 이는 도메인 갭에 의한 것입니다.

**데이터 구성 과정에서의 결정**

- AI Hub 데이터에 정면·측면 이미지가 섞여 있어 두 구성을 비교한 결과 측면 단독 구성의 정확도가 더 높아, 측면 이미지만 사용하기로 확정했습니다.
- 지루성피부염(Seborrheic) 클래스는 아토피와 혼동 가능성이 높아 후보에서 제외했습니다.
- 클래스별 이미지 수 차이를 보정하기 위해 Class Weighting(클래스별 샘플 수 역비례 가중치)을 적용했습니다.

---

## ⚙️ 실행 방법

### 1. 환경 설치

```bash
pip install -r requirements.txt      # 전체 파이프라인(학습 포함)
# 웹 앱만 실행할 경우:
pip install -r app/requirements.txt
```

### 2. 이미지 모델 학습 (순서대로 실행)

```bash
python image_model/01_image_preprocess.py
python image_model/02_image_model_compare.py
python image_model/03_image_tune_optuna.py
python image_model/04_image_train.py
python image_model/05_image_evaluate.py
```

### 3. 문진 관련도 모델 학습 (순서대로 실행)

```bash
python survey_model/01_survey_eda_modeling.py    # 원본 KNHANES 전처리 + EDA
python survey_model/02_prepare_features.py       # 모델 입력 피처 생성
python survey_model/03_survey_tune_cv.py         # 5-Fold CV 튜닝
python survey_model/04_survey_select_retrain.py  # Best 모델 재학습 + 캘리브레이션
python survey_model/05_threshold_analysis.py     # 운영 threshold 산출
```

### 4. 웹 앱 실행

`app/config.py`의 경로를 본인 환경에 맞게 수정한 후 (아래 "경로 설정" 참고):

```bash
streamlit run app/main.py
```

---

## 🔧 경로 설정

`app/config.py`는 기본적으로 저장소 루트 기준 상대경로를 사용합니다 — `image_model/04_image_train.py`와 `survey_model/04_survey_select_retrain.py`를 순서대로 실행했다면 별도 수정 없이 그대로 동작합니다.

```python
# app/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_MODEL_PATH  = PROJECT_ROOT / "image_model" / "models" / "mixed_v3" / "efficientnetv2_s_tuned.pth"
SURVEY_MODEL_PATH = PROJECT_ROOT / "survey_model" / "outputs" / "final_model" / "best_model.pkl"
UPLOAD_DIR        = PROJECT_ROOT / "app" / "data" / "uploads"
DB_PATH           = PROJECT_ROOT / "app" / "data" / "skin_app.db"
```

모델 파일을 다른 위치에 두었다면 이 네 경로만 수정하면 됩니다. `image_model/`, `survey_model/` 스크립트도 각 파일 상단에서 저장소 루트 기준 상대경로(`data/`, `models/`, `outputs/`)를 기본값으로 사용하므로, 데이터를 다른 위치에 두었을 때만 해당 블록을 수정하세요.

---

## 🌐 솔루션 흐름

```
사용자
  │
  ├─ 피부 사진 업로드
  │       └─ EfficientNetV2-S → 5종 질환 확률 (Softmax)
  │               │
  │               └─ 아토피 1위 예측 시 → 7문항 생활습관 설문
  │                       └─ Logistic Regression → 아토피 진단 이력 관련 예측 확률 (0~1)
  │                               └─ 예측에 영향을 준 주요 특성 TOP 3 + 참고 정보
  │
  └─ 종합 보고서 (HTML 저장 / PDF 인쇄 가능)
```

---

## 🔍 트러블슈팅

### 1. 실사 이미지에서 드러난 도메인 갭

**문제** AI Hub 합성 이미지 기준 99.60%였던 정확도가, 인터넷에서 직접 수집한 실사 이미지 25장으로 테스트하자 52%까지 떨어졌습니다. DermNet 실사 이미지(94.43%)에서도 이미 5%p 가량 하락한 상태였는데, 통제되지 않은 환경의 이미지에서는 격차가 훨씬 컸습니다. 클래스별로 뜯어보면 하락 폭도 균일하지 않았습니다 — 건선은 5/5(100%)를 맞혔지만 주사는 0/5(0%), 정상은 1/5(20%)에 그쳤습니다.

**분석** 오분류 케이스를 직접 확인한 결과, 주사→아토피/여드름 오분류는 붉은 발진이 시각적으로 유사해서, 정상→아토피 오분류는 학습 데이터 자체가 붉은 피부톤 이미지에 편향돼 있어서 발생한 것으로 파악했습니다. 즉 단순한 데이터 부족이 아니라, 학습 데이터가 조명·배경·해상도가 표준화된 합성·임상 이미지 위주였다는 구조적 원인이었습니다. 정확도 하나로 성능을 요약하지 않고 클래스별 breakdown과 실제 오분류 사례를 뜯어봐서 원인을 구체적으로 짚어내는 과정을 거쳤습니다.

### 2. 두 모델을 합치려다 갈아엎은 서비스 설계

**문제** 서비스 설계 초안은 "아토피 확률 20% 이상이면 설문 진행 → 이미지 모델과 문진 모델의 확률을 합산해 하나의 위험도 점수로 출력"하는 구조였습니다.

**전환** 개발 도중, 이미지 모델은 "사진이 아토피처럼 보이는가"를 답하고 문진 모델은 "생활습관상 과거 아토피 진단 이력과 관련이 있는가"를 답하는, 애초에 서로 다른 질문에 대한 답이라는 걸 인지했습니다. 두 확률을 단순 합산하는 게 개념적으로 맞지 않다고 판단해 두 모델을 완전히 분리 운영하는 구조로 다시 설계했습니다. 최종적으로는 아토피가 1위로 예측된 경우에만 설문을 진행하고, 합산 점수 대신 예측에 영향을 준 특성(계수 × 입력값 기여도) 목록만 보여주는 방식으로 바꿨습니다. 의료 스크리닝 서비스에서 근거가 약한 합산 점수를 섣불리 제시하는 것의 위험성을 고려한 결정이기도 합니다.

### 3. 근거가 불명확했던 위험 판정 기준(threshold)

**문제** 앱은 원래 예측 확률 25% 이상이면 "높음", 10% 이상이면 "보통"으로 3단계 표시했는데, 이 두 숫자의 근거가 코드에 남아있지 않았습니다.

**분석** 팀 자료를 다시 확인한 결과, calibration holdout set에서 Youden's J(민감도+특이도 균형)와 F1-optimal 두 기준으로 각각 threshold를 산출해둔 분석(`05_threshold_analysis.py`)이 별도로 있었습니다. 두 기준으로 각각 0.11, 0.17이 나왔고, test set에는 이 값을 고정해 1회만 적용해 성능을 측정한 상태였습니다(F1 0.40 / Precision 0.29~0.30 / Recall 0.62~0.67 수준).

**개선** Youden's J 기준 threshold(0.11)를 운영 기준으로 채택해 앱에 반영했습니다. 기존 3단계(낮음/보통/높음) 대신 기준치 이상/미만의 이진 판정으로 단순화했습니다 — 분석으로 확보한 운영점은 하나뿐이라, 등급을 더 나누면 그 경계에 대한 별도 통계적 근거가 없어지기 때문입니다. F1-optimal 값(0.17)은 비교 참고용으로 코드에 남겨뒀습니다.

### 4. DermNet 아토피 클래스 정제 실험 (v4, 미채택)

DermNet "Atopic Dermatitis Photos" 폴더에 keratosis·ichthyosis 등 혼입 이미지가 섞여 있어, 파일명에 `atopic`이 포함된 이미지만 남기고 재학습을 시도했습니다 (`image_model/experiments/`).

| 버전 | 실사 이미지 스팟체크 정확도 |
|------|------------------------------|
| v3 (현재 채택 모델) | **52.0%** |
| v4 (아토피 클래스 정제) | **44.0%** |

정제된 데이터로 학습했음에도 실제 사진 테스트에서 오히려 성능이 하락해 **미채택**했습니다. 코드와 결과는 재현·참고용으로 남겨둡니다.

---

## 🧹 프로젝트 종료 후 개인 개선

포트폴리오로 정리하며 배포 중이던 저장소를 다시 검토했고, 실제 문제들을 발견해 수정했습니다.

- **개인정보 노출**: `.gitignore`에 `app/data/`를 등록해뒀지만 파일명 오타(`gitignore` → `.gitignore`)로 무시 규칙이 전혀 동작하지 않아, 실사용자 계정(비밀번호 해시)과 업로드된 피부 사진 3장이 공개 저장소에 그대로 커밋돼 있었습니다. git 히스토리에서 완전히 제거했습니다.
- **파일 내용 뒤바뀜**: `03_image_tune_optuna.py`와 `04_image_train.py`의 실제 코드 내용이 서로 바뀐 채로 커밋돼 있어서, 파일명만 보고 실행하면 의도한 것과 다른 스크립트가 돌아가는 상태였습니다. 로컬 원본과 대조해 내용을 맞바꿨습니다.
- **설문 파이프라인 복원**: README에는 `01→02→03` 3단계로만 설명돼 있었지만, 실제로는 01의 출력(`preprocessed_for_ta.csv`)을 모델 입력 피처로 바꿔주는 중간 스크립트가 저장소에서 빠져 있었습니다. 팀 자료에서 실제 사용했던 스크립트를 찾아 `02_prepare_features.py`로 추가해 파이프라인을 다시 연결했습니다.
- **위험 판정 기준 재정비**: 근거가 불명확했던 3단계 threshold(0.10/0.25)를, 실제 분석 스크립트(`05_threshold_analysis.py`)를 근거로 Youden's J 기준 이진 threshold(0.11)로 교체했습니다. 자세한 내용은 트러블슈팅 3번("근거가 불명확했던 위험 판정 기준") 참고.
- **"위험 요인" 표현 정정**: 로지스틱 회귀 계수 기반 기여도를 "위험 요인"으로 표시하던 부분을 "예측에 영향을 준 주요 특성"으로 바꾸고, 인과관계처럼 읽히던 권고 문구(예: "미혼 → 사회적 지지 부족")를 통계적 연관성 수준으로 낮춰 다시 작성했습니다.
- **모델 선정 기준 통일**: 이미지 모델 비교·체크포인트 저장 기준이 Accuracy였는데, Optuna 튜닝은 Macro F1을 최적화하고 있어 기준이 서로 달랐습니다. 클래스 불균형이 있는 문제라 Macro F1로 통일했습니다.
- **CUDA 하드코딩 제거**: 학습 스크립트들이 `torch.amp.autocast("cuda")`를 조건 없이 사용해 GPU가 없는 환경에서 그대로 실행되지 않았습니다. `DEVICE.type`에 따라 AMP를 켜고 끄도록 수정했습니다.
- **하드코딩된 경로 정리**: `E:\skin\...`, `C:\Users\asia\...`, `D:\시각화세미2\찐찐찐최종\...` 같은 개발자 PC의 절대경로가 모든 학습 스크립트에 남아있던 것을 저장소 루트 기준 상대경로로 교체했습니다.
- **config 경로 실제 적용**: 이미지 업로드 페이지가 `config.py`의 `UPLOAD_DIR`을 무시하고 별도 경로에 직접 저장하고 있던 것을 `UPLOAD_DIR`을 실제로 사용하도록 수정했습니다.
- **인증 방식 교체**: 비밀번호를 salt 없는 SHA-256 해시로 저장하던 것을 bcrypt로 교체했습니다.
- **의존성 정리**: 앱에서 쓰지 않는 패키지(folium 등)를 제거하고, 전체 파이프라인용 `requirements.txt`와 앱 전용 `app/requirements.txt`를 분리했습니다.
- **통계 인용 수정**: 소개 문단의 "성인 유병률 2008년 대비 2.6배 증가"라는 문구가 팀이 실제로 인용했던 자료(질병관리청 국민건강영양조사, 2013년 3.4%→2022년 6.3%)와 맞지 않아 정정했습니다.

---

## 💡 프로젝트에서 배운 점

**성능 숫자 하나로 멈추지 않는 모델 진단**
합성 이미지에서 99%대였던 정확도가 실제 촬영 환경에서는 절반 수준으로 떨어지는 걸 보고, 전체 정확도 대신 클래스별 breakdown과 오분류 케이스를 직접 들여다봐서 원인(학습 데이터의 조명·배경 편향)을 구체적으로 짚어내는 진단 과정을 거쳤습니다.

**모델이 무엇을 답하는지 재정의하는 판단**
이미지 모델과 문진 모델의 확률을 단순히 더해서 하나의 점수로 만들려던 초기 설계를, 두 모델이 애초에 서로 다른 질문에 답한다는 걸 인지하고 분리 운영 구조로 다시 설계했습니다. 성능을 올리는 대신 무엇을 측정하고 있는지를 다시 정의한 경험이었습니다.

**결과물을 낸 뒤에도 다시 검증하는 태도**
포트폴리오 정리 과정에서 이미 배포된 저장소를 다시 훑어보다가 개인정보 노출, 파일 내용 뒤바뀜, 근거 없는 판정 기준 같은 실제 문제를 발견하고 고쳤습니다. "완성했다"와 "검증됐다"는 다르다는 걸 확인한 경험입니다.

---

## ⚠️ 한계 및 향후 개선 방향

- **문진 모델의 해석 범위**: 타깃이 아토피 진단 이력(과거)이라, 미래 발병 위험을 예측하는 모델이 아닙니다. 서비스 문구도 이에 맞춰 "위험도 예측"이 아닌 "관련도 분석"으로 표현하고 있습니다.
- **도메인 갭**: 학습 데이터의 상당 부분이 합성 이미지 — 실사 이미지 추가 수집 필요 (ISIC 등)
- **이미지 분할 방식**: DermNet은 원본 train/test를 통합한 뒤 이미지 단위로 재분할했습니다. 환자 단위 식별자가 데이터에 없어 환자 기준 분리는 하지 못했고, 동일 인물의 유사 이미지가 train/test에 함께 섞였을 가능성을 배제할 수 없습니다.
- **설명 가능성**: Grad-CAM 시각화 미적용 — 모델이 어떤 피부 부위를 근거로 판단하는지 검증 필요
- **오분류 클래스**: DermNet 테스트 기준 건선 ↔ 아토피 혼동이 가장 빈번(9건) — 클래스 경계 집중 학습 필요
- **문진 모델 성능**: PR-AUC 0.33 수준 — 불균형 데이터(아토피 약 10%) 특성상 한계 존재, 다년도 데이터 누적 시 개선 가능

---

## ⚠️ 사용 안내

본 프로젝트는 학습/연구 목적으로 제작되었으며, 별도의 오픈소스 라이선스는 지정하지 않았습니다. 사용한 데이터셋(AI Hub, DermNet, KNHANES)은 각 제공처의 이용 조건을 따릅니다. AtoCatch는 의료 진단을 대체하지 않습니다. 정확한 진단은 반드시 피부과 전문의와 상담하세요.
