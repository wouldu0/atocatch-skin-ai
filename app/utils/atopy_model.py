"""
ML 모델 (LogisticRegression + class_weight) 래퍼
- 구조: sklearn Pipeline (SimpleImputer → LogisticRegression)
- 피처: 7개 (이진 범주형)
- 타깃: 아토피피부염 의사진단 여부(DL1_dg, 과거 진단 이력) — 미래 발병 위험 예측이 아님
- 출력: 예측 확률 (Isotonic Calibration 적용, 0~1)
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st

from config import SURVEY_MODEL_PATH

MODEL_PKL = SURVEY_MODEL_PATH

# 운영 threshold (survey_model/05_threshold_analysis.py 산출)
# calibration holdout set에서 Youden's J(sensitivity+specificity 균형) 기준으로 선정,
# 최종 test set에는 이 값을 고정해 1회만 적용함 (test 재사용 없음).
THRESHOLD = 0.11

# 참고: 같은 calib set에서 F1-optimal 기준으로는 0.17이 나왔으나,
# 앱 운영 기준으로는 채택하지 않고 비교값으로만 기록해 둠.
F1_OPTIMAL_THRESHOLD_REFERENCE = 0.17

# 피처 한글 라벨
FEATURE_LABELS = {
    "DJ8_dg_1.0":      "알레르기 비염 진단력",
    "DJ4_dg_1.0":      "천식 진단력",
    "marri_1_2.0":     "미혼",
    "age_group_15-34": "15~34세 연령대",
    "town_t_2.0":      "농어촌 거주",
    "BD1_11_6.0":      "잦은 음주 (주 4회 이상)",
    "sm_presnt_1.0":   "현재 흡연",
}

# 피처별 참고 정보
# 주의: 아래 문구는 모델이 도출한 인과관계가 아니라 일반적으로 알려진 정보입니다.
# 모델은 이 피처들과 과거 아토피 진단 이력 사이의 통계적 연관성만 학습했습니다.
FEATURE_RECOMMENDATIONS = {
    "DJ8_dg_1.0":      "알레르기 비염은 아토피 피부염과 함께 나타나는 경우가 많은 알레르기 질환군으로 알려져 있습니다. 비염 증상 관리를 꾸준히 하는 것이 도움이 될 수 있습니다.",
    "DJ4_dg_1.0":      "천식·비염·아토피는 함께 나타나는 경우가 많은 알레르기 질환군입니다. 알레르기 전문의와 통합적으로 관리하는 것을 권장합니다.",
    "marri_1_2.0":     "이 데이터에서 통계적으로 연관성이 있었던 항목입니다. 혼인 상태 자체가 원인이라기보다 다른 요인과 함께 작용했을 가능성이 있어, 특정 행동을 권고하기보다 참고 정보로 안내드립니다.",
    "age_group_15-34": "이 데이터에서 15~34세 연령대는 아토피 진단 비율이 상대적으로 높게 나타났습니다. 정기적인 피부 상태 확인과 보습 관리를 권장합니다.",
    "town_t_2.0":      "거주 지역에 따른 환경적 요인(꽃가루·알레르겐 노출 등) 차이와 관련이 있을 수 있습니다.",
    "BD1_11_6.0":      "잦은 음주는 피부 장벽 기능 저하와 관련이 있다고 알려져 있습니다. 음주 빈도를 줄이고 충분한 수분 섭취와 보습을 병행하는 것이 도움이 될 수 있습니다.",
    "sm_presnt_1.0":   "흡연은 피부 장벽을 손상시킬 수 있다고 알려져 있습니다. 금연은 피부 건강 전반에 도움이 될 수 있습니다.",
}


@st.cache_resource
def _load_model():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(MODEL_PKL, "rb") as f:
            return pickle.load(f)


def _make_df(features: dict, feature_order: list) -> pd.DataFrame:
    return pd.DataFrame([{f: features[f] for f in feature_order}])


def predict_atopy_prob(features: dict) -> float:
    m = _load_model()
    df = _make_df(features, m["feature_order"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob = float(m["sklearn_pipeline"].predict_proba(df)[0][1])

    if m.get("isotonic") is not None:
        prob = float(m["isotonic"].predict([prob])[0])

    return prob


def get_risk_level(prob: float) -> tuple[str, str]:
    """prob과 운영 threshold(THRESHOLD)를 비교해 이진 스크리닝 결과를 반환.

    3단계(낮음/보통/높음) 대신 이진 판정만 쓰는 이유: 분석으로 얻은 운영점은
    THRESHOLD 하나뿐이라, 등급을 더 나누면 그 경계에 대한 별도 통계적 근거가
    없어집니다.
    """
    if prob >= THRESHOLD:
        return (
            "스크리닝 기준치 이상",
            "입력한 정보에서 아토피 진단군과 유사한 패턴이 확인되었습니다. 피부 증상이 있다면 전문의 상담을 고려해 주세요.",
        )
    else:
        return (
            "스크리닝 기준치 미만",
            "입력한 정보에서 아토피 진단군과 유사한 패턴이 뚜렷하게 확인되지 않았습니다. 피부 청결과 보습을 꾸준히 유지하세요.",
        )


def get_risk_factors(features: dict) -> list[dict]:
    m = _load_model()
    clf = m["sklearn_pipeline"].named_steps["clf"]
    coef = clf.coef_[0]

    factors = []
    for feat, c in zip(m["feature_order"], coef):
        val = features.get(feat, 0)
        contrib = c * val
        if contrib > 0.05:
            factors.append({
                "label":          FEATURE_LABELS.get(feat, feat),
                "contribution":   float(contrib),
                "recommendation": FEATURE_RECOMMENDATIONS.get(feat, ""),
            })

    return sorted(factors, key=lambda x: x["contribution"], reverse=True)
