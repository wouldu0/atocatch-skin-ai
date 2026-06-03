"""
ML 모델 (LogisticRegression + class_weight) 래퍼
- 구조: sklearn Pipeline (SimpleImputer → LogisticRegression)
- 피처: 7개 (이진 범주형)
- 출력: 아토피 위험 확률 (Isotonic Calibration 적용, 0~1)
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

# 위험도 기준
THRESH_HIGH   = 0.25   # 25% 이상 → 높음
THRESH_MEDIUM = 0.10   # 10% 이상 → 보통

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

# 피처별 권고 메시지
FEATURE_RECOMMENDATIONS = {
    "DJ8_dg_1.0":      "알레르기 비염은 아토피와 같은 면역 경로를 공유합니다. 집먼지진드기·꽃가루 등 알레르겐 노출을 줄이고 비염 치료를 꾸준히 유지하세요.",
    "DJ4_dg_1.0":      "천식·비염·아토피는 함께 나타나는 경우가 많습니다. 알레르기 전문의와 함께 통합적으로 관리하는 것을 권장합니다.",
    "marri_1_2.0":     "사회적 지지가 부족할수록 스트레스 대처 자원이 줄어들 수 있습니다. 가족·지인과의 교류를 늘리거나 필요 시 심리 상담을 활용해보세요.",
    "age_group_15-34": "15~34세 연령대는 아토피 피부염 유병률이 상대적으로 높습니다. 정기적인 피부 상태 확인과 보습 관리를 꾸준히 실천하세요.",
    "BD1_11_6.0":      "잦은 음주는 피부 장벽 기능을 약화시키고 아토피 염증 반응을 악화시킬 수 있습니다. 음주 횟수를 줄이고 충분한 수분 섭취와 보습을 병행하세요.",
    "sm_presnt_1.0":   "흡연은 피부 장벽을 손상시키고 아토피 염증 반응을 촉진합니다. 금연은 피부 증상 개선에 가장 직접적인 효과를 줄 수 있습니다.",
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
    if prob >= THRESH_HIGH:
        return "높음", "아토피 피부염 위험이 높습니다. 피부과 전문의 상담을 강력히 권장합니다."
    elif prob >= THRESH_MEDIUM:
        return "보통", "아토피 위험 요인이 일부 있습니다. 증상 변화를 주의 깊게 관찰하세요."
    else:
        return "낮음", "현재 아토피 위험도는 낮습니다. 피부 청결과 보습을 꾸준히 유지하세요."


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
