import streamlit as st
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.atopy_model import predict_atopy_prob, get_risk_level
from utils.db import update_record_survey
from utils.style import apply_global_style

st.set_page_config(page_title="아토피 설문", page_icon="📋", layout="centered")
apply_global_style()

# ── 로그인 / 피부 분석 선행 확인 ────────────────────────────────
if "user" not in st.session_state or st.session_state.user is None:
    st.warning("로그인이 필요합니다.")
    st.page_link("main.py", label="로그인 페이지로 이동")
    st.stop()

if "last_record_id" not in st.session_state:
    st.warning("먼저 피부 분석을 진행해주세요.")
    st.page_link("pages/1_피부질환_예측.py", label="📷 피부 분석으로 이동")
    st.stop()

st.title("📋 아토피 관련도 설문")
st.caption("아래 7개 문항에 응답하면 AI가 KNHANES 데이터 기반으로 아토피 진단 이력과의 관련도를 분석합니다.")
st.divider()

with st.form("survey_form"):

    # ── Q1. 나이 (슬라이더) ────────────────────────────────────
    st.markdown("**Q1. 현재 만 나이를 입력해 주세요.**")
    age_val = st.slider("나이 선택", min_value=0, max_value=80, value=25, step=1,
                        format="%d세", label_visibility="collapsed", key="q_age")
    st.caption("슬라이더를 움직여 나이를 선택하세요. (기준: 15~34세)")

    st.markdown("")
    # ── Q2. 알레르기 비염 진단 ────────────────────────────────
    st.markdown("**Q2. 의사로부터 알레르기 비염을 진단받은 적이 있나요?**")
    dj8_opt = st.radio("알레르기 비염 진단 여부", ["아니오", "예"], horizontal=True, label_visibility="collapsed", key="q_dj8")

    st.markdown("")
    # ── Q3. 천식 진단 ─────────────────────────────────────────
    st.markdown("**Q3. 의사로부터 천식을 진단받은 적이 있나요?**")
    dj4_opt = st.radio("천식 진단 여부", ["아니오", "예"], horizontal=True, label_visibility="collapsed", key="q_dj4")

    st.markdown("")
    # ── Q4. 결혼 상태 ─────────────────────────────────────────
    st.markdown("**Q4. 현재 결혼 여부를 선택해 주세요.**")
    marri_opt = st.radio("결혼 여부", ["기혼", "미혼"], horizontal=True, label_visibility="collapsed", key="q_marri")

    st.markdown("")
    # ── Q5. 현재 흡연 ─────────────────────────────────────────
    st.markdown("**Q5. 현재 담배(전자담배 포함)를 피우고 있나요?**")
    smoke_opt = st.radio("현재 흡연 여부", ["아니오", "예"], horizontal=True, label_visibility="collapsed", key="q_smoke")

    st.markdown("")
    # ── Q6. 거주 지역 ─────────────────────────────────────────
    st.markdown("**Q6. 현재 거주 지역은 어디인가요?**")
    town_opt = st.radio("거주 지역", ["도시(동 지역)", "농어촌(읍·면 지역)"],
                        horizontal=True, label_visibility="collapsed", key="q_town")

    st.markdown("")
    # ── Q7. 음주 빈도 ─────────────────────────────────────────
    st.markdown("**Q7. 최근 1년간 음주 빈도는 어떻게 되나요?**")
    drink_opt = st.radio(
        "음주 빈도",
        ["마시지 않는다", "월 1회 미만", "월 1~3회", "주 1~2회", "주 3~4회", "주 4회 이상"],
        horizontal=False,
        label_visibility="collapsed",
        key="q_drink",
    )

    st.markdown("")
    submitted = st.form_submit_button("결과 확인", width='stretch')

# ── 결과 계산 ───────────────────────────────────────────────────
if submitted:
    features = {
        "DJ8_dg_1.0":      1 if dj8_opt == "예" else 0,
        "DJ4_dg_1.0":      1 if dj4_opt == "예" else 0,
        "marri_1_2.0":     1 if marri_opt == "미혼" else 0,
        "age_group_15-34": 1 if 15 <= age_val <= 34 else 0,
        "town_t_2.0":      1 if "농어촌" in town_opt else 0,
        "BD1_11_6.0":      1 if drink_opt == "주 4회 이상" else 0,
        "sm_presnt_1.0":   1 if smoke_opt == "예" else 0,
    }

    atopy_prob = predict_atopy_prob(features)
    risk_level, risk_desc = get_risk_level(atopy_prob)

    # DB 업데이트 (survey_score = 확률 퍼센트 정수)
    survey_score_int = int(round(atopy_prob * 100))
    update_record_survey(
        record_id=st.session_state["last_record_id"],
        survey_score=survey_score_int,
        risk_level=risk_level,
        survey_features=features,
    )

    # 세션에 저장 (보고서 페이지에서 사용)
    st.session_state["survey_score"]    = survey_score_int
    st.session_state["atopy_prob"]      = atopy_prob
    st.session_state["risk_level"]      = risk_level
    st.session_state["risk_desc"]       = risk_desc
    st.session_state["survey_features"] = features   # 예측 기여 특성 분석용

    # ── 결과 출력 ────────────────────────────────────────────
    st.divider()
    st.success("설문이 완료됐습니다! 보고서에서 예측에 영향을 준 특성을 확인하세요.")
    st.markdown("")
    st.page_link("pages/3_위험도_보고서.py", label="📊 최종 보고서 확인",
                 width='stretch')
