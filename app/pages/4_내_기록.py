import streamlit as st
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
from PIL import Image
from utils.db import get_user_records, delete_record
from utils.atopy_model import get_risk_level
from utils.style import apply_global_style

st.set_page_config(page_title="내 기록", page_icon="🗂️", layout="wide")
apply_global_style()

if "user" not in st.session_state or st.session_state.user is None:
    st.warning("로그인이 필요합니다.")
    st.page_link("main.py", label="로그인 페이지로 이동")
    st.stop()

user    = st.session_state.user
records = get_user_records(user["id"])

st.title("🗂️ 내 기록")
st.caption(f"총 {len(records)}건의 분석 기록")
st.divider()

if not records:
    st.info("아직 분석 기록이 없습니다.")
    st.page_link("pages/1_피부질환_예측.py", label="📷 피부 분석 시작하기")
    st.stop()

# ── 아토피 확률 추이 그래프 ──────────────────────────────────
atopy_records = [
    {"날짜": r["created"][:10],
     "아토피 확률": next((p["prob"]*100 for p in r["predictions"] if p["label"]=="아토피"), 0)}
    for r in reversed(records)   # 오래된 순 → 최신 순 (x축 시간 흐름)
]
atopy_df = pd.DataFrame(atopy_records)

if len(atopy_df) > 1:
    st.subheader("📈 아토피 예측 확률 추이")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(atopy_df["날짜"], atopy_df["아토피 확률"],
            marker="o", color="#e74c3c", linewidth=2)
    ax.axhline(y=30, color="gray", linestyle="--", alpha=0.5, label="주의선 (30%)")
    ax.set_ylabel("아토피 확률 (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    plt.xticks(rotation=30, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    st.divider()

# ── 기록 목록 ────────────────────────────────────────────────
st.subheader("📋 분석 기록 목록")

for r in records:
    with st.expander(
        f"{r['created'][:16]}  |  **{r['top_label']}** {r['top_prob']*100:.1f}%",
        expanded=False,
    ):
        col1, col2 = st.columns([1, 2])

        with col1:
            if r["image_path"] and os.path.exists(r["image_path"]):
                st.image(Image.open(r["image_path"]), use_container_width=True)
            else:
                st.caption("이미지 없음")

        with col2:
            st.markdown("**질환별 예측 확률**")
            for p in sorted(r["predictions"], key=lambda x: x["prob"], reverse=True):
                bar_color = "#e74c3c" if p["label"] == r["top_label"] else "#3498db"
                st.markdown(
                    f"""
                    <div style='display:flex; align-items:center; margin-bottom:4px;'>
                        <span style='width:70px; font-size:12px;'>{p['label']}</span>
                        <div style='flex:1; background:#eee; border-radius:4px; height:14px; margin:0 6px;'>
                            <div style='width:{p["prob"]*100:.1f}%; background:{bar_color};
                                        border-radius:4px; height:14px;'></div>
                        </div>
                        <span style='font-size:11px; width:40px;'>{p["prob"]*100:.1f}%</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


        # ── 위험 요인 ────────────────────────────────────────
        if r.get("survey_features"):
            from utils.atopy_model import get_risk_factors
            factors = get_risk_factors(r["survey_features"])
            if factors:
                st.markdown("**📌 아토피 위험 요인 TOP 3**")
                for i, f in enumerate(factors[:3], 1):
                    st.markdown(
                        f"""
                        <div style='background:#f8f9fa; border-left:4px solid #e67e22;
                                    border-radius:4px; padding:8px 12px; margin-bottom:6px;'>
                            <div style='font-weight:600; font-size:13px;'>{i}. {f['label']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # ── 버튼 영역 ────────────────────────────────────────
        st.markdown("")

        # 설문 진행 가능 여부: 아토피 1위 & 설문 미완료
        can_survey = (r["top_label"] == "아토피" and r["survey_score"] is None)
        btn_col1, btn_col2, btn_col3, _ = st.columns([1.2, 1.4, 1.2, 2.2])

        # 보고서 보기 버튼
        with btn_col1:
            if st.button("📊 보고서 보기", key=f"report_{r['id']}", use_container_width=True):
                st.session_state["last_record_id"]   = r["id"]
                st.session_state["last_predictions"] = r["predictions"]
                st.session_state["last_top_label"]   = r["top_label"]
                st.session_state["last_top_prob"]    = r["top_prob"]
                st.session_state["last_image_path"]  = r.get("image_path")
                if r["survey_score"] is not None:
                    atopy_prob = r["survey_score"] / 100
                    st.session_state["atopy_prob"]      = atopy_prob
                    st.session_state["risk_level"]      = r["risk_level"]
                    _, risk_desc = get_risk_level(atopy_prob)
                    st.session_state["risk_desc"]       = risk_desc
                    st.session_state["survey_features"] = r.get("survey_features")
                else:
                    for k in ["atopy_prob", "risk_level", "risk_desc", "survey_features"]:
                        st.session_state.pop(k, None)
                st.switch_page("pages/3_위험도_보고서.py")

        # 설문 진행 버튼 (아토피 1위 & 설문 미완료일 때만)
        with btn_col2:
            if can_survey:
                if st.button("📋 설문 진행하기", key=f"survey_{r['id']}", use_container_width=True):
                    st.session_state["last_record_id"]   = r["id"]
                    st.session_state["last_predictions"] = r["predictions"]
                    st.session_state["last_top_label"]   = r["top_label"]
                    st.session_state["last_top_prob"]    = r["top_prob"]
                    st.session_state["last_image_path"]  = r.get("image_path")
                    for k in ["atopy_prob", "risk_level", "risk_desc", "survey_features"]:
                        st.session_state.pop(k, None)
                    st.switch_page("pages/2_아토피_설문.py")

        # 삭제 버튼
        with btn_col3:
            confirm_key = f"confirm_{r['id']}"
            if st.session_state.get(confirm_key):
                st.warning("정말 삭제할까요?")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅ 확인", key=f"del_ok_{r['id']}"):
                        delete_record(r["id"], user["id"])
                        st.session_state.pop(confirm_key, None)
                        st.rerun()
                with c2:
                    if st.button("취소", key=f"del_cancel_{r['id']}"):
                        st.session_state.pop(confirm_key, None)
                        st.rerun()
            else:
                if st.button("🗑️ 기록 삭제", key=f"del_{r['id']}", use_container_width=True):
                    st.session_state[confirm_key] = True
                    st.rerun()

