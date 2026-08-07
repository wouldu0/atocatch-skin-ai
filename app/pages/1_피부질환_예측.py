import streamlit as st
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image
from utils.predict import predict, CLASS_NAMES
from utils.db import save_record
from utils.style import apply_global_style
from config import UPLOAD_DIR
import datetime

st.set_page_config(page_title="피부 분석", page_icon="📷", layout="centered")
apply_global_style()

# ── 로그인 확인 ─────────────────────────────────────────────
if "user" not in st.session_state or st.session_state.user is None:
    st.warning("로그인이 필요합니다.")
    st.page_link("main.py", label="로그인 페이지로 이동")
    st.stop()

user = st.session_state.user

st.title("📷 피부 분석")
st.caption("피부 사진을 업로드하면 5가지 피부 질환 예측 결과를 확인할 수 있습니다.")
st.divider()

# ── 이미지 업로드 ────────────────────────────────────────────
uploaded = st.file_uploader(
    "피부 사진을 업로드하세요 (JPG, PNG)",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.image(image, caption="업로드된 이미지", use_container_width=True)

    with col2:
        with st.spinner("AI 분석 중..."):
            results = predict(image)

        top = results[0]
        st.subheader(f"예측 결과: **{top['label']}**")
        st.metric("예측 확률", f"{top['prob']*100:.1f}%")

        st.markdown("#### 전체 질환별 확률")
        for r in results:
            color = "#e74c3c" if r["label"] == top["label"] else "#3498db"
            st.markdown(
                f"""
                <div style='display:flex; align-items:center; margin-bottom:6px;'>
                    <span style='width:80px; font-size:14px;'>{r['label']}</span>
                    <div style='flex:1; background:#eee; border-radius:4px; height:18px; margin:0 8px;'>
                        <div style='width:{r["prob"]*100:.1f}%; background:{color};
                                    border-radius:4px; height:18px;'></div>
                    </div>
                    <span style='font-size:13px; width:45px; text-align:right;'>
                        {r["prob"]*100:.1f}%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # ── 이미지 저장 ──────────────────────────────────────────
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    img_filename = f"{user['id']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    img_path = os.path.join(UPLOAD_DIR, img_filename)
    image.save(img_path, "JPEG", quality=92)

    # ── DB 저장 ──────────────────────────────────────────────
    record_id = save_record(
        user_id=user["id"],
        image_path=img_path,
        predictions=results,
        top_label=top["label"],
        top_prob=top["prob"],
    )
    st.session_state["last_record_id"]   = record_id
    st.session_state["last_predictions"] = results
    st.session_state["last_top_label"]   = top["label"]
    st.session_state["last_top_prob"]    = top["prob"]
    st.session_state["last_image_path"]  = img_path

    # 새 이미지 → 이전 설문 결과 초기화 (보고서에 섞이지 않도록)
    for _key in ["survey_score", "atopy_prob", "risk_level", "risk_desc", "survey_features"]:
        st.session_state.pop(_key, None)

    # ── 아토피 분기 ──────────────────────────────────────────
    if top["label"] == "아토피":
        st.success("아토피 가능성이 감지됐습니다. 생활습관 설문을 통해 예측에 영향을 준 특성을 확인해보세요.")
        st.page_link("pages/2_아토피_설문.py", label="📋 아토피 설문 바로가기",
                     width='stretch')
    else:
        st.info(f"**{top['label']}** 가능성이 높습니다. 정확한 진단은 전문의와 상담하세요.")
        st.page_link("pages/3_위험도_보고서.py", label="📊 보고서 확인",
                     width='stretch')

    st.caption("⚠️ 이 결과는 의료 진단이 아닙니다. 반드시 전문의와 상담하세요.")
    