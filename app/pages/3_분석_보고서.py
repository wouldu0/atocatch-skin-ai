import streamlit as st
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.atopy_model import get_risk_factors, get_threshold
from utils.style import apply_global_style

st.set_page_config(page_title="분석 보고서", page_icon="📊", layout="centered")
apply_global_style()

if "user" not in st.session_state or st.session_state.user is None:
    st.warning("로그인이 필요합니다.")
    st.page_link("main.py", label="로그인 페이지로 이동")
    st.stop()

if "last_predictions" not in st.session_state:
    st.warning("먼저 피부 분석을 진행해주세요.")
    st.page_link("pages/1_피부질환_예측.py", label="📷 피부 분석으로 이동")
    st.stop()

predictions  = st.session_state["last_predictions"]
top_label    = st.session_state["last_top_label"]
top_prob     = st.session_state["last_top_prob"]
atopy_prob   = st.session_state.get("atopy_prob")
risk_level   = st.session_state.get("risk_level")
risk_desc    = st.session_state.get("risk_desc")
image_path   = st.session_state.get("last_image_path")

st.title("📊 종합 분석 보고서")
st.divider()

# ── 1. AI 이미지 분석 결과 ──────────────────────────────────────
st.subheader("1. AI 피부 분석 결과")

if image_path and os.path.exists(image_path):
    col_img, col_result = st.columns([1, 2])
    with col_img:
        from PIL import Image as PILImage
        st.image(PILImage.open(image_path), use_container_width=True)
    with col_result:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("예측 질환", top_label)
        with col2:
            st.metric("예측 확률", f"{top_prob*100:.1f}%")
else:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("예측 질환", top_label)
    with col2:
        st.metric("예측 확률", f"{top_prob*100:.1f}%")

st.markdown("**질환별 예측 확률**")
for r in predictions:
    color = "#e74c3c" if r["label"] == top_label else "#3498db"
    st.markdown(
        f"""
        <div style='display:flex; align-items:center; margin-bottom:5px;'>
            <span style='width:80px; font-size:13px;'>{r['label']}</span>
            <div style='flex:1; background:#eee; border-radius:4px; height:16px; margin:0 8px;'>
                <div style='width:{r["prob"]*100:.1f}%; background:{color};
                            border-radius:4px; height:16px;'></div>
            </div>
            <span style='font-size:12px; width:45px; text-align:right;'>{r["prob"]*100:.1f}%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── 2. 종합 소견 ─────────────────────────────────────────────────
st.divider()
st.subheader("2. 종합 소견")

if top_label == "정상":
    st.success("현재 피부 상태가 정상으로 예측됩니다. 정기적인 피부 관리를 유지하세요.")
elif top_label == "아토피":
    st.warning("아토피 가능성이 있습니다. 아래 예측에 영향을 준 특성을 확인하고 피부과 전문의 상담을 권장합니다.")
elif top_label == "건선":
    st.warning("건선 가능성이 있습니다. 건선은 면역 질환으로, 피부과 전문의 진단이 필요합니다.")
elif top_label == "여드름":
    st.warning("여드름 가능성이 높습니다. 올바른 세안과 보습 관리, 필요 시 피부과 상담을 권장합니다.")
elif top_label == "주사":
    st.warning("주사(Rosacea) 가능성이 있습니다. 자극 요인을 피하고 피부과 상담을 권장합니다.")
else:
    st.warning(f"**{top_label}** 가능성이 있습니다. 정확한 진단을 위해 피부과 방문을 권장합니다.")

# ── 3. 아토피 가능성 스크리닝 결과 (설문 모델) ─────────────────────
st.divider()
st.subheader("3. 아토피 가능성 스크리닝 결과")
survey_features = st.session_state.get("survey_features")
if survey_features and atopy_prob is not None:
    threshold = get_threshold()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("스크리닝 확률 (calibrated)", f"{atopy_prob*100:.1f}%")
    with col2:
        st.metric("운영 기준치", f"{threshold*100:.1f}%")
    if risk_level == "스크리닝 기준치 이상":
        st.warning(f"**{risk_level}** — {risk_desc}")
    else:
        st.success(f"**{risk_level}** — {risk_desc}")
    st.caption(
        "⚠️ 이 확률은 미래 아토피 발병 확률이 아닙니다. KNHANES 설문 응답 패턴이 "
        "과거 아토피 의사진단 이력이 있는 집단과 얼마나 유사한지를 보여주는 "
        "스크리닝 지표이며, 위 1번의 이미지 모델 확률과는 서로 다른 정보를 측정하므로 "
        "합산하지 않고 독립적으로 해석합니다."
    )
else:
    st.info("아토피 설문을 진행하지 않았습니다. 아토피가 1위로 예측된 경우 설문을 통해 스크리닝 결과를 확인할 수 있습니다.")
    st.page_link("pages/2_아토피_설문.py", label="📋 아토피 설문 바로가기")

# ── 4. 예측에 영향을 준 주요 특성 ─────────────────────────────────
st.divider()
st.subheader("4. 예측에 영향을 준 주요 특성")
st.caption("모델이 인과관계를 판단한 것이 아니라, 입력값이 예측 점수에 기여한 정도를 보여줍니다.")
if survey_features:
    factors = get_risk_factors(survey_features)
    st.markdown("")
    if factors:
        st.markdown("**📌 예측 점수에 영향을 준 주요 특성 및 참고 정보**")
        for i, f in enumerate(factors[:3], 1):
            st.markdown(
                f"""
                <div style='background:#f8f9fa; border-left:4px solid #e67e22;
                            border-radius:4px; padding:10px 14px; margin-bottom:8px;'>
                    <div style='font-weight:600; margin-bottom:4px;'>
                        {i}. {f['label']}
                    </div>
                    <div style='font-size:13px; color:#555;'>
                        💡 {f['recommendation']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.caption("✅ 설문 응답에서 예측 점수에 영향을 준 주요 특성이 발견되지 않았습니다.")
else:
    st.info("아토피 설문을 진행하지 않았습니다.")

# ── 5. 주변 피부과 찾기 ──────────────────────────────────────────
st.divider()
st.subheader("5. 주변 피부과 찾기")
st.caption("아래 버튼을 누르면 현재 위치 기준으로 가까운 피부과를 검색할 수 있습니다.")

col1, col2 = st.columns(2)
with col1:
    st.link_button(
        "🗺️ 네이버 지도에서 검색",
        "https://map.naver.com/v5/search/피부과",
        use_container_width=True,
    )
with col2:
    st.link_button(
        "🗺️ 카카오 지도에서 검색",
        "https://map.kakao.com/?q=피부과",
        use_container_width=True,
    )

st.caption("※ 버튼을 누르면 새 탭에서 현재 위치 기준 피부과 검색 결과가 열립니다.")

st.divider()
st.caption("⚠️ 이 보고서는 AI 기반 참고 자료이며 의료 진단을 대체하지 않습니다.")

# ── 6. 보고서 저장 ────────────────────────────────────────────────
st.divider()
st.subheader("6. 보고서 저장")

import datetime

color_map_save = {"스크리닝 기준치 이상": "#e74c3c", "스크리닝 기준치 미만": "#2ecc71"}
risk_color     = color_map_save.get(risk_level, "#888888")
now_str        = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

# 이미지 base64 변환 (HTML에 embed)
import base64
img_tag = ""
if image_path and os.path.exists(image_path):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    img_tag = f"""
    <div style='margin-bottom:16px;'>
        <img src='data:image/jpeg;base64,{img_b64}'
             style='max-width:220px; border-radius:8px; border:1px solid #ddd;'/>
    </div>"""

# 질환별 확률 바 HTML (PDF 호환 테이블 방식)
prob_bars = "<table style='width:100%; border-collapse:collapse; margin-bottom:12px;'>"
for r in predictions:
    color  = "#e74c3c" if r["label"] == top_label else "#3498db"
    pct    = r["prob"] * 100
    filled = int(pct)          # 색칠 칸 수 (최대 100)
    empty  = 100 - filled
    prob_bars += f"""
    <tr style='margin-bottom:6px;'>
        <td style='width:70px; font-size:14px; padding:3px 6px 3px 0;'>{r['label']}</td>
        <td style='padding:3px 8px;'>
            <table style='width:100%; border-collapse:collapse; height:16px;'>
                <tr>
                    <td style='width:{pct:.1f}%; background:{color};
                               -webkit-print-color-adjust:exact; print-color-adjust:exact;
                               height:16px;'></td>
                    <td style='width:{100-pct:.1f}%; background:#eeeeee;
                               -webkit-print-color-adjust:exact; print-color-adjust:exact;
                               height:16px;'></td>
                </tr>
            </table>
        </td>
        <td style='width:45px; font-size:13px; text-align:right; padding:3px 0;'>{pct:.1f}%</td>
    </tr>"""
prob_bars += "</table>"

# 아토피 가능성 스크리닝 결과 섹션 (설문 모델)
screening_section = ""
if survey_features and atopy_prob is not None:
    _threshold = get_threshold()
    screening_section = f"""
    <h2>2. 아토피 가능성 스크리닝 결과</h2>
    <div class='highlight'>
      <p><strong>스크리닝 확률(calibrated):</strong> {atopy_prob*100:.1f}% &nbsp;|&nbsp;
         <strong>운영 기준치:</strong> {_threshold*100:.1f}% &nbsp;|&nbsp;
         <strong>판정:</strong> <span style='color:{risk_color}; font-weight:600;'>{risk_level}</span></p>
      <p>{risk_desc}</p>
    </div>
    <p style='font-size:12px; color:#888;'>
      ⚠️ 이 확률은 미래 아토피 발병 확률이 아닙니다. KNHANES 설문 응답 패턴이 과거 아토피
      의사진단 이력이 있는 집단과 얼마나 유사한지를 보여주는 스크리닝 지표이며,
      위 이미지 모델 확률과는 서로 다른 정보를 측정하므로 합산하지 않고 독립적으로 해석합니다.
    </p>"""

# 예측에 영향을 준 주요 특성 섹션
factors_section = ""
if survey_features:
    _factors = get_risk_factors(survey_features)
    if _factors:
        _factors_html = ""
        for i, f in enumerate(_factors[:3], 1):
            _factors_html += f"""
            <div style='background:#f8f9fa; border-left:4px solid #e67e22;
                        border-radius:4px; padding:10px 14px; margin-bottom:8px;'>
                <div style='font-weight:600; margin-bottom:4px;'>{i}. {f['label']}</div>
                <div style='font-size:13px; color:#555;'>💡 {f['recommendation']}</div>
            </div>"""
        factors_section = f"""
        <h2>3. 예측에 영향을 준 주요 특성 및 참고 정보</h2>
        {_factors_html}"""
    else:
        factors_section = """
        <h2>3. 예측에 영향을 준 주요 특성</h2>
        <p style='color:#27ae60;'>✅ 설문 응답에서 예측 점수에 영향을 준 주요 특성이 발견되지 않았습니다.</p>"""

html_report = f"""<!DOCTYPE html>
<html lang='ko'>
<head>
<meta charset='UTF-8'>
<title>피부 질환 AI 스크리닝 보고서</title>
<style>
  * {{ -webkit-print-color-adjust: exact !important;
       print-color-adjust: exact !important; }}
  body {{ font-family: 'Malgun Gothic', Arial, sans-serif; max-width: 800px;
          margin: 40px auto; padding: 20px; color: #333; }}
  h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px; }}
  h2 {{ color: #2c3e50; margin-top: 30px; }}
  .meta {{ color: #888; font-size: 14px; margin-bottom: 20px; }}
  .highlight {{ background: #f0f8ff; padding: 12px; border-radius: 6px; }}
  .warning {{ background: #fff3e0; border-left: 4px solid #f39c12;
              padding: 12px; border-radius: 4px; margin-top: 20px; }}
  @media print {{
    * {{ -webkit-print-color-adjust: exact !important;
         print-color-adjust: exact !important; }}
  }}
</style>
</head>
<body>
<h1>📊 피부 질환 AI 스크리닝 보고서</h1>
<p class='meta'>생성일시: {now_str} &nbsp;|&nbsp; ⚠️ 본 보고서는 AI 기반 참고 자료이며 의료 진단을 대체하지 않습니다.</p>

<h2>1. AI 피부 분석 결과</h2>
{img_tag}
<div class='highlight'>
  <p><strong>예측 질환:</strong> {top_label} &nbsp;|&nbsp;
     <strong>예측 확률:</strong> {top_prob*100:.1f}%</p>
</div>
<p><strong>질환별 예측 확률</strong></p>
{prob_bars}

{screening_section}

{factors_section}

<div class='warning'>
  ⚠️ 이 보고서는 AI 기반 참고 자료이며 의료 진단을 대체하지 않습니다.
  정확한 진단은 반드시 피부과 전문의와 상담하세요.
</div>
</body>
</html>"""

st.download_button(
    label="💾 보고서 HTML로 저장 (브라우저에서 열어 PDF 인쇄 가능)",
    data=html_report.encode("utf-8"),
    file_name=f"피부보고서_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.html",
    mime="text/html",
    use_container_width=True,
)
