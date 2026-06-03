import streamlit as st
import hashlib
import sys, os
sys.path.append(os.path.dirname(__file__))

from utils.db import init_db, get_user, create_user
from utils.style import apply_global_style

# "C:/Users/asia/anaconda3/envs/skin/Scripts/streamlit.exe" run E:/skin/app/main.py
# C:/Users/asia/anaconda3/envs/skin/python.exe -m streamlit run E:/skin/app/main.py

# ── 초기화 ──────────────────────────────────────────────────
init_db()

st.set_page_config(
    page_title="AtoCatch",
    page_icon="E:/skin/app/assets/logo.png",
    layout="centered",
)
apply_global_style()

def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

# ── 세션 초기화 ─────────────────────────────────────────────
if "user" not in st.session_state:
    st.session_state.user = None

# ── 로그인/회원가입 ─────────────────────────────────────────
LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets/logo.png")

if st.session_state.user is None:
    col_logo, col_title = st.columns([1, 2.5])
    with col_logo:
        st.image(LOGO_PATH, width=150)
    with col_title:
        st.title("AtoCatch")
        st.caption("아토피 조기 예측 & 케어 솔루션")
    st.divider()

    tab_login, tab_signup = st.tabs(["로그인", "회원가입"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("아이디")
            password = st.text_input("비밀번호", type="password")
            submitted = st.form_submit_button("로그인", width='stretch')
        if submitted:
            user = get_user(username)
            if user and user["password"] == hash_pw(password):
                st.session_state.user = user
                st.success(f"{username}님, 환영합니다!")
                st.rerun()
            else:
                st.error("아이디 또는 비밀번호가 올바르지 않습니다.")

    with tab_signup:
        with st.form("signup_form"):
            new_user = st.text_input("사용할 아이디")
            new_pw   = st.text_input("비밀번호", type="password")
            new_pw2  = st.text_input("비밀번호 확인", type="password")
            submitted = st.form_submit_button("회원가입", width='stretch')
        if submitted:
            if new_pw != new_pw2:
                st.error("비밀번호가 일치하지 않습니다.")
            elif len(new_pw) < 4:
                st.error("비밀번호는 4자 이상이어야 합니다.")
            elif create_user(new_user, hash_pw(new_pw)):
                st.success("가입 완료! 로그인해주세요.")
            else:
                st.error("이미 사용 중인 아이디입니다.")

# ── 로그인 후 홈 ────────────────────────────────────────────
else:
    user = st.session_state.user
    col_logo, col_title = st.columns([1, 2.5])
    with col_logo:
        st.image(LOGO_PATH, width=150)
    with col_title:
        st.title("AtoCatch")
        st.caption("아토피 조기 예측 & 케어 솔루션")
    st.divider()

    st.success(f"👤 {user['username']}님으로 로그인 중")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.page_link("pages/1_피부질환_예측.py", label="📷 피부 분석", width='stretch')
    with col2:
        st.page_link("pages/2_아토피_설문.py", label="📋 아토피 설문", width='stretch')
    with col3:
        st.page_link("pages/3_위험도_보고서.py", label="📊 위험도 보고서", width='stretch')
    with col4:
        st.page_link("pages/4_내_기록.py", label="🗂️ 내 기록", width='stretch')

    st.divider()
    st.markdown("""
    ### 사용 방법
    1. **📷 피부 분석** — 피부 사진을 업로드하면 5가지 피부 질환 예측 결과를 확인할 수 있습니다.
    2. **📋 아토피 설문** — 아토피가 1위로 예측된 경우 생활습관·병력 설문을 진행합니다.
    3. **📊 위험도 보고서** — 예측 결과와 아토피 위험 요인 및 맞춤 권고사항을 확인합니다.
    4. **🗂️ 내 기록** — 과거 분석 기록과 아토피 확률 변화 추이를 확인합니다.

    > ⚠️ AtoCatch는 의료 진단을 대체하지 않습니다. 정확한 진단은 반드시 전문의와 상담하세요.
    """)

    if st.button("로그아웃", width='content'):
        st.session_state.user = None
        st.session_state.pop("last_record_id", None)
        st.session_state.pop("last_predictions", None)
        st.rerun()
