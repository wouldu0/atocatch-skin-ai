import streamlit as st

def apply_global_style():
    st.markdown("""
    <style>
        /* 전체 본문 폰트 크기 */
        html, body, [class*="css"] {
            font-size: 17px !important;
        }
        /* 일반 텍스트 */
        p, li, label, div {
            font-size: 17px !important;
        }
        /* 입력창 텍스트 */
        input, textarea, select {
            font-size: 16px !important;
        }
        /* 라디오·체크박스 라벨 */
        .stRadio label, .stCheckbox label {
            font-size: 16px !important;
        }
        /* 캡션 */
        .stCaption, small {
            font-size: 14px !important;
        }
        /* metric 값 */
        [data-testid="stMetricValue"] {
            font-size: 28px !important;
        }
    </style>
    """, unsafe_allow_html=True)
