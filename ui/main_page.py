# ui/main_page.py
"""
앱 메인 페이지 설정

Streamlit 페이지 설정 및 기본 레이아웃
"""

import streamlit as st

def setup_page_config():
    """앱 페이지 설정"""
    st.set_page_config(
        page_title="문서 처리 파이프라인",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )