# app.py
"""
문서 처리 파이프라인 메인 앱

PDF 문서를 마크다운으로 변환하고, 청킹하여 임베딩한 후 Qdrant에 저장하는 파이프라인
"""

import streamlit as st
import os
import logging

from ui import (
    setup_page_config,
    render_sidebar,
    render_pdf_to_md_tab,
    render_md_upload_tab,
    render_history_tab,
    render_settings_tab,
    render_help_tab
)

from utils import load_env_config, setup_logging

# 로깅 설정
setup_logging()
logger = logging.getLogger(__name__)

# 앱 초기화
def main():
    """메인 앱 실행"""
    logger.info("앱 시작")

    # 페이지 설정
    setup_page_config()

    # 세션 상태 초기화
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []

    # 중앙 집중식 설정 관리 - 초기 설정 로드 (세션에 없는 경우)
    if "config" not in st.session_state:
        initial_config = load_env_config()
        st.session_state.config = initial_config
        config = initial_config
    else:
        # 세션에서 설정 불러오기
        config = st.session_state.config

    # 필요한 디렉토리 생성
    os.makedirs(config["markdown_dir"], exist_ok=True)
    os.makedirs(config["embedding_model_dir"], exist_ok=True)
    os.makedirs(config["reranker_model_dir"], exist_ok=True)

    # 사이드바 렌더링 - 설정 업데이트 받기
    sidebar_config = render_sidebar(config)

    # 설정 업데이트 반영
    if sidebar_config:
        config.update(sidebar_config)
        st.session_state.config = config

    # 메인 화면 렌더링
    st.title("📚 문서 처리 파이프라인")
    st.caption("PDF 문서를 마크다운으로 변환 후 임베딩하여 벡터 DB에 저장합니다")

    # 탭 렌더링
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 PDF → Markdown",
        "📤 Markdown → 임베딩",
        "📋 처리 이력",
        "⚙️ 설정",
        "❓ 도움말"
    ])

    with tab1:
        render_pdf_to_md_tab(config)

    with tab2:
        render_md_upload_tab(config)

    with tab3:
        render_history_tab()

    with tab4:
        # 설정 업데이트 받기
        updated_config = render_settings_tab(config)

        # 설정 업데이트 반영
        if updated_config:
            config.update(updated_config)
            st.session_state.config = config

    with tab5:
        render_help_tab()

    st.markdown("---")
    st.caption("문서 처리 파이프라인 WebUI - v1.2")

if __name__ == "__main__":
    main()