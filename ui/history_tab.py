# ui/history_tab.py
"""
처리 이력 탭

PDF→Markdown과 Markdown→임베딩 처리 이력을 표시하는 탭
"""

import streamlit as st
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def render_history_tab():
    """처리 이력 탭 렌더링"""
    st.header("처리 이력")
    
    # 세션 상태 초기화
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    
    if st.session_state.processed_files:
        # 처리 이력을 데이터프레임으로 변환
        df = pd.DataFrame(st.session_state.processed_files)
        
        # 이력 표시
        st.dataframe(df, use_container_width=True)
        
        # CSV 다운로드 버튼
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "CSV로 다운로드",
            data=csv,
            file_name="document_processing_history.csv",
            mime="text/csv"
        )
        
        # 이력 삭제 버튼
        if st.button("이력 삭제", type="secondary"):
            st.session_state.processed_files = []
            st.success("처리 이력이 삭제되었습니다.")
            st.rerun()
    else:
        st.info("아직 처리된 파일이 없습니다.")
