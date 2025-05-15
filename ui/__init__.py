# ui/__init__.py
"""
UI 관련 모듈

이 패키지는 Streamlit UI 컴포넌트 및 페이지를 제공합니다.
"""

from .main_page import setup_page_config
from .sidebar import render_sidebar
from .pdf_to_md_tab import render_pdf_to_md_tab
from .md_upload_tab import render_md_upload_tab
from .history_tab import render_history_tab
from .setting_tab import render_settings_tab
from .help_tab import render_help_tab
from .process_tab import process_file  # 기존 코드와의 호환성을 위한 import

__all__ = [
    'setup_page_config',
    'render_sidebar',
    'process_file',  # 레거시 호환성 함수
    'render_pdf_to_md_tab',
    'render_md_upload_tab',
    'render_history_tab',
    'render_settings_tab',
    'render_help_tab',
]