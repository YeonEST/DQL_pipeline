# utils/logging.py
"""
로깅 설정 유틸리티

애플리케이션 로깅을 초기화하고 설정하는 함수를 제공합니다.
"""

import logging
import os
from datetime import datetime

def setup_logging(log_to_file: bool = False, log_level: int = logging.INFO):
    """
    로깅 설정
    
    Args:
        log_to_file: 파일에 로그를 기록할지 여부
        log_level: 로깅 레벨 (기본값: INFO)
    """
    # 로그 포맷 설정
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 콘솔 핸들러 추가
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 파일 핸들러 추가 (선택사항)
    if log_to_file:
        # 로그 디렉토리 생성
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # 현재 날짜를 파일명에 포함
        log_filename = f"{log_dir}/app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # 파일 핸들러 설정
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # 로깅 설정 완료 알림
    logging.info("로깅 설정 완료")
    
    return root_logger