# utils/presets.py
"""
사용자 설정 프리셋 및 API 정보 관리 유틸리티

사용자 설정을 저장하고 로드하는 기능과 API에서 받아온 정보를 캐싱하는 기능을 제공합니다.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import time

logger = logging.getLogger(__name__)

# 기본 디렉토리 설정
PRESETS_DIR = "user"
CACHE_DIR = "cache"

def ensure_dirs():
    """필요한 디렉토리 생성"""
    os.makedirs(PRESETS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

def get_preset_list() -> List[str]:
    """
    사용 가능한 프리셋 목록 반환

    Returns:
        프리셋 이름 목록
    """
    ensure_dirs()
    preset_files = [f for f in os.listdir(PRESETS_DIR) if f.endswith('.json')]
    return [os.path.splitext(f)[0] for f in preset_files]

def save_preset(name: str, settings: Dict[str, Any]) -> bool:
    """
    사용자 설정 프리셋 저장

    Args:
        name: 프리셋 이름
        settings: 저장할 설정 딕셔너리

    Returns:
        성공 여부
    """
    ensure_dirs()
    try:
        # 타임스탬프 추가
        settings['last_modified'] = datetime.now().isoformat()

        # 파일 저장
        preset_path = os.path.join(PRESETS_DIR, f"{name}.json")
        with open(preset_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)

        logger.info(f"프리셋 저장 완료: {name}")
        return True
    except Exception as e:
        logger.error(f"프리셋 저장 실패: {e}", exc_info=True)
        return False

def load_preset(name: str) -> Optional[Dict[str, Any]]:
    """
    사용자 설정 프리셋 로드

    Args:
        name: 프리셋 이름

    Returns:
        설정 딕셔너리 또는 None (실패시)
    """
    ensure_dirs()
    preset_path = os.path.join(PRESETS_DIR, f"{name}.json")
    try:
        if os.path.exists(preset_path):
            with open(preset_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            logger.info(f"프리셋 로드 완료: {name}")
            return settings
        else:
            logger.warning(f"프리셋을 찾을 수 없음: {name}")
            return None
    except Exception as e:
        logger.error(f"프리셋 로드 실패: {e}", exc_info=True)
        return None

def delete_preset(name: str) -> bool:
    """
    사용자 설정 프리셋 삭제

    Args:
        name: 프리셋 이름

    Returns:
        성공 여부
    """
    ensure_dirs()
    preset_path = os.path.join(PRESETS_DIR, f"{name}.json")
    try:
        if os.path.exists(preset_path):
            os.remove(preset_path)
            logger.info(f"프리셋 삭제 완료: {name}")
            return True
        else:
            logger.warning(f"삭제할 프리셋을 찾을 수 없음: {name}")
            return False
    except Exception as e:
        logger.error(f"프리셋 삭제 실패: {e}", exc_info=True)
        return False

def save_api_cache(name: str, data: Dict[str, Any], max_age_hours: int = 24) -> bool:
    """
    API 응답 데이터 캐싱

    Args:
        name: 캐시 항목 이름
        data: 저장할 데이터
        max_age_hours: 최대 캐시 유효 시간 (시간)

    Returns:
        성공 여부
    """
    ensure_dirs()
    try:
        # 타임스탬프 및 유효기간 추가
        cache_data = {
            'data': data,
            'timestamp': time.time(),
            'expires_at': time.time() + (max_age_hours * 3600)
        }

        # 파일 저장
        cache_path = os.path.join(CACHE_DIR, f"{name}.json")
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)

        logger.info(f"API 캐시 저장 완료: {name}")
        return True
    except Exception as e:
        logger.error(f"API 캐시 저장 실패: {e}", exc_info=True)
        return False

def load_api_cache(name: str, max_age_hours: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    API 응답 데이터 로드

    Args:
        name: 캐시 항목 이름
        max_age_hours: 최대 캐시 유효 시간 (시간) - 지정하면 이전 설정 무시

    Returns:
        캐시된 데이터 또는 None (캐시 없음 또는 만료)
    """
    ensure_dirs()
    cache_path = os.path.join(CACHE_DIR, f"{name}.json")
    try:
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            # 만료 확인
            current_time = time.time()
            expires_at = cache_data.get('expires_at')

            # max_age_hours가 명시적으로 설정된 경우 유효 기간 업데이트
            if max_age_hours is not None:
                expires_at = cache_data.get('timestamp', current_time) + (max_age_hours * 3600)

            if expires_at and current_time < expires_at:
                logger.info(f"API 캐시 로드 완료: {name}")
                return cache_data.get('data')
            else:
                logger.info(f"API 캐시가 만료됨: {name}")
                return None
        else:
            logger.info(f"API 캐시를 찾을 수 없음: {name}")
            return None
    except Exception as e:
        logger.error(f"API 캐시 로드 실패: {e}", exc_info=True)
        return None

def clear_api_cache(name: Optional[str] = None) -> bool:
    """
    API 캐시 삭제

    Args:
        name: 특정 캐시 항목 이름 (None이면 모든 캐시 삭제)

    Returns:
        성공 여부
    """
    ensure_dirs()
    try:
        if name:
            # 특정 캐시 파일만 삭제
            cache_path = os.path.join(CACHE_DIR, f"{name}.json")
            if os.path.exists(cache_path):
                os.remove(cache_path)
                logger.info(f"API 캐시 삭제 완료: {name}")
            else:
                logger.warning(f"삭제할 API 캐시를 찾을 수 없음: {name}")
        else:
            # 모든 캐시 파일 삭제
            cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.json')]
            for cache_file in cache_files:
                os.remove(os.path.join(CACHE_DIR, cache_file))
            logger.info(f"모든 API 캐시 삭제 완료 ({len(cache_files)}개)")

        return True
    except Exception as e:
        logger.error(f"API 캐시 삭제 실패: {e}", exc_info=True)
        return False