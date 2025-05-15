# ui/sidebar.py
"""
앱 사이드바

설정 요약 및 빠른 설정을 위한 사이드바
"""

import streamlit as st
import os
import torch
from typing import Dict, Any
import logging

from services.qdrant_service import QdrantService
from services.lm_service import LMStudioService
from utils.model_utils import get_downloaded_models

logger = logging.getLogger(__name__)

def render_sidebar(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    사이드바 렌더링

    Args:
        config: 현재 설정 딕셔너리

    Returns:
        업데이트된 설정 딕셔너리
    """
    st.sidebar.title("⚙️ 빠른 설정")

    # 입력으로 받은 config가 None이면 빈 딕셔너리로 초기화
    if config is None:
        config = {}

    # 현재 설정 복사 (원본 변경 방지)
    updated_config = config.copy()

    # 서비스 카테고리 선택
    service_category = st.sidebar.radio(
        "서비스 설정",
        options=["Qdrant", "임베딩", "LM Studio"],
        index=0,
        horizontal=True
    )

    # 선택한 카테고리에 따라 다른 설정 표시
    if service_category == "Qdrant":
        qdrant_settings = render_qdrant_settings(config)
        updated_config.update(qdrant_settings)
    elif service_category == "임베딩":
        embedding_settings = render_embedding_settings(config)
        updated_config.update(embedding_settings)
    elif service_category == "LM Studio":
        lm_settings = render_lm_studio_settings(config)
        updated_config.update(lm_settings)

    # 앱 정보
    st.sidebar.markdown("---")
    st.sidebar.caption("문서 처리 파이프라인 v1.2")
    st.sidebar.caption("PDF → Markdown → 임베딩 → Qdrant")

    return updated_config

def render_qdrant_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """Qdrant 설정 렌더링"""
    st.sidebar.subheader("Qdrant 설정")
    qdrant_status = st.sidebar.empty()

    # Qdrant URL 설정 - config에서 값 가져오기
    qdrant_url = st.sidebar.text_input(
        "Qdrant URL",
        value=config.get("qdrant_url", "http://localhost:6333"),
        key="sidebar_qdrant_url"
    )

    # Qdrant 연결 테스트
    try:
        qdrant = QdrantService(qdrant_url)
        collections = qdrant.get_collections()
        qdrant_status.success(f"Qdrant 연결됨: {len(collections)}개 컬렉션")
    except Exception as e:
        qdrant_status.error(f"Qdrant 연결 실패: {str(e)}")
        collections = []

    # 컬렉션 선택 (있을 경우)
    collection_name = config.get("collection_name", "documents")
    if collections:
        selected_index = 0
        # 이미 존재하는 컬렉션이면 해당 인덱스 찾기
        if collection_name in collections:
            selected_index = collections.index(collection_name)

        selected_collection = st.sidebar.selectbox(
            "컬렉션 선택",
            options=collections,
            index=selected_index,
            key="sidebar_collection_selectbox"
        )
        collection_name = selected_collection

    # 변경된 설정 반환
    return {
        "qdrant_url": qdrant_url,
        "collection_name": collection_name
    }

def render_embedding_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """임베딩 설정 렌더링"""
    st.sidebar.subheader("임베딩 모델")

    # 다운로드된 모델 가져오기
    downloaded_models = get_downloaded_models("embedding")

    # 현재 선택된 모델
    current_model = config.get("st_model", os.getenv("ST_MODEL", "dragonkue/snowflake-arctic-embed-l-v2.0-ko"))

    if downloaded_models:
        # 기본 선택 인덱스 계산
        default_index = 0

        # 모델 ID로 찾기
        if current_model in downloaded_models:
            default_index = downloaded_models.index(current_model)
        # 경로의 마지막 부분으로 찾기 (author/name 형식인 경우)
        elif "/" in current_model:
            model_name = current_model.split("/")[-1]
            matching_models = [i for i, m in enumerate(downloaded_models) if m.endswith("/" + model_name)]
            if matching_models:
                default_index = matching_models[0]

        selected_model = st.sidebar.selectbox(
            "임베딩 모델",
            options=downloaded_models,
            index=default_index,
            key="sidebar_model_selectbox"
        )

        # 모델 정보 표시
        if "/" in selected_model:
            display_name = selected_model.split("/")[-1]
            st.sidebar.info(f"선택된 모델: {display_name} (from {selected_model.split('/')[0]})")
        else:
            st.sidebar.info(f"선택된 모델: {selected_model}")
    else:
        # 모델 입력 (다운로드된 모델이 없는 경우)
        st.sidebar.warning("다운로드된 모델이 없습니다. 설정 탭에서 모델을 다운로드하세요.")
        selected_model = current_model

    # 디바이스 선택
    current_device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = st.sidebar.selectbox(
        "디바이스",
        options=["cuda", "cpu"],
        index=0 if torch.cuda.is_available() and current_device == "cuda" else 1,
        key="sidebar_device_selectbox"
    )

    # 변경된 설정 반환
    return {
        "st_model": selected_model,
        "device": device
    }

def render_lm_studio_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """LM Studio 설정 렌더링"""
    st.sidebar.subheader("LM Studio 설정")
    lm_status = st.sidebar.empty()

    # LM Studio URL 설정
    lm_studio_url = st.sidebar.text_input(
        "LM Studio URL",
        value=config.get("lm_studio_url", "http://localhost:1234/v1"),
        key="sidebar_lm_studio_url"
    )

    # LM Studio 연결 테스트
    lm_service = LMStudioService(api_url=lm_studio_url)
    available_models = []

    if st.sidebar.button("연결 테스트", key="sidebar_test_lm_connection"):
        try:
            if lm_service.test_connection():
                models_data = lm_service.get_available_models()
                if "data" in models_data and models_data["data"]:
                    available_models = [model["id"] for model in models_data["data"]]
                    lm_status.success(f"LM Studio 연결됨: {len(available_models)}개 모델")

                    # 세션 상태에 모델 목록 저장
                    st.session_state.lm_studio_models = available_models
                else:
                    lm_status.warning("LM Studio 연결됨, 모델 없음")
            else:
                lm_status.error("LM Studio 연결 실패")
        except Exception as e:
            lm_status.error(f"LM Studio 오류: {str(e)}")

    # 세션에 저장된 모델 목록 사용
    if "lm_studio_models" in st.session_state:
        available_models = st.session_state.lm_studio_models

    # 모델 선택 또는 입력
    current_lm_model = config.get("default_lm_model", "")
    selected_model = current_lm_model

    if available_models:
        default_index = 0
        if current_lm_model in available_models:
            default_index = available_models.index(current_lm_model)

        selected_model = st.sidebar.selectbox(
            "기본 LM 모델",
            options=available_models,
            index=default_index,
            key="sidebar_lm_model_selectbox"
        )
    else:
        if not lm_status.empty:
            st.sidebar.warning("사용 가능한 모델 목록을 가져오지 못했습니다.")

    # 변경된 설정 반환
    return {
        "lm_studio_url": lm_studio_url,
        "default_lm_model": selected_model
    }