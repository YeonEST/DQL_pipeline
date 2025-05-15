# ui/setting_tab.py
"""
설정 탭

앱 설정을 관리하는 탭
"""

import streamlit as st
import os
from typing import Dict, Any, Optional, List, Tuple
import logging
import torch

from utils.model_utils import get_downloaded_models, download_model, get_embedding_dimension, get_model_info
from services.qdrant_service import QdrantService
from services.lm_service import LMStudioService

logger = logging.getLogger(__name__)

def render_settings_tab(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    설정 탭 렌더링

    Args:
        config: 현재 설정 딕셔너리

    Returns:
        업데이트된 설정 딕셔너리 또는 None (변경 없음)
    """
    from utils.presets import get_preset_list, save_preset, load_preset, delete_preset, clear_api_cache
    import time

    st.header("⚙️ 설정")

    # 현재 설정 복사 (원본 변경 방지)
    updated_config = config.copy()

    # 설정 프리셋 관리
    st.subheader("설정 프리셋")
    col1, col2 = st.columns([3, 1])

    with col1:
        # 프리셋 선택
        preset_list = ["-- 새 프리셋 --"] + get_preset_list()
        selected_preset = st.selectbox(
            "설정 프리셋 선택",
            options=preset_list,
            index=0,
            key="preset_selectbox"
        )
        # 프리셋 이름 입력
        if selected_preset == "-- 새 프리셋 --":
            preset_name = st.text_input("새 프리셋 이름", key="new_preset_name")
        else:
            preset_name = selected_preset

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # 간격 조정
        # 프리셋 관리 버튼
        if st.button("프리셋 저장", key="save_preset_button"):
            if preset_name and preset_name != "-- 새 프리셋 --":
                # 현재 config에서 중요 설정만 저장
                settings_to_save = {
                    "st_model": updated_config.get("st_model", ""),
                    "device": updated_config.get("device", ""),
                    "qdrant_url": updated_config.get("qdrant_url", ""),
                    "collection_name": updated_config.get("collection_name", ""),
                    "lm_studio_url": updated_config.get("lm_studio_url", ""),
                    "default_lm_model": updated_config.get("default_lm_model", ""),
                    "default_temperature": updated_config.get("default_temperature", 0.2),
                    "default_top_p": updated_config.get("default_top_p", 0.95),
                    "default_top_k": updated_config.get("default_top_k", 40),
                    "default_min_p": updated_config.get("default_min_p", 0.05),
                    "default_context_template": updated_config.get("default_context_template", "custom"),
                }

                success = save_preset(preset_name, settings_to_save)
                if success:
                    st.success(f"프리셋 '{preset_name}' 저장 완료!")
                else:
                    st.error(f"프리셋 저장 실패!")
            else:
                st.error("유효한 프리셋 이름을 입력해주세요.")

        if selected_preset != "-- 새 프리셋 --":
            # 프리셋 로드 버튼
            if st.button("프리셋 로드", key="load_preset_button"):
                preset_data = load_preset(selected_preset)
                if preset_data:
                    # 설정에 프리셋 적용
                    for key, value in preset_data.items():
                        if key != "last_modified":  # 메타데이터 제외
                            updated_config[key] = value
                    st.success(f"프리셋 '{selected_preset}' 로드 완료!")
                    # 중앙 설정 저장소에 반영
                    st.session_state.config = updated_config
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"프리셋 로드 실패!")

            # 프리셋 삭제 버튼
            if st.button("프리셋 삭제", key="delete_preset_button"):
                success = delete_preset(selected_preset)
                if success:
                    st.success(f"프리셋 '{selected_preset}' 삭제 완료!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"프리셋 삭제 실패!")

    # API 캐시 관리
    st.markdown("---")

    if st.button("API 캐시 초기화", key="clear_api_cache_button"):
        success = clear_api_cache()
        if success:
            st.success("모든 API 캐시가 초기화되었습니다.")
        else:
            st.error("API 캐시 초기화 실패!")

    st.markdown("---")
    # 설정 카테고리 선택
    settings_tab = st.radio(
        "설정 카테고리",
        ["임베딩 모델", "Qdrant 컬렉션", "LM Studio"],
        horizontal=True
    )

    st.markdown("---")

    # 선택된 카테고리에 따른 설정 섹션 렌더링
    if settings_tab == "임베딩 모델":
        # 모델 관리 섹션 렌더링
        model_settings = render_model_management_section(updated_config)
        updated_config.update(model_settings)

    elif settings_tab == "Qdrant 컬렉션":
        # Qdrant 컬렉션 관리 섹션 렌더링
        qdrant_settings = render_qdrant_collection_management(updated_config)
        updated_config.update(qdrant_settings)

    elif settings_tab == "LM Studio":
        # LM Studio 설정 섹션 렌더링
        lm_settings = render_lm_studio_settings(updated_config)
        updated_config.update(lm_settings)

    # 설정 적용 버튼
    st.markdown("---")
    if st.button("설정 적용", type="primary"):
        # 세션 상태에 설정 저장
        st.session_state.config = updated_config

        st.success("설정이 적용되었습니다!")
        return updated_config

    return None  # 설정 변경 없음

def render_model_management_section(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    모델 관리 섹션을 렌더링합니다.

    Args:
        config: 현재 설정 딕셔너리

    Returns:
        업데이트된 설정 딕셔너리
    """
    st.subheader("임베딩 모델 관리")

    # 현재 설정 복사 (원본 변경 방지)
    updated_config = {}

    # 디바이스 선택
    current_device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = st.selectbox(
        "디바이스",
        options=["cuda", "cpu"],
        index=0 if torch.cuda.is_available() and current_device == "cuda" else 1,
        key="settings_device_selectbox"
    )
    updated_config["device"] = device

    # 다운로드된 모델 목록 가져오기
    downloaded_models = get_downloaded_models("embedding")

    # 모델 선택 영역과 다운로드 영역을 컬럼으로 분리
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("설치된 모델 선택")

        # 모델 목록이 있는 경우 선택 드롭다운 표시
        selected_model_name = None

        if downloaded_models:
            # 현재 선택된 모델
            current_model = config.get("st_model", "")
            selected_model_index = 0

            # 모델 ID가 정확히 일치하는지 확인
            if current_model in downloaded_models:
                selected_model_index = downloaded_models.index(current_model)
            # 또는 모델 이름의 마지막 부분이 일치하는지 확인
            elif "/" in current_model:
                model_name = current_model.split("/")[-1]
                matching_models = [i for i, m in enumerate(downloaded_models) if m.endswith("/" + model_name)]
                if matching_models:
                    selected_model_index = matching_models[0]

            selected_model_name = st.selectbox(
                "설치된 모델",
                options=downloaded_models,
                index=selected_model_index,
                key="model_select_dropdown"
            )

            # 선택된 모델 정보 표시
            if selected_model_name:
                try:
                    # 모델 정보 로드 시도
                    model_info = get_model_info(selected_model_name)

                    # 정보 표시
                    if model_info["is_loaded"]:
                        st.success(f"모델: {selected_model_name}\n임베딩 차원: {model_info['dimension']}")

                        # 추가 정보가 있으면 표시
                        if "model_type" in model_info and model_info["model_type"] != "Unknown":
                            st.info(f"모델 타입: {model_info['model_type']}")

                        # 설정에 선택된 모델 저장
                        updated_config["st_model"] = selected_model_name
                        updated_config["selected_model_dimension"] = model_info["dimension"]
                    else:
                        # 로드 실패 시 오류 메시지 표시
                        st.error(f"모델 정보 로드 실패: {model_info.get('error', '알 수 없는 오류')}")
                        st.info("기본 임베딩 차원 768을 사용합니다.")
                        updated_config["st_model"] = selected_model_name
                        updated_config["selected_model_dimension"] = 768
                except Exception as e:
                    st.error(f"모델 정보 로드 중 오류: {str(e)}")
                    st.info("기본 임베딩 차원 768을 사용합니다.")
                    updated_config["st_model"] = selected_model_name
                    updated_config["selected_model_dimension"] = 768
        else:
            st.info("설치된 모델이 없습니다. 모델을 다운로드해주세요.")

    with col2:
        st.subheader("새 모델 다운로드")
        new_model_name = st.text_input(
            "모델 식별자(HuggingFace)",
            value="dragonkue/snowflake-arctic-embed-l-v2.0-ko",
            key="new_model_input"
        )

        download_status = st.empty()

        if st.button("모델 다운로드", key="download_model_button"):
            with st.spinner(f"'{new_model_name}' 모델 다운로드 중..."):
                success = download_model(new_model_name)
                if success:
                    download_status.success(f"'{new_model_name}' 모델 다운로드 완료!")
                    # 설정 업데이트 후 페이지 새로고침
                    updated_config["st_model"] = new_model_name
                    st.session_state.config.update(updated_config)
                    st.rerun()
                else:
                    download_status.error(f"'{new_model_name}' 모델 다운로드 실패!")

    return updated_config

# ui/setting_tab.py (계속)

def render_qdrant_collection_management(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Qdrant 컬렉션 관리 섹션을 렌더링합니다.

    Args:
        config: 현재 설정 딕셔너리

    Returns:
        업데이트된 설정 딕셔너리
    """
    st.subheader("Qdrant 컬렉션 관리")

    # 현재 설정 복사 (원본 변경 방지)
    updated_config = {}

    # Qdrant 설정
    col1, col2 = st.columns(2)
    with col1:
        qdrant_url = st.text_input(
            "Qdrant URL",
            value=config.get("qdrant_url", "http://localhost:6333")
        )
        updated_config["qdrant_url"] = qdrant_url

    with col2:
        default_collection = st.text_input(
            "기본 컬렉션 이름",
            value=config.get("collection_name", "documents")
        )
        updated_config["collection_name"] = default_collection

    # Qdrant 서비스 초기화
    qdrant_service = QdrantService(qdrant_url=qdrant_url)

    # 컬렉션 가져오기
    if st.button("컬렉션 목록 가져오기"):
        with st.spinner("Qdrant에서 컬렉션 목록을 가져오는 중..."):
            collections = qdrant_service.get_collections(use_cache=False)
            if collections:
                st.session_state.collections = collections
                st.success(f"{len(collections)}개 컬렉션을 가져왔습니다: {', '.join(collections)}")
            else:
                st.warning("컬렉션을 가져오지 못했습니다. Qdrant 설정을 확인하세요.")

    # 새 컬렉션 생성
    with st.expander("새 컬렉션 생성"):
        new_collection_name = st.text_input("새 컬렉션 이름", value="new_collection")

        # 현재 선택된 모델의 차원 정보 가져오기
        current_model = config.get("st_model", "")
        vector_size = 768  # 기본값

        if current_model:
            try:
                # 모델 정보 가져오기
                model_info = get_model_info(current_model)
                vector_size = model_info.get("dimension", 768)
                st.info(f"현재 선택된 모델 '{current_model}'의 차원: {vector_size}")
            except Exception as e:
                st.warning(f"모델 차원 확인 중 오류: {str(e)}. 기본값 768을 사용합니다.")

        # 벡터 차원 슬라이더
        custom_dimension = st.checkbox("사용자 정의 벡터 차원 사용", value=False)

        if custom_dimension:
            vector_size = st.slider(
                "벡터 차원 크기",
                min_value=32,
                max_value=2048,
                value=vector_size,
                step=32,
                key="vector_dimension_slider"
            )

        # 컬렉션 생성 버튼
        if st.button("컬렉션 생성"):
            with st.spinner(f"'{new_collection_name}' 컬렉션 생성 중..."):
                success = qdrant_service.create_collection(new_collection_name, vector_size)
                if success:
                    st.success(f"컬렉션 '{new_collection_name}'이(가) 생성되었습니다!")
                    if "collections" not in st.session_state:
                        st.session_state.collections = []
                    if new_collection_name not in st.session_state.collections:
                        st.session_state.collections.append(new_collection_name)

                    # 새로 생성한 컬렉션을 현재 컬렉션으로 설정할지 물어보기
                    if st.checkbox("새 컬렉션을 현재 컬렉션으로 설정", value=True):
                        updated_config["collection_name"] = new_collection_name
                else:
                    st.error(f"컬렉션 '{new_collection_name}' 생성 실패!")

    return updated_config

def render_lm_studio_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    LM Studio 설정 섹션을 렌더링합니다.

    Args:
        config: 현재 설정 딕셔너리

    Returns:
        업데이트된 설정 딕셔너리
    """
    st.subheader("LM Studio 설정")

    # 현재 설정 복사 (원본 변경 방지)
    updated_config = {}

    # LM Studio URL
    lm_studio_url = st.text_input(
        "LM Studio API URL",
        value=config.get("lm_studio_url", "http://localhost:1234/v1")
    )
    updated_config["lm_studio_url"] = lm_studio_url

    # API 키 (기본값: "lm-studio-local")
    lm_studio_key = st.text_input(
        "LM Studio API 키",
        value=config.get("lm_studio_key", "lm-studio-local"),
        type="password"
    )
    updated_config["lm_studio_key"] = lm_studio_key

    # LM Studio 서비스 초기화
    lm_service = LMStudioService(api_url=lm_studio_url, api_key=lm_studio_key)

    # 연결 테스트
    if st.button("LM Studio 연결 테스트"):
        with st.spinner("LM Studio 연결 테스트 중..."):
            if lm_service.test_connection():
                st.success("LM Studio 연결 성공!")

                # 사용 가능한 모델 목록 가져오기
                models_data = lm_service.get_available_models(use_cache=False)
                if "data" in models_data and models_data["data"]:
                    models = [model["id"] for model in models_data["data"]]
                    st.info(f"사용 가능한 모델 ({len(models)}개): {', '.join(models)}")

                    # 세션 상태에 모델 목록 저장
                    st.session_state.lm_studio_models = models

                    # 모델 선택 섹션 표시
                    current_model = config.get("default_lm_model", "")
                    default_index = 0

                    if current_model in models:
                        default_index = models.index(current_model)

                    selected_model = st.selectbox(
                        "기본 모델 선택",
                        options=models,
                        index=default_index
                    )
                    updated_config["default_lm_model"] = selected_model
                else:
                    st.warning("사용 가능한 모델이 없습니다.")
            else:
                st.error("LM Studio 연결 실패! URL과 API 키를 확인하세요.")

    # 기본 생성 파라미터 설정
    with st.expander("기본 생성 파라미터 설정", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            # Temperature
            default_temperature = st.slider(
                "기본 Temperature",
                min_value=0.0,
                max_value=1.0,
                value=config.get("default_temperature", 0.2),
                step=0.05,
                key="default_temperature_slider"
            )
            updated_config["default_temperature"] = default_temperature

            # Top P
            default_top_p = st.slider(
                "기본 Top P",
                min_value=0.0,
                max_value=1.0,
                value=config.get("default_top_p", 0.95),
                step=0.01,
                key="default_top_p_slider"
            )
            updated_config["default_top_p"] = default_top_p

        with col2:
            # Top K
            default_top_k = st.slider(
                "기본 Top K",
                min_value=1,
                max_value=100,
                value=config.get("default_top_k", 40),
                step=1,
                key="default_top_k_slider"
            )
            updated_config["default_top_k"] = default_top_k

            # Min P
            default_min_p = st.slider(
                "기본 Min P",
                min_value=0.0,
                max_value=0.5,
                value=config.get("default_min_p", 0.05),
                step=0.01,
                key="default_min_p_slider"
            )
            updated_config["default_min_p"] = default_min_p

        # 컨텍스트 템플릿 선택
        context_templates = ["custom", "chatML", "Alpaca", "mistral"]
        default_context_template = st.selectbox(
            "기본 컨텍스트 템플릿",
            options=context_templates,
            index=context_templates.index(config.get("default_context_template", "custom")),
            key="default_context_template_select"
        )
        updated_config["default_context_template"] = default_context_template

    # 기본 프롬프트 템플릿 설정
    with st.expander("기본 마크다운 변환 프롬프트 템플릿", expanded=False):
        default_prompt = config.get("default_markdown_prompt", """다음 텍스트를 깔끔한 마크다운 형식으로 변환해주세요. 
원본 텍스트는 PDF에서 추출되어 구조가 깨진 상태일 수 있습니다.
적절한 제목, 부제목, 목록, 인용 등의 마크다운 요소를 사용하여 구조화된 문서로 만들어주세요.
불필요한 설명은 제외하고 변환된 마크다운만 출력해주세요.

원본 텍스트:
{text}

변환된 마크다운:""")

        markdown_prompt = st.text_area(
            "기본 프롬프트 템플릿",
            value=default_prompt,
            height=300
        )
        updated_config["default_markdown_prompt"] = markdown_prompt

        # 프롬프트 템플릿 재설정 버튼
        if st.button("기본값으로 재설정"):
            updated_config["default_markdown_prompt"] = default_prompt
            st.session_state.config.update(updated_config)
            st.rerun()

    return updated_config