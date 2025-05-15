# ui/pdf_to_md_tab.py
"""
PDF를 마크다운으로 변환하는 UI 탭

PDF 파일을 업로드하여 마크다운으로 변환하고 저장하는 UI를 제공합니다.
LM Studio를 사용한 마크다운 후처리 기능도 지원합니다.
"""

import streamlit as st
import os
import tempfile
import pandas as pd
import time
from typing import Dict, Any, List
import logging
from pathlib import Path

from services.pdf_service import PDFService
from services.lm_service import LMStudioService

logger = logging.getLogger(__name__)

# 기본 프롬프트 템플릿
DEFAULT_PROMPT_TEMPLATE = """끝에 있는 입력 내용에 대해 다음 마크다운 노트 형식으로 정리해줘:
노트는 다음 형식을 따라야 해:
1. 주요 개념은 ## 제목으로 시작
2. 하위 개념은 ###, #### 제목으로 시작하고, 더 세부적인 내용은 #####이나 ###### 처럼 세부적일수록 #의 개수를 늘려가
3. 중요 용어와 개념은 **볼드체**로 강조
4. 항목은 - 기호로 시작하는 불릿 리스트로 정리
5. 항목의 세부 내용은 들여쓰기와 함께 - 기호 사용
6. 각 섹션은 명확하게 구분하고 --- 구분선 활용
7. 정의나 설명에서 핵심 내용을 간결하게 표현
8. 관련 개념들은 체계적으로 하위 리스트로 구조화
9. 핵심 포인트는 리스트 형태로 정리
10. 필요시 표나 코드 블록 활용
위 형식에 맞게 정보를 체계적으로 정리하고, 흐름이 논리적으로 이어지도록 구성해줘.

그리고 내용이 중간에 사라지거나 끊기는 등 손실이 발생되었을 수 있는데, 관련 내용으로 복구 및 보강을 해서 작성해줘.

예시:

## 네트워크 모델
### OSI 7 계층 통신 프로토콜
#### OSI 7 계층 모델 개요
- 네트워크 동작 과정을 설명하는 대표적인 model
- ISO 에서 제정
- 실제 시스템 X, 개념적 모델 O
    - 추가내용) TCP/IP는 실제 시스템
#### 데이터 단위 정리
- 응용 계층(L7): 메시지(Message) 또는 데이터(Data)
- 표현 계층(L6): 데이터(Data)
- 세션 계층(L5): 데이터(Data)
- 전송 계층(L4): 세그먼트(Segment, TCP) / 데이터그램(Datagram, UDP)
- 네트워크 계층(L3): 패킷(Packet)
- 데이터링크 계층(L2): 프레임(Frame)
- 물리 계층(L1): 비트(Bit)
#### 계층
##### L7 : 응용 게층 (Application Layer)
- 사용자에 UI 제공
##### L6 : 표현 계층 (Presentaoin Layer)
- 7 계층 (응용)에서 다루는 데이터의 형식을 변환(Format conversion) 함
- 암호화, 복호화, 인코딩, 디코딩 등 수행
##### L5 : 세션 계층 (Session Layer)
- 두 개의 응용 프로세스 사이 통신 관리
- 통신 = 세션
- 전이중 (Full Duplex) : 동시에 송신/수신 2개 다 가능
    - Intra band Full Duplex
    - Inter band Full Duplex
- 반이중 (Half Duplex) : 한 번에 송신/수신 둘 중 1개만
- 단방향 (Simpelx) : 송신/수신 중 1가지만 가능 (다른거 불가능)
##### L4 : 전송 계층 (Transport Layer)
- 데이터 전송 시 Loss가 없게 보장
- 흐름 제어 (Flow control)을 담당
    - 양 끝단 (End to End)에서 받은 데이터 오류를 검출 (패리티?)
    - 오류가 있다면 재전송 요청
- 전송 계층 상하 관계
    - 상위 계층 L5 (세션 계층)는 하위 계층 L4 (전송 계층)에 관여 X
    - L5의 1개 메시지 => 2개의 L4 Segment로 쪼개짐
    - 각 Segment는 네트워크 전송에 적절하게 쪼갬
    - 페이로드 (Payload) : 상위 계층의 데이터
    - 캡슐화 (Encapsulation)
        - 쪼갠 메세지에 Header를 더함
        - 하위 계층에서 상위 계층의 Payload에 헤더를 붙이는 것
        - L4의 Header : Flow control을 위한 정보 포함
        - 오류 검출에 헤더를 사용가능
- L4의 예시
    - TCP
        - 흐름제어 O = 고신뢰성
        - 느림
    - UDP
        - 흐름제어 X = 저신뢰성
        - 빠름
##### L3 : 네트워크 계층 (Network Layer)
- 라우팅 (Routing)을 처리
    - 라우팅 : 데이터 전송 경로를 선택
- Segment (L4의 데이터)에 라우팅 관련 L3 Header 추가
- 패킷 (Packet) : Segment + L3 Header를 해서 캡슐화 한 3 계층 데이터 단위
- 패킷 헤더 : 경로 선택을 위한 기초 자료
    - 자료 => 각 노드 사이 전송 비용 추정치
    - 기존 경로를 이용 불가능해지면 헤더의 추정치를 통해 대체 경로를 찾음
- 예시 : IP (Internet Protocol) 프로토콜
##### L2 : 데이터 링크 계층 (Data Link Layer)
- 단말 사이 전송의 신뢰성을 보장
- Point to Point 간 신뢰성
    - L1 (물리 계층)의 오류를 찾고 수정
    - 패리티 검사 (Parity Check) 등 기법 사용
- L2의 데이터 Encapsulation
    - 프레임 (Frame) : Payload인 Packet의 *앞, 뒤*에 Header와 Tail을 붙임
- MAC 관련 계층
    - MAC Medium Access Control 
- 예시 : Ethernet, Token Ring
##### L1 : 물리 계층 (Physical Layer)
- 단말 사이를 물리적으로 연결
- 단말 (Terminal)
    - 네트워크 간 연결을 하는 노드 (L3 계층) 포함
    - 네트워크에 참여한 모든 단말을 포함
---

원본 텍스트:
{text}

변환된 마크다운:"""

def render_pdf_to_md_tab(config: Dict[str, Any]):
    """
    PDF를 마크다운으로 변환하는 탭 렌더링

    Args:
        config: 앱 설정 딕셔너리
    """
    st.header("PDF → Markdown 변환")

    # 탭 선택
    tab1, tab2 = st.tabs(["기본 변환", "고급 변환 (LM Studio)"])

    with tab1:
        render_basic_conversion_tab(config)

    with tab2:
        render_advanced_conversion_tab(config)

def render_basic_conversion_tab(config: Dict[str, Any]):
    """
    기본 변환 탭 렌더링

    Args:
        config: 앱 설정 딕셔너리
    """
    st.write("PDF 파일을 업로드하여 마크다운으로 변환합니다. 변환된 파일은 'output/md/' 디렉토리에 저장됩니다.")

    # 마크다운 출력 디렉토리 표시
    md_output_dir = os.path.abspath(config.get("markdown_dir", "output/md"))
    st.info(f"마크다운 출력 디렉토리: {md_output_dir}")

    # CPU 모드 설정
    use_cpu = st.checkbox("CPU 모드 사용 (GPU 오류가 발생할 경우 선택)", value=True)
    if use_cpu:
        st.caption("CPU 모드가 활성화되었습니다. 변환 속도가 느려질 수 있지만 호환성이 향상됩니다.")

    # PDF 파일 업로더
    uploaded_files = st.file_uploader(
        "PDF 파일을 업로드하세요 (드래그 앤 드롭 가능)",
        type=["pdf"],
        accept_multiple_files=True,
        key="basic_pdf_uploader"
    )

    if uploaded_files:
        if st.button("PDF 파일 변환하기", type="primary", key="basic_convert_button"):
            # PDF 서비스 초기화 - CPU 모드 설정 적용
            pdf_service = PDFService(output_dir=md_output_dir, use_cpu=use_cpu)

            # 진행 상황 표시를 위한 컴포넌트
            progress_text = st.empty()
            progress_bar = st.progress(0)

            for i, uploaded_file in enumerate(uploaded_files):
                # 진행 상황 업데이트
                progress = i / len(uploaded_files)
                progress_bar.progress(progress)
                progress_text.info(f"({i+1}/{len(uploaded_files)}) '{uploaded_file.name}' 처리 중...")

                # 임시 파일 생성
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
                    temp.write(uploaded_file.getvalue())
                    temp_path = temp.name

                status_text = st.empty()
                status_text.info(f"'{uploaded_file.name}' 마크다운 변환 중...")

                try:
                    # 시작 시간 기록
                    start_time = time.time()

                    # PDF를 마크다운으로 변환
                    result = pdf_service.convert_pdf_to_markdown(
                        pdf_path=temp_path,
                        original_filename=uploaded_file.name
                    )

                    # 처리 시간 계산
                    elapsed_time = time.time() - start_time

                    if result.get("success", False):
                        # 처리 결과 업데이트
                        status_text.success(
                            f"✅ '{uploaded_file.name}' 변환 완료! -> {result.get('output_file')}"
                            f" (처리 시간: {elapsed_time:.2f}초)"
                        )

                        # 세션 상태에 처리 이력 추가
                        if "processed_files" not in st.session_state:
                            st.session_state.processed_files = []

                        st.session_state.processed_files.append({
                            "filename": uploaded_file.name,
                            "output": result.get("output_file", ""),
                            "size_kb": result.get("file_size_kb", 0),
                            "type": "PDF→Markdown (Basic)",
                            "status": "완료",
                            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                    else:
                        # 오류 처리
                        status_text.error(f"❌ '{uploaded_file.name}' 변환 실패! 오류: {result.get('error', '알 수 없는 오류')}")

                        # 세션 상태에 처리 이력 추가
                        if "processed_files" not in st.session_state:
                            st.session_state.processed_files = []

                        st.session_state.processed_files.append({
                            "filename": uploaded_file.name,
                            "output": "",
                            "size_kb": 0,
                            "type": "PDF→Markdown (Basic)",
                            "status": "실패",
                            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                        })

                except Exception as e:
                    logger.error(f"파일 변환 중 오류 발생: {e}", exc_info=True)
                    status_text.error(f"❌ '{uploaded_file.name}' 변환 실패! 오류: {str(e)}")

                    # 세션 상태에 처리 이력 추가
                    if "processed_files" not in st.session_state:
                        st.session_state.processed_files = []

                    st.session_state.processed_files.append({
                        "filename": uploaded_file.name,
                        "output": "",
                        "size_kb": 0,
                        "type": "PDF→Markdown (Basic)",
                        "status": "실패",
                        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

                finally:
                    # 임시 파일 삭제
                    os.unlink(temp_path)

                # 진행 상황 업데이트
                progress_bar.progress((i + 1) / len(uploaded_files))

            # 모든 작업 완료
            progress_bar.progress(1.0)
            progress_text.success("모든 파일 변환 완료!")

            # 처리 완료 후 마크다운 파일 목록 표시
            st.success("모든 파일 변환이 완료되었습니다!")
            show_markdown_files(md_output_dir)

    # 기존 마크다운 파일 표시
    show_markdown_files(md_output_dir)

def show_markdown_files(md_dir: str):
    """
    마크다운 파일 목록 표시

    Args:
        md_dir: 마크다운 파일이 저장된 디렉토리 경로
    """
    st.subheader("변환된 마크다운 파일")

    # 디렉토리가 없으면 생성
    os.makedirs(md_dir, exist_ok=True)

    # 마크다운 파일 목록 가져오기
    md_files = list(Path(md_dir).glob("*.md"))

    if md_files:
        # 파일 정보 목록 생성
        file_info = []
        for md_file in md_files:
            file_info.append({
                "filename": md_file.name,
                "path": str(md_file),
                "size_kb": round(os.path.getsize(md_file) / 1024, 2),
                "modified": time.ctime(os.path.getmtime(md_file))
            })

        # 데이터프레임으로 변환하여 표시
        df = pd.DataFrame(file_info)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("변환된 마크다운 파일이 없습니다.")

# ui/pdf_to_md_tab.py (일부 - LM Studio 관련 코드)

def render_advanced_conversion_tab(config: Dict[str, Any]):
    """
    고급 변환 탭 렌더링 (LM Studio 활용)

    Args:
        config: 앱 설정 딕셔너리
    """
    st.write("PDF 텍스트를 LM Studio를 사용하여 개선된 마크다운으로 변환합니다.")

    # 마크다운 출력 디렉토리 설정 - config에서 가져오기
    md_output_dir = config.get("markdown_dir", "output/md")

    # LM Studio 설정 - config에서 가져오기
    lm_studio_url = st.text_input(
        "LM Studio URL",
        value=config.get("lm_studio_url", "http://localhost:1234/v1")
    )

    # LM Studio 연결 확인
    lm_studio_service = LMStudioService(api_url=lm_studio_url)
    lm_status = st.empty()

    # 연결 테스트 및 모델 목록 가져오기
    available_models = []
    try:
        if lm_studio_service.test_connection():
            # 세션 상태에 저장된 모델 목록 사용 (있으면)
            if "lm_studio_models" in st.session_state:
                available_models = st.session_state.lm_studio_models
                lm_status.success(f"LM Studio 연결 성공: {len(available_models)}개 모델 사용 가능")
            else:
                # 없으면 API 호출해서 가져오기
                models_data = lm_studio_service.get_available_models()
                if "data" in models_data and models_data["data"]:
                    available_models = [model["id"] for model in models_data["data"]]
                    st.session_state.lm_studio_models = available_models  # 세션에 저장
                    lm_status.success(f"LM Studio 연결 성공: {len(available_models)}개 모델 사용 가능")
                else:
                    lm_status.warning("LM Studio에 사용 가능한 모델이 없습니다.")
        else:
            lm_status.error("LM Studio 연결 실패")
    except Exception as e:
        lm_status.error(f"LM Studio 연결 오류: {str(e)}")

    # CPU 모드 설정
    use_cpu = st.checkbox("Docling CPU 모드 사용", value=True, key="advanced_cpu_mode")
    if use_cpu:
        st.caption("CPU 모드가 활성화되었습니다. 변환 속도가 느려질 수 있지만 호환성이 향상됩니다.")

    # LM 모델 설정
    col1, col2 = st.columns(2)

    with col1:
        # 모델 선택 - config 기본값 사용
        selected_model = config.get("default_lm_model", "")

        if available_models:
            # 모델 목록에서 선택
            default_index = 0
            if selected_model in available_models:
                default_index = available_models.index(selected_model)

            selected_model = st.selectbox(
                "LM Studio 모델 선택",
                options=available_models,
                index=default_index,
                key="lm_model_selector"
            )

            # 선택된 모델의 정보 가져오기
            if selected_model:
                try:
                    model_info = lm_studio_service.get_model_info(selected_model)

                    # 컨텍스트 길이 표시
                    if "context_length" in model_info:
                        context_length = model_info["context_length"]
                        source = model_info.get("context_length_source", "")

                        if source in ["estimated", "default", "fallback"]:
                            # 추정된 경우 설명 추가
                            st.info(f"모델 최대 컨텍스트 길이: 약 {context_length:,} 토큰 (추정값)")
                        else:
                            st.info(f"모델 최대 컨텍스트 길이: {context_length:,} 토큰")
                except Exception as e:
                    logger.warning(f"모델 정보 가져오기 실패: {e}")
        else:
            # 모델 직접 입력
            selected_model = st.text_input(
                "모델명 입력",
                value=selected_model,
                key="lm_model_input"
            )
            st.caption("모델 목록을 가져오지 못했습니다. 모델명을 직접 입력하세요.")

    with col2:
        # 생성 파라미터 - config 기본값 사용
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=config.get("default_temperature", 0.2),
            step=0.1,
            key="temperature_slider"
        )

    # 생성 파라미터 섹션

    # 고급 생성 파라미터 섹션
    with st.expander("고급 생성 파라미터", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            # Temperature (이미 위에서 설정했으므로 같은 값 사용)
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=temperature,
                step=0.05,
                key="advanced_temperature_slider",
                help="값이 높을수록 창의적이지만 일관성이 떨어질 수 있습니다."
            )

            # Top P
            top_p = st.slider(
                "Top P",
                min_value=0.0,
                max_value=1.0,
                value=config.get("default_top_p", 0.95),
                step=0.01,
                key="advanced_top_p_slider",
                help="확률 질량의 상위 p% 내에서 토큰 선택 (nucleus sampling)"
            )

        with col2:
            # Top K
            top_k = st.slider(
                "Top K",
                min_value=1,
                max_value=100,
                value=config.get("default_top_k", 40),
                step=1,
                key="advanced_top_k_slider",
                help="확률 상위 k개 토큰 내에서 선택"
            )

            # Min P
            min_p = st.slider(
                "Min P",
                min_value=0.0,
                max_value=0.5,
                value=config.get("default_min_p", 0.05),
                step=0.01,
                key="advanced_min_p_slider",
                help="확률 하한값 (min_p 미만인 토큰 제외)"
            )

        # 컨텍스트 템플릿 선택 - config 값 사용
        context_templates = ["custom", "chatML", "Alpaca", "mistral"]
        default_template = config.get("default_context_template", "custom")
        template_index = context_templates.index(default_template) if default_template in context_templates else 0

        context_template = st.selectbox(
            "컨텍스트 템플릿",
            options=context_templates,
            index=template_index,
            key="advanced_context_template_select",
            help="모델과의 대화 형식을 정의하는 템플릿"
        )

        # 모델 정보 및 최대 컨텍스트 길이 가져오기
        model_info_container = st.empty()
        max_context_length = 32000  # 기본값을 더 큰 값으로 설정

        if selected_model:
            try:
                with st.spinner("모델 정보 가져오는 중..."):
                    model_info = lm_studio_service.get_model_info(selected_model)

                    if "context_length" in model_info:
                        max_context_length = model_info["context_length"]
                        source = model_info.get("context_length_source", "")

                        if source in ["estimated", "default", "fallback"]:
                            model_info_container.info(f"모델 '{selected_model}'의 최대 컨텍스트 길이: 약 {max_context_length:,} 토큰 (추정값)")
                        else:
                            model_info_container.info(f"모델 '{selected_model}'의 최대 컨텍스트 길이: {max_context_length:,} 토큰")
                    else:
                        model_info_container.warning(f"모델 '{selected_model}'의 컨텍스트 길이 정보를 가져올 수 없습니다. 기본값 32,000 토큰을 사용합니다.")
            except Exception as e:
                logger.warning(f"모델 정보 가져오기 실패: {e}")
                model_info_container.warning(f"모델 정보 가져오기 실패: {str(e)}. 기본값 32,000 토큰을 사용합니다.")

        # 최대 토큰 수 설정 - 더 큰 값으로 조정
        safe_max_tokens = min(max_context_length - 1000, 151072)  # 입력 여유분 확보
        max_token_options = [1000, 2000, 4000, 8000, 12000, 16000, 20000, 24000, 28000, 32000, 40000, 60000, 131072]

        # 옵션 중 safe_max_tokens보다 작거나 같은 최대값 찾기
        default_max_tokens = next((x for x in reversed(max_token_options) if x <= safe_max_tokens), 4000)

        max_tokens = st.select_slider(
            "최대 생성 토큰 수",
            options=max_token_options,
            value=default_max_tokens,
            key="advanced_max_tokens_slider",
            help="응답으로 생성할 최대 토큰 수. 큰 문서는 더 많은 토큰이 필요할 수 있습니다."
        )

        st.caption("참고: 입력 텍스트와 생성 토큰의 합이 모델의 최대 컨텍스트 길이를 초과하면 오류가 발생할 수 있습니다.")

    # 프롬프트 템플릿 설정 - config 값 사용
    with st.expander("마크다운 변환 프롬프트 설정", expanded=False):
        # 세션에 저장된 값이 없으면 config에서 가져오기
        # render_advanced_conversion_tab 함수 내에서
        if "markdown_prompt_template" not in st.session_state:
            # 여기를 DEFAULT_PROMPT_TEMPLATE으로 변경
            st.session_state.markdown_prompt_template = DEFAULT_PROMPT_TEMPLATE

        # 컨텍스트 템플릿 설명 표시
        if context_template != "custom":
            st.info(f"{context_template} 템플릿을 사용합니다. 이 경우 프롬프트는 해당 템플릿 형식에 맞게 자동으로 포맷됩니다.")

        # Custom 템플릿인 경우에만 편집 가능
        if context_template == "custom":
            prompt_template = st.text_area(
                "프롬프트 템플릿",
                value=st.session_state.markdown_prompt_template,
                height=200,
                key="prompt_template_input",
                help="텍스트를 마크다운으로 변환하기 위한 프롬프트 템플릿. {text}는 추출된 텍스트로 대체됩니다."
            )

            # 템플릿 재설정 버튼
            if st.button("기본 템플릿으로 재설정", key="reset_template_button"):
                prompt_template = config.get("default_markdown_prompt", st.session_state.markdown_prompt_template)
                st.session_state.markdown_prompt_template = prompt_template
                st.rerun()

            # 템플릿 저장
            st.session_state.markdown_prompt_template = prompt_template
        else:
            # 다른 템플릿의 경우 읽기 전용으로 표시
            st.text_area(
                "프롬프트 템플릿 (읽기 전용)",
                value=st.session_state.markdown_prompt_template,
                height=200,
                key="prompt_template_readonly",
                disabled=True
            )
            st.caption("이 템플릿은 선택한 컨텍스트 템플릿에 맞게 자동으로 포맷됩니다. 직접 편집하려면 'custom' 템플릿을 선택하세요.")

    # PDF 파일 업로더
    uploaded_files = st.file_uploader(
        "PDF 파일을 업로드하세요 (드래그 앤 드롭 가능)",
        type=["pdf"],
        accept_multiple_files=True,
        key="advanced_pdf_uploader"
    )

    # LM Studio를 사용한 변환 버튼
    if uploaded_files and selected_model:
        if st.button("LM Studio를 사용하여 변환하기", type="primary", key="advanced_convert_button"):
            # 설정한 변수들 저장
            md_output_dir = os.path.abspath(config.get("markdown_dir", "output/md"))

            # PDF 서비스 초기화
            pdf_service = PDFService(output_dir=md_output_dir, use_cpu=use_cpu)

            # 진행 상황 표시를 위한 컴포넌트
            progress_text = st.empty()
            progress_bar = st.progress(0)

            for i, uploaded_file in enumerate(uploaded_files):
                # 진행 상황 업데이트
                progress = i / len(uploaded_files)
                progress_bar.progress(progress)
                progress_text.info(f"({i+1}/{len(uploaded_files)}) '{uploaded_file.name}' 처리 중...")

                # 임시 파일 생성
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
                    temp.write(uploaded_file.getvalue())
                    temp_path = temp.name

                # 상태 표시
                docling_status = st.empty()
                lm_status = st.empty()
                final_status = st.empty()

                try:
                    # 1. PDF를 텍스트로 변환 (저장하지 않음)
                    docling_status.info(f"'{uploaded_file.name}' 텍스트 추출 중...")
                    start_time = time.time()

                    text_result = pdf_service.convert_pdf_to_markdown(
                        pdf_path=temp_path,
                        original_filename=uploaded_file.name,
                        save_to_file=False,
                        output_format="markdown"
                    )

                    docling_elapsed = time.time() - start_time

                    if not text_result.get("success", False):
                        docling_status.error(f"텍스트 추출 실패: {text_result.get('error', '알 수 없는 오류')}")
                        continue

                    extracted_text = text_result.get("content", "")
                    docling_status.success(f"텍스트 추출 완료 ({docling_elapsed:.2f}초)")

                    # 2. LM Studio를 사용하여 텍스트를 마크다운으로 변환
                    lm_status.info(f"LM Studio를 사용하여 마크다운으로 변환 중...")
                    start_time = time.time()

                    prompt_template = st.session_state.markdown_prompt_template
                    lm_result = lm_studio_service.markdown_conversion(
                        text=extracted_text,
                        model=selected_model,
                        prompt_template=prompt_template,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        top_k=top_k,
                        min_p=min_p,
                        context_template=context_template
                    )

                    lm_elapsed = time.time() - start_time

                    if not lm_result.get("success", False):
                        lm_status.error(f"마크다운 변환 실패: {lm_result.get('error', '알 수 없는 오류')}")
                        continue

                    markdown_text = lm_result.get("markdown", "")
                    lm_status.success(f"마크다운 변환 완료 ({lm_elapsed:.2f}초)")

                    # 3. 마크다운 파일로 저장
                    base_filename = os.path.splitext(uploaded_file.name)[0]
                    output_file = os.path.join(md_output_dir, f"{base_filename}_lm.md")

                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(markdown_text)

                    file_size_kb = round(os.path.getsize(output_file) / 1024, 2)
                    final_status.success(
                        f"✅ '{uploaded_file.name}' 변환 완료! -> {output_file} " +
                        f"(크기: {file_size_kb}KB, 총 처리 시간: {docling_elapsed + lm_elapsed:.2f}초)"
                    )

                    # 세션 상태에 처리 이력 추가
                    if "processed_files" not in st.session_state:
                        st.session_state.processed_files = []

                    st.session_state.processed_files.append({
                        "filename": uploaded_file.name,
                        "output": output_file,
                        "size_kb": file_size_kb,
                        "type": "PDF→Markdown (LM Studio)",
                        "model": selected_model,
                        "status": "완료",
                        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

                except Exception as e:
                    logger.error(f"고급 변환 중 오류 발생: {e}", exc_info=True)
                    final_status.error(f"❌ '{uploaded_file.name}' 변환 실패! 오류: {str(e)}")

                    # 세션 상태에 처리 이력 추가
                    if "processed_files" not in st.session_state:
                        st.session_state.processed_files = []

                    st.session_state.processed_files.append({
                        "filename": uploaded_file.name,
                        "output": "",
                        "size_kb": 0,
                        "type": "PDF→Markdown (LM Studio)",
                        "model": selected_model,
                        "status": "실패",
                        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

                finally:
                    # 임시 파일 삭제
                    os.unlink(temp_path)

                # 진행 상황 업데이트
                progress_bar.progress((i + 1) / len(uploaded_files))

            # 모든 작업 완료
            progress_bar.progress(1.0)
            progress_text.success("모든 파일 변환 완료!")

            # 처리 완료 후 마크다운 파일 목록 표시
            st.success("모든 파일 변환이 완료되었습니다!")
            show_markdown_files(md_output_dir)

    # 기존 마크다운 파일 표시
    show_markdown_files(md_output_dir)