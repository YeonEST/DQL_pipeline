# ui/help_tab.py
"""
도움말 탭

앱 사용 방법과 관련 정보를 제공하는 탭
"""

import streamlit as st

def render_help_tab():
    """도움말 탭 렌더링"""
    st.header("❓ 도움말")

    st.subheader("사용 방법")
    st.markdown("""
    ### 처리 순서

    1. **PDF → Markdown 변환**
       - **기본 변환**: docling을 사용하여 PDF 파일을 마크다운으로 변환합니다.
       - **고급 변환 (LM Studio)**: docling으로 추출한 텍스트를 LM Studio를 통해 정제된 마크다운으로 변환합니다.
       - 변환된 마크다운 파일은 `output/md/` 디렉토리에 저장됩니다.
       - 변환된 파일 목록을 확인할 수 있습니다.

    2. **Markdown 임베딩 및 업로드**
       - 변환된 마크다운 파일 중 처리할 파일을 선택합니다.
       - 선택한 파일은 청킹 후 임베딩되어 Qdrant에 업로드됩니다.
       - 원하는 컬렉션 이름을 지정할 수 있습니다.

    3. **처리 이력**
       - 모든 처리 단계(PDF→Markdown, Markdown→임베딩)에 대한 이력을 확인할 수 있습니다.
       - 이력은 CSV 형식으로 다운로드할 수 있습니다.

    4. **설정**
       - **임베딩 모델**: 다운로드 및 모델 선택을 관리합니다.
       - **Qdrant 컬렉션**: 컬렉션을 생성하고 관리합니다.
       - **LM Studio**: LM Studio 연결 및 프롬프트 템플릿을 설정합니다.
    """)

    st.subheader("환경 설정")
    st.code("""
    # .env 파일에 다음 설정을 추가할 수 있습니다:
    
    # Qdrant 설정
    QDRANT_URL=http://localhost:6333
    
    # 모델 설정
    ST_MODEL=dragonkue/snowflake-arctic-embed-l-v2.0-ko
    RERANKER_MODEL=dragonkue/bge-reranker-v2-m3-ko
    
    # LM Studio 설정
    LM_STUDIO_URL=http://localhost:1234/v1
    LM_STUDIO_KEY=lm-studio-local
    DEFAULT_LM_MODEL=
    DEFAULT_MARKDOWN_PROMPT=
    
    # LM Studio 생성 파라미터 기본값
    DEFAULT_TEMPERATURE=0.2
    DEFAULT_TOP_P=0.95
    DEFAULT_TOP_K=40
    DEFAULT_MIN_P=0.05
    DEFAULT_CONTEXT_TEMPLATE=custom  # custom, chatML, Alpaca, mistral
    
    # 디바이스 설정
    DEVICE=cuda  # 또는 cpu
    
    # 다운로드 디렉토리 설정
    DOWNLOAD_DIR=models
    EMBEDDING_MODEL_DIR=models/embedding
    RERANKER_MODEL_DIR=models/reranking
    
    # 출력 디렉토리 설정
    OUTPUT_DIR=output
    MARKDOWN_DIR=output/md
    """, language="bash")

    st.subheader("LM Studio 파라미터 설명")
    st.markdown("""
    ### 생성 파라미터

    **Temperature (온도)**
    - 텍스트 생성의 무작위성을 제어합니다.
    - 값이 낮을수록(0에 가까울수록) 더 일관된 출력을 생성합니다.
    - 값이 높을수록(1에 가까울수록) 더 다양하고 창의적인 출력을 생성합니다.
    - 추천 범위: 0.1 ~ 0.8 (마크다운 변환의 경우 0.2 ~ 0.4가 적합)

    **Top P (상위 확률)**
    - Nucleus 샘플링을 제어합니다.
    - 모델이 확률 질량의 상위 P%에 해당하는 토큰만 고려합니다.
    - 값이 낮을수록 더 안정적이고 예측 가능한 출력이 생성됩니다.
    - 추천 범위: 0.9 ~ 0.95 (정확한 마크다운 변환을 원하면 0.95 이상 권장)

    **Top K (상위 K개)**
    - 각 단계에서 고려할 가능성 있는 다음 토큰의 수를 제한합니다.
    - 값이 낮을수록 더 안정적인 출력이 생성됩니다.
    - 추천 범위: 40 ~ 60

    **Min P (최소 확률)**
    - 확률이 Top Token 확률의 Min P 미만인 토큰을 제외합니다.
    - 부적절하거나 관련 없는 토큰을 필터링하는 데 도움이 됩니다.
    - 추천 범위: 0.05 ~ 0.1

    ### 컨텍스트 템플릿

    **custom**
    - 사용자 정의 프롬프트 템플릿을 직접 편집할 수 있습니다.
    - 기본 ChatML 형식을 사용하지만 자유롭게 조정 가능합니다.

    **chatML**
    - OpenAI의 ChatML 포맷을 사용합니다.
    - `{"role": "system", "content": "..."}`, `{"role": "user", "content": "..."}`와 같은 형식

    **Alpaca**
    - Alpaca 지시 형식을 사용합니다.
    - `### 지시사항:`, `### 입력:`, `### 응답:` 형식

    **mistral**
    - Mistral AI의 템플릿 형식을 사용합니다.
    - `<s>[INST] ... [/INST] ... </s>` 형식
    """)

    st.subheader("프로젝트 구조")
    st.code("""
    DQL_pipeline/
    ├── app.py                   # 메인 앱 진입점
    ├── requirements.txt         # 필요한 패키지 정의
    ├── .env                     # 환경 변수 설정
    ├── output/                  # 출력 디렉토리
    │   └── md/                  # 마크다운 출력 저장 디렉토리
    ├── services/                # 서비스 패키지
    │   ├── pdf_service.py       # PDF 변환 서비스 (docling 사용)
    │   ├── chunking_service.py  # 마크다운 청킹 서비스
    │   ├── embedding_service.py # 임베딩 생성 서비스
    │   ├── reranking_service.py # 리랭킹 서비스
    │   ├── lm_service.py        # LM Studio 연동 서비스
    │   └── qdrant_service.py    # Qdrant 관련 서비스
    ├── ui/                      # UI 패키지
    │   ├── pdf_to_md_tab.py     # PDF → Markdown 탭
    │   ├── md_upload_tab.py     # Markdown → 임베딩 탭
    │   ├── history_tab.py       # 처리 이력 탭
    │   └── ...                  # 기타 UI 모듈
    └── utils/                   # 유틸리티 패키지
        ├── config.py            # 환경 설정 유틸리티
        ├── logging.py           # 로깅 유틸리티
        └── model_utils.py       # 모델 관련 유틸리티
    """, language="plaintext")

    st.subheader("앱 실행 방법")
    st.code("""
    # 앱 실행
    streamlit run app.py
    
    # Windows에서 실행 (배치 파일 사용)
    run.bat
    """, language="bash")