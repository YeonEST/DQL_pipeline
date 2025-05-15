# 문서 처리 파이프라인

PDF 문서를 마크다운으로 변환한 후, 청킹하여 임베딩하고 Qdrant 벡터 데이터베이스에 저장하는 파이프라인 애플리케이션입니다.

## 주요 기능

1. **PDF → Markdown 변환**
    - **기본 변환**: [langchain-docling](https://github.com/langchain-ai/langchain-docling)을 사용하여 PDF 문서를 마크다운으로 변환합니다.
    - **고급 변환**: LM Studio API를 통해 PDF에서 추출한 텍스트를 더 정제된 마크다운으로 변환합니다.
    - 변환된 마크다운 파일은 `output/md/` 디렉토리에 저장됩니다.

2. **마크다운 청킹 및 임베딩**
    - 마크다운 헤더(#, ##, ###)를 기준으로 문서를 청킹합니다.
    - Sentence Transformer 모델을 사용하여 텍스트 임베딩을 생성합니다.
    - 임베딩된 청크를 Qdrant 벡터 데이터베이스에 저장합니다.

3. **처리 이력 관리**
    - 모든 처리 과정의 이력을 기록하고 관리합니다.
    - 이력은 CSV 형식으로 내보낼 수 있습니다.

4. **LM Studio 연동**
    - [LM Studio](https://lmstudio.ai/)를 통해 로컬에서 실행 중인 LLM 모델을 활용합니다.
    - PDF에서 추출한 텍스트를 보다 구조화된 마크다운으로 변환합니다.
    - 사용자 정의 프롬프트 템플릿을 활용한 맞춤형 변환이 가능합니다.

## 모듈화된 아키텍처

- **서비스 계층**: 각 기능을 독립적인 서비스로 분리하여 모듈화
    - `PDFService`: PDF → 마크다운/텍스트 변환 담당
    - `ChunkingService`: 마크다운 문서 청킹 담당
    - `EmbeddingService`: 텍스트 임베딩 생성 담당
    - `QdrantService`: 벡터 데이터베이스 연동 담당
    - `RerankingService`: 검색 결과 리랭킹 담당 (선택 사항)
    - `LMStudioService`: LM Studio API 연동 및 마크다운 정제 담당

- **UI 계층**: 각 기능별 탭으로 구분된 사용자 인터페이스
    - PDF → 마크다운 변환 탭 (기본/고급 변환 지원)
    - 마크다운 → 임베딩 탭
    - 처리 이력 탭
    - 설정 탭 (임베딩 모델, Qdrant, LM Studio 설정)
    - 도움말 탭

## 시스템 요구사항

- Python 3.9 이상
- [Qdrant](https://qdrant.tech/) 벡터 데이터베이스 (로컬 또는 원격)
- 임베딩 모델 실행을 위한 CUDA 지원 (GPU 권장)
- [LM Studio](https://lmstudio.ai/) - 고급 마크다운 변환에 사용 (선택 사항)

## 설치 방법

1. 저장소 복제
   ```bash
   git clone https://github.com/yourusername/document-processing-pipeline.git
   cd document-processing-pipeline
   ```

2. 가상 환경 생성 및 활성화
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Linux/Mac
   python -m venv .venv
   source .venv/bin/activate
   ```

3. 필요한 패키지 설치
   ```bash
   pip install -r requirements.txt
   ```

4. 환경 설정
   ```bash
   # .env.example을 .env로 복사하고 필요한 설정 변경
   cp .env.example .env
   ```

## 실행 방법

### Windows
```bash
# 배치 파일 실행
run.bat

# 또는 직접 실행
streamlit run app.py
```

### Linux/Mac
```bash
# 쉘 스크립트 실행 (실행 권한 필요)
chmod +x run.sh
./run.sh

# 또는 직접 실행
streamlit run app.py
```

## 사용 방법

1. **PDF → Markdown 변환**
    - **기본 변환**:
        - "PDF → Markdown" 탭의 "기본 변환" 서브탭에서 PDF 파일을 업로드합니다.
        - "PDF 파일 변환하기" 버튼을 클릭하여 마크다운으로 변환합니다.
        - 변환된 파일은 탭 하단에 표시되며, `output/md/` 디렉토리에 저장됩니다.
    - **고급 변환 (LM Studio)**:
        - "PDF → Markdown" 탭의 "고급 변환 (LM Studio)" 서브탭으로 이동합니다.
        - LM Studio URL을 확인하고 사용할 모델을 선택합니다.
        - 필요시 프롬프트 템플릿을 조정할 수 있습니다.
        - PDF 파일을 업로드한 후 "LM Studio를 사용하여 변환하기" 버튼을 클릭합니다.
        - 변환된 파일은 `output/md/` 디렉토리에 저장됩니다.

2. **Markdown → 임베딩**
    - "Markdown → 임베딩" 탭에서 처리할 마크다운 파일을 선택합니다.
    - 컬렉션 이름을 입력합니다.
    - "선택한 파일 임베딩 및 업로드" 버튼을 클릭합니다.
    - 각 파일은 청킹되어 임베딩된 후 Qdrant에 저장됩니다.

3. **처리 이력 확인**
    - "처리 이력" 탭에서 모든 처리 과정의 이력을 확인할 수 있습니다.
    - "CSV로 다운로드" 버튼을 클릭하여 이력을 CSV 파일로 내보낼 수 있습니다.

4. **설정 관리**
    - "설정" 탭에서 세 가지 카테고리의 설정을 관리할 수 있습니다:
        - **임베딩 모델**: 모델 다운로드 및 선택
        - **Qdrant 컬렉션**: 연결 설정 및 컬렉션 관리
        - **LM Studio**: API 연결 및 기본 프롬프트 템플릿 설정

## LM Studio 파라미터

### 생성 파라미터

- **Temperature (온도)** (0.0 ~ 1.0): 텍스트 생성의 무작위성 제어 (낮을수록 일관된 출력)
- **Top P (상위 확률)** (0.0 ~ 1.0): Nucleus 샘플링 제어 (확률 질량의 상위 P%만 고려)
- **Top K (상위 K개)** (1 ~ 100): 고려할 다음 토큰 후보 수 제한
- **Min P (최소 확률)** (0.0 ~ 0.5): 확률이 최상위 토큰 확률의 Min P 미만인 토큰 제외

### 컨텍스트 템플릿

- **custom**: 사용자 정의 프롬프트 템플릿 (자유롭게 편집 가능)
- **chatML**: OpenAI의 ChatML 포맷 (`{"role": "system", "content": "..."}` 형식)
- **Alpaca**: Alpaca 지시 형식 (`### 지시사항:`, `### 입력:`, `### 응답:` 형식)
- **mistral**: Mistral AI의 템플릿 형식 (`<s>[INST] ... [/INST] ... </s>` 형식)

## 참고 사항

- Qdrant는 별도로 설치해야 합니다. [Qdrant 설치 가이드](https://qdrant.tech/documentation/install/)를 참조하세요.
- 임베딩 모델은 처음 사용 시 자동으로 다운로드됩니다.
- 대용량 PDF 파일 처리 시 충분한 메모리가 필요합니다.
- LM Studio는 API 서버로 실행해야 합니다. [LM Studio 공식 문서](https://lmstudio.ai/)를 참조하세요.

## 라이선스

MIT 라이선스