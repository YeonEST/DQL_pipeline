# utils/config.py
"""
환경 설정 유틸리티

환경 변수 로드 및 설정 관련 함수를 제공합니다.
"""

import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def load_env_config() -> Dict[str, Any]:
    """
    환경 변수 설정 로드
    
    .env 파일에서 환경 변수를 로드하고, 기본 설정 값을 반환합니다.
    
    Returns:
        환경 설정 딕셔너리
    """
    # .env 파일에서 환경 변수 로드
    load_dotenv()

    # 기본 설정 값
    config = {
        # LM Studio 설정
        "lm_studio_url": os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1"),
        "lm_studio_key": os.getenv("LM_STUDIO_KEY", "lm-studio-local"),
        "default_lm_model": os.getenv("DEFAULT_LM_MODEL", ""),
        "default_markdown_prompt": os.getenv("DEFAULT_MARKDOWN_PROMPT", """끝에 있는 입력 내용에 대해 다음 마크다운 노트 형식으로 정리해줘:
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

변환된 마크다운:"""),
        # LM Studio 생성 파라미터 기본값
        "default_temperature": float(os.getenv("DEFAULT_TEMPERATURE", "0.2")),
        "default_top_p": float(os.getenv("DEFAULT_TOP_P", "0.95")),
        "default_top_k": int(os.getenv("DEFAULT_TOP_K", "40")),
        "default_min_p": float(os.getenv("DEFAULT_MIN_P", "0.05")),
        "default_context_template": os.getenv("DEFAULT_CONTEXT_TEMPLATE", "custom"),

        # Qdrant 설정
        "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),

        # 모델 설정
        "st_model": os.getenv("ST_MODEL", "dragonkue/snowflake-arctic-embed-l-v2.0-ko"),
        "reranker_model": os.getenv("RERANKER_MODEL", "dragonkue/bge-reranker-v2-m3-ko"),

        # 디바이스 설정
        "device": os.getenv("DEVICE", "cuda"),

        # 다운로드 디렉토리 설정
        "download_dir": os.getenv("DOWNLOAD_DIR", "models"),
        "embedding_model_dir": os.getenv("EMBEDDING_MODEL_DIR", "models/embedding"),
        "reranker_model_dir": os.getenv("RERANKER_MODEL_DIR", "models/reranking"),

        # 출력 디렉토리 설정
        "output_dir": os.getenv("OUTPUT_DIR", "output"),
        "markdown_dir": os.getenv("MARKDOWN_DIR", "output/md"),
    }

    # 필요한 디렉토리 생성
    os.makedirs(config["embedding_model_dir"], exist_ok=True)
    os.makedirs(config["reranker_model_dir"], exist_ok=True)
    os.makedirs(config["markdown_dir"], exist_ok=True)

    logger.info("환경 변수 로드 완료")
    return config