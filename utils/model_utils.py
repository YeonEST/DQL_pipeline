# utils/model_utils.py
"""
모델 관련 유틸리티 함수

임베딩 모델, 리랭킹 모델 등의 초기화 및 관리를 위한 유틸리티 함수를 제공합니다.
"""

import os
import logging
import torch
import json
import shutil
from typing import List, Optional, Union, Dict, Any
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

# 모델 관련 기본 경로 설정
DEFAULT_EMBEDDING_MODEL = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
DEFAULT_RERANKER_MODEL = "dragonkue/bge-reranker-v2-m3-ko"
DEFAULT_EMBEDDING_DIR = "models/embedding"
DEFAULT_RERANKER_DIR = "models/reranking"

def get_device() -> str:
    """
    사용 가능한 최적의 디바이스 반환

    Returns:
        디바이스 문자열 ("cuda" 또는 "cpu")
    """
    return "cuda" if torch.cuda.is_available() else "cpu"

def init_embedding_model(
        model_name: str = None,
        device: str = None,
        download_dir: str = None
) -> torch.nn.Module:
    """
    임베딩 모델 초기화

    Args:
        model_name: 모델 이름 또는 경로
        device: 사용할 디바이스
        download_dir: 모델 다운로드 디렉토리

    Returns:
        초기화된 Transformers 모델
    """
    # 기본값 설정
    model_name = model_name or os.getenv("ST_MODEL", DEFAULT_EMBEDDING_MODEL)
    device = device or get_device()
    download_dir = download_dir or os.getenv("EMBEDDING_MODEL_DIR", DEFAULT_EMBEDDING_DIR)

    logger.info(f"임베딩 모델 로드 중: {model_name}")

    # 모델 다운로드 디렉토리 생성
    os.makedirs(download_dir, exist_ok=True)
    logger.info(f"모델 다운로드 경로: {download_dir}")

    # 환경 변수 설정
    os.environ["TRANSFORMERS_CACHE"] = download_dir

    # 모델 로드 시도
    try:
        model = AutoModel.from_pretrained(model_name, cache_dir=download_dir)
        model = model.to(device)
        logger.info(f"모델 로드 성공: {model_name}")
        return model
    except Exception as e:
        logger.error(f"모델 로드 중 오류 발생: {e}")
        raise

def init_reranker_model(
        model_name: str = None,
        device: str = None,
        download_dir: str = None
) -> tuple:
    """
    리랭커 모델 초기화

    Args:
        model_name: 모델 이름 또는 경로
        device: 사용할 디바이스
        download_dir: 모델 다운로드 디렉토리

    Returns:
        (tokenizer, model) 튜플 또는 (None, None)
    """
    # 기본값 설정
    model_name = model_name or os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
    device = device or get_device()
    download_dir = download_dir or os.getenv("RERANKER_MODEL_DIR", DEFAULT_RERANKER_DIR)

    # 모델명이 없으면 None 반환
    if not model_name:
        logger.warning("리랭커 모델이 지정되지 않았습니다.")
        return None, None

    logger.info(f"리랭커 모델 로드 중: {model_name}")

    # 모델 다운로드 디렉토리 생성
    os.makedirs(download_dir, exist_ok=True)
    os.environ["TRANSFORMERS_CACHE"] = download_dir

    # 모델 로드 시도
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=download_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=download_dir)
        model = model.to(device)
        model.eval()

        logger.info(f"리랭커 모델 로드 성공: {model_name}")
        return tokenizer, model
    except Exception as e:
        logger.error(f"리랭커 모델 로드 중 오류 발생: {e}")
        raise

def get_model_config(model_name_or_path: str) -> Dict[str, Any]:
    """
    Transformers 모델 구성 가져오기

    Args:
        model_name_or_path: 모델 이름 또는 경로

    Returns:
        모델 구성 정보
    """
    try:
        # 모델 구성 로드
        config = AutoConfig.from_pretrained(model_name_or_path)
        return config.to_dict()
    except Exception as e:
        logger.error(f"모델 구성 로드 중 오류 발생: {e}", exc_info=True)
        return {"hidden_size": 768}  # 기본값 반환

def get_embedding_dimension(model_name_or_path: str) -> int:
    """
    임베딩 모델의 차원 크기를 반환합니다.

    Args:
        model_name_or_path: 모델 이름 또는 경로

    Returns:
        임베딩 차원 크기 (기본값: 768)
    """
    try:
        # 모델 구성 로드
        config = get_model_config(model_name_or_path)

        # 차원 정보 찾기
        if 'hidden_size' in config:
            dim = config['hidden_size']
            logger.info(f"Config에서 임베딩 차원 추출: {dim}")
            return dim
        elif 'dim' in config:
            dim = config['dim']
            logger.info(f"Config에서 임베딩 차원 추출: {dim}")
            return dim
        else:
            logger.warning(f"모델 구성에서 차원 정보를 찾을 수 없습니다. 기본값 768 사용")
            return 768
    except Exception as e:
        logger.error(f"임베딩 차원 확인 중 오류 발생: {e}", exc_info=True)
        return 768  # 기본값 반환

def get_model_info(model_name_or_path: str) -> Dict[str, Any]:
    """
    모델에 대한 정보를 반환합니다.

    Args:
        model_name_or_path: 모델 이름 또는 경로

    Returns:
        모델 정보를 담은 딕셔너리
    """
    try:
        # 임베딩 차원 가져오기
        dimension = get_embedding_dimension(model_name_or_path)

        # 모델 타입 확인
        model_type = "Transformer"

        # 모델 이름 추출
        if '/' in model_name_or_path:
            model_name = model_name_or_path.split('/')[-1]
        else:
            model_name = os.path.basename(model_name_or_path)

        # 모델 정보 반환
        return {
            "dimension": dimension,
            "model_type": model_type,
            "model_name": model_name,
            "is_loaded": True
        }
    except Exception as e:
        logger.error(f"모델 정보 확인 중 오류 발생: {e}", exc_info=True)
        return {
            "dimension": 768,  # 기본값
            "model_type": "Unknown",
            "is_loaded": False,
            "error": str(e)
        }

def get_downloaded_models(model_type: str = "embedding") -> List[str]:
    """
    다운로드된 모델 목록 반환

    Args:
        model_type: "embedding" 또는 "reranking"

    Returns:
        다운로드된 모델명 목록
    """
    # 모델 타입에 따른 디렉토리 설정
    if model_type == "embedding":
        model_dir = os.getenv("EMBEDDING_MODEL_DIR", DEFAULT_EMBEDDING_DIR)
    elif model_type == "reranking":
        model_dir = os.getenv("RERANKER_MODEL_DIR", DEFAULT_RERANKER_DIR)
    else:
        raise ValueError(f"지원되지 않는 모델 타입: {model_type}")

    # 디렉토리가 없으면 생성 후 빈 리스트 반환
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        return []

    model_ids = []

    # 1. HuggingFace Hub 캐시 디렉토리 확인 (models--*)
    hf_cache_path = Path(model_dir)
    models_prefix = "models--"

    # models-- 로 시작하는 디렉토리 탐색
    for model_path in hf_cache_path.glob(f"{models_prefix}*"):
        if model_path.is_dir():
            try:
                # 모델 ID 추출 (models--author--name 형식)
                model_name_parts = model_path.name.split("--")
                if len(model_name_parts) >= 3:
                    author = model_name_parts[1]
                    name = "--".join(model_name_parts[2:])
                    model_id = f"{author}/{name}"
                    model_ids.append(model_id)
                    logger.info(f"HF 캐시에서 모델 발견: {model_id}")
            except Exception as e:
                logger.warning(f"모델 ID 추출 중 오류: {e}")

    # 2. 개별 모델 디렉토리도 확인 (캐시 디렉토리가 아닌 직접 저장된 모델)
    for item in os.listdir(model_dir):
        item_path = os.path.join(model_dir, item)

        # 폴더이면서 모델 파일이 있는지 확인
        if os.path.isdir(item_path) and not item.startswith(models_prefix):
            # config.json 또는 pytorch_model.bin 파일 확인
            if (os.path.exists(os.path.join(item_path, "config.json")) or
                    os.path.exists(os.path.join(item_path, "pytorch_model.bin"))):
                model_ids.append(item)
                logger.info(f"직접 저장된 모델 발견: {item}")

    logger.info(f"발견된 모델 ID: {model_ids}")
    return model_ids

def download_model(model_name: str, model_type: str = "embedding") -> bool:
    """
    Hugging Face 모델 다운로드 (지정된 경로에 저장)

    Args:
        model_name: 다운로드할 모델명 (author/name 형식)
        model_type: "embedding" 또는 "reranking"

    Returns:
        성공 여부 (True/False)
    """
    try:
        if model_type == "embedding":
            # 임베딩 모델 다운로드
            download_dir = os.getenv("EMBEDDING_MODEL_DIR", DEFAULT_EMBEDDING_DIR)
            os.makedirs(download_dir, exist_ok=True)

            # HuggingFace 캐시 폴더 설정
            os.environ["TRANSFORMERS_CACHE"] = download_dir

            logger.info(f"모델 '{model_name}' 다운로드 시작, 캐시 경로: {download_dir}")

            try:
                # 모델 토크나이저와 모델 다운로드
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=download_dir)
                model = AutoModel.from_pretrained(model_name, cache_dir=download_dir)

                # 간단한 차원 확인 - 유효한 모델인지 검증
                config = model.config
                dimension = getattr(config, "hidden_size", 768)
                logger.info(f"모델 임베딩 차원: {dimension}")

                # 저장된 캐시 경로 표시
                if "/" in model_name:
                    author, name = model_name.split("/", 1)
                    cache_dir = os.path.join(download_dir, f"models--{author}--{name}")
                    cache_dir = cache_dir.replace("--", "-").replace("/", "--").replace(":", "--")
                    logger.info(f"모델이 캐시 디렉토리 '{cache_dir}'에 저장되었습니다.")
                else:
                    logger.info(f"모델이 '{download_dir}' 디렉토리에 저장되었습니다.")

                return True
            except Exception as e:
                logger.error(f"Transformer 모델 다운로드 실패: {e}", exc_info=True)
                return False

        elif model_type == "reranking":
            # 리랭킹 모델 다운로드
            download_dir = os.getenv("RERANKER_MODEL_DIR", DEFAULT_RERANKER_DIR)
            os.makedirs(download_dir, exist_ok=True)

            # HuggingFace 캐시 폴더 설정
            os.environ["TRANSFORMERS_CACHE"] = download_dir

            logger.info(f"리랭커 모델 '{model_name}' 다운로드 시작, 캐시 경로: {download_dir}")

            # 모델 다운로드 및 저장
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=download_dir)
                model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=download_dir)

                # 저장된 캐시 경로 표시
                if "/" in model_name:
                    author, name = model_name.split("/", 1)
                    cache_dir = os.path.join(download_dir, f"models--{author}--{name}")
                    cache_dir = cache_dir.replace("--", "-").replace("/", "--").replace(":", "--")
                    logger.info(f"리랭커 모델이 캐시 디렉토리 '{cache_dir}'에 저장되었습니다.")
                else:
                    logger.info(f"리랭커 모델이 '{download_dir}' 디렉토리에 저장되었습니다.")

                return True
            except Exception as e:
                logger.error(f"리랭커 모델 다운로드 실패: {e}", exc_info=True)
                return False

        else:
            logger.error(f"지원되지 않는 모델 타입: {model_type}")
            return False

    except Exception as e:
        logger.error(f"{model_type} 모델 '{model_name}' 다운로드 중 오류 발생: {e}", exc_info=True)
        return False