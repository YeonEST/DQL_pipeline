# services/embedding_service.py
"""
텍스트 임베딩 서비스

Hugging Face Transformers를 사용하여 텍스트 임베딩 벡터를 생성하는 서비스
"""

import os
import logging
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from transformers import AutoModel, AutoTokenizer

from utils.model_utils import get_device

logger = logging.getLogger(__name__)

class EmbeddingService:
    """임베딩 서비스"""

    def __init__(
            self,
            model_name: str = os.getenv("ST_MODEL", "dragonkue/snowflake-arctic-embed-l-v2.0-ko"),
            device: Optional[str] = None,
            download_dir: Optional[str] = None
    ):
        """
        EmbeddingService 초기화

        Args:
            model_name: 사용할 임베딩 모델 이름
            device: 사용할 디바이스 (cuda 또는 cpu)
            download_dir: 모델 다운로드 디렉토리
        """
        self.model_name = model_name
        self.device = device or get_device()
        self.download_dir = download_dir or os.getenv("EMBEDDING_MODEL_DIR", "models/embedding")

        # 모델 초기화
        self._init_model()

    def _init_model(self):
        """임베딩 모델 초기화"""
        try:
            logger.info(f"임베딩 모델 초기화: {self.model_name}")

            # HuggingFace 환경 변수 설정
            os.environ["TRANSFORMERS_CACHE"] = self.download_dir

            # 모델 이름이 로컬 경로인지 확인
            is_local_path = os.path.exists(self.model_name) and os.path.isdir(self.model_name)

            # 모델 이름이 author/name 형식인지 확인
            is_model_id = "/" in self.model_name and not os.path.exists(self.model_name)

            # 1. 모델 ID가 주어진 경우 (author/name)
            if is_model_id:
                # HF Hub에서 모델 다운로드
                logger.info(f"HF Hub에서 모델 로드: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.download_dir)
                self.model = AutoModel.from_pretrained(self.model_name, cache_dir=self.download_dir)

            # 2. 로컬 경로가 주어진 경우
            elif is_local_path:
                # 로컬 경로에서 모델 로드
                logger.info(f"로컬 경로에서 모델 로드: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)

            # 3. 캐시된 모델 ID가 주어진 경우 (캐시 디렉토리 내에서 조합)
            else:
                # models-- 형식이 아니면 HF 형식으로 취급
                if "/" in self.model_name:
                    author, name = self.model_name.split("/", 1)
                    cache_dir = os.path.join(self.download_dir, f"models--{author}--{name}")
                else:
                    # 일반 디렉토리 이름이면 그대로 사용
                    cache_dir = os.path.join(self.download_dir, self.model_name)

                if os.path.exists(cache_dir):
                    logger.info(f"캐시 디렉토리에서 모델 로드: {cache_dir}")
                    self.tokenizer = AutoTokenizer.from_pretrained(cache_dir)
                    self.model = AutoModel.from_pretrained(cache_dir)
                else:
                    # 캐시 디렉토리가 없으면 모델 ID로 취급하여 다운로드
                    logger.info(f"모델을 찾을 수 없어 HF Hub에서 다운로드: {self.model_name}")
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.download_dir)
                    self.model = AutoModel.from_pretrained(self.model_name, cache_dir=self.download_dir)

            # 디바이스 설정
            self.model = self.model.to(self.device)

            # 임베딩 차원 획득
            with torch.no_grad():
                dummy_input = self.tokenizer("text", return_tensors="pt", padding=True, truncation=True)
                dummy_input = {k: v.to(self.device) for k, v in dummy_input.items()}
                outputs = self.model(**dummy_input)
                self.embedding_dim = outputs.last_hidden_state.size(-1)

            logger.info(f"임베딩 모델 로드 완료. 임베딩 차원: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"임베딩 모델 초기화 실패: {e}", exc_info=True)
            raise

    def get_embedding_dimension(self) -> int:
        """
        임베딩 차원 반환

        Returns:
            임베딩 차원 수
        """
        return self.embedding_dim

    def create_embeddings(self, texts: List[str], batch_size: int = 16, max_length: int = 512) -> List[List[float]]:
        """
        텍스트 목록에 대한 임베딩 생성

        Args:
            texts: 임베딩할 텍스트 목록
            batch_size: 배치 크기 (기본값: 16)
            max_length: 최대 토큰 길이 (기본값: 512)

        Returns:
            임베딩 벡터 리스트
        """
        try:
            logger.info(f"{len(texts)}개 텍스트에 대한 임베딩 생성 (배치 크기: {batch_size}, 최대 길이: {max_length})")

            embeddings = []

            # 배치 단위로 처리
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]

                # 토큰화
                encoded_input = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )

                # 디바이스 이동
                encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

                # 임베딩 생성
                with torch.no_grad():
                    model_output = self.model(**encoded_input)

                # CLS 토큰의 임베딩 사용 또는 평균 임베딩 사용
                # 여기서는 CLS 토큰 (첫 번째 토큰) 임베딩을 사용합니다
                sentence_embeddings = model_output.last_hidden_state[:, 0, :]

                # 정규화 (선택 사항)
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

                # 임베딩을 리스트로 변환하여 추가
                batch_embeddings = sentence_embeddings.cpu().numpy()
                embeddings.extend(batch_embeddings.tolist())

                logger.debug(f"배치 {i//batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size} 처리 완료")

            return embeddings
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}", exc_info=True)
            raise