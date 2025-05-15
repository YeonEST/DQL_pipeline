# services/qdrant_service.py
"""
Qdrant 벡터 데이터베이스 연동 서비스

Qdrant 컬렉션 생성, 관리, 쿼리 등의 기능을 제공합니다.
"""

import os
import logging
import uuid
import time
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

logger = logging.getLogger(__name__)

class QdrantService:
    """Qdrant 서비스"""

    def __init__(
            self,
            qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333"),
            collection_name: str = "documents"
    ):
        """
        QdrantService 초기화
        
        Args:
            qdrant_url: Qdrant 서버 URL
            collection_name: 사용할 컬렉션 이름
        """
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        
        # Qdrant 클라이언트 초기화
        self.client = QdrantClient(url=qdrant_url)

    # QdrantService 클래스의 get_collections 메서드 수정
    def get_collections(self, use_cache: bool = True) -> List[str]:
        """
        모든 컬렉션 이름 가져오기

        Args:
            use_cache: 캐시 사용 여부

        Returns:
            컬렉션 이름 목록
        """
        from utils.presets import load_api_cache, save_api_cache

        # 캐시 사용인 경우 캐시 확인
        if use_cache:
            cached_data = load_api_cache("qdrant_collections", max_age_hours=1)
            if cached_data:
                logger.info("캐시된 Qdrant 컬렉션 목록 사용")
                return cached_data

        try:
            collections_data = self.client.get_collections().collections
            collections = [collection.name for collection in collections_data]

            # 데이터 캐싱 (1시간 유효)
            if use_cache:
                save_api_cache("qdrant_collections", collections, max_age_hours=1)

            return collections
        except Exception as e:
            logger.error(f"컬렉션 목록 가져오기 실패: {e}", exc_info=True)
            return []
            
    def create_collection(self, collection_name: str, vector_size: int) -> bool:
        """
        새 컬렉션 생성
        
        Args:
            collection_name: 생성할 컬렉션 이름
            vector_size: 벡터 차원 크기
            
        Returns:
            성공 여부
        """
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            logger.info(f"컬렉션 생성 성공: {collection_name}, 벡터 크기: {vector_size}")
            return True
        except Exception as e:
            logger.error(f"컬렉션 생성 실패: {e}", exc_info=True)
            return False
            
    def collection_exists(self, collection_name: str) -> bool:
        """
        컬렉션 존재 여부 확인
        
        Args:
            collection_name: 확인할 컬렉션 이름
            
        Returns:
            존재 여부
        """
        try:
            collections = self.get_collections()
            return collection_name in collections
        except Exception as e:
            logger.error(f"컬렉션 확인 실패: {e}", exc_info=True)
            return False
            
    def ensure_collection(self, collection_name: str, vector_size: int) -> bool:
        """
        컬렉션이 존재하는지 확인하고, 없으면 생성
        
        Args:
            collection_name: 컬렉션 이름
            vector_size: 벡터 차원 크기
            
        Returns:
            성공 여부
        """
        try:
            if not self.collection_exists(collection_name):
                return self.create_collection(collection_name, vector_size)
            
            # 기존 컬렉션 차원과 일치 확인
            collection_info = self.client.get_collection(collection_name=collection_name)
            existing_size = collection_info.config.params.vectors.size
            
            if existing_size != vector_size:
                logger.warning(
                    f"컬렉션 '{collection_name}'의 벡터 차원({existing_size})이 "
                    f"요청한 차원({vector_size})과 일치하지 않습니다."
                )
            
            return True
        except Exception as e:
            logger.error(f"컬렉션 확인/생성 실패: {e}", exc_info=True)
            return False
            
    def store_vectors(
            self,
            collection_name: str,
            vectors: List[List[float]],
            texts: List[str],
            metadata_list: List[Dict[str, Any]]
    ) -> bool:
        """
        벡터 데이터 저장
        
        Args:
            collection_name: 저장할 컬렉션 이름
            vectors: 벡터 데이터 목록
            texts: 원본 텍스트 목록
            metadata_list: 메타데이터 목록
            
        Returns:
            성공 여부
        """
        try:
            if len(vectors) != len(texts) or len(texts) != len(metadata_list):
                raise ValueError("벡터, 텍스트, 메타데이터 수가 일치하지 않습니다")
                
            # 고유 ID 생성을 위한 타임스탬프 (밀리초)
            timestamp = int(time.time() * 1000)
            
            # 벡터 데이터 준비
            points = [
                {
                    "id": f"{timestamp}_{i}_{uuid.uuid4().hex[:8]}", # 고유 ID 생성
                    "vector": vector,
                    "payload": {
                        "text": text,
                        **metadata
                    }
                }
                for i, (vector, text, metadata) in enumerate(zip(vectors, texts, metadata_list))
            ]
            
            # Qdrant에 벡터 저장
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            logger.info(f"{len(points)}개 벡터 저장 완료 (컬렉션: {collection_name})")
            return True
        except Exception as e:
            logger.error(f"벡터 저장 실패: {e}", exc_info=True)
            return False
            
    def search(
            self,
            collection_name: str,
            query_vector: List[float],
            top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        벡터 검색
        
        Args:
            collection_name: 검색할 컬렉션 이름
            query_vector: 쿼리 벡터
            top_k: 반환할 결과 수
            
        Returns:
            검색 결과 목록
        """
        try:
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k
            )
            
            # 결과 정리
            results = [
                {
                    "text": hit.payload.get("text", ""),
                    "score": hit.score,
                    "metadata": {k: v for k, v in hit.payload.items() if k != "text"}
                }
                for hit in search_results
            ]
            
            return results
        except Exception as e:
            logger.error(f"벡터 검색 실패: {e}", exc_info=True)
            return []
