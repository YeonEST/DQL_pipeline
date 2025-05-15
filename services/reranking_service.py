# services/reranking_service.py
"""
검색 결과 리랭킹 서비스

AutoModelForSequenceClassification을 사용하여 검색 결과를 리랭킹하는 서비스를 제공합니다.
"""

import os
import logging
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils.model_utils import get_device

logger = logging.getLogger(__name__)

class RerankingService:
    """리랭킹 서비스"""

    def __init__(
            self,
            model_name: str = os.getenv("RERANKER_MODEL", "dragonkue/bge-reranker-v2-m3-ko"),
            device: Optional[str] = None,
            download_dir: Optional[str] = None
    ):
        """
        RerankingService 초기화

        Args:
            model_name: 사용할 리랭킹 모델 이름
            device: 사용할 디바이스 (cuda 또는 cpu)
            download_dir: 모델 다운로드 디렉토리
        """
        self.model_name = model_name
        self.device = device or get_device()
        self.download_dir = download_dir or os.getenv("RERANKER_MODEL_DIR", "models/reranking")

        # 모델 초기화
        if self.model_name:
            self._init_model()
        else:
            self.model = None
            self.tokenizer = None
            logger.warning("리랭커 모델이 지정되지 않았습니다. 리랭킹이 비활성화됩니다.")

    def _init_model(self):
        """리랭커 모델 초기화"""
        try:
            logger.info(f"리랭커 모델 초기화: {self.model_name}")

            # 환경 변수 설정
            os.environ["TRANSFORMERS_CACHE"] = self.download_dir

            # 토크나이저 및 모델 초기화
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.download_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, cache_dir=self.download_dir)

            # 디바이스 설정
            self.model = self.model.to(self.device)
            self.model.eval()  # 평가 모드로 설정

            logger.info("리랭커 모델 로드 완료")
        except Exception as e:
            logger.error(f"리랭커 모델 초기화 실패: {e}", exc_info=True)
            self.model = None
            self.tokenizer = None

    def rerank(
            self,
            query: str,
            results: List[Dict[str, Any]],
            top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        검색 결과 리랭킹

        Args:
            query: 원본 쿼리
            results: 검색 결과 목록
            top_k: 반환할 결과 수

        Returns:
            리랭킹된 결과 목록
        """
        # 모델이 없으면 원본 결과 반환
        if not self.model or not self.tokenizer:
            logger.warning("리랭커 모델이 로드되지 않았습니다. 원본 결과를 반환합니다.")
            return results[:top_k]

        try:
            logger.info(f"{len(results)}개 검색 결과 리랭킹 시작")

            # 결과에서 텍스트 추출
            texts = [result["text"] for result in results]

            # 배치 크기 설정
            batch_size = 8
            all_scores = []

            # 배치 단위로 처리
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_scores = self._score_pairs(query, batch_texts)
                all_scores.extend(batch_scores)

            # 결과 업데이트
            for i, score in enumerate(all_scores):
                results[i]["score"] = float(score)

            # 점수 기준 내림차순 정렬
            results = sorted(results, key=lambda x: x["score"], reverse=True)

            # top_k 반환
            return results[:top_k]

        except Exception as e:
            logger.error(f"검색 결과 리랭킹 실패: {e}", exc_info=True)
            return results[:top_k]

    def _score_pairs(self, query: str, passages: List[str]) -> List[float]:
        """
        쿼리와 문서 쌍의 관련성 점수 계산

        Args:
            query: 쿼리 문자열
            passages: 문서/구절 리스트

        Returns:
            관련성 점수 리스트
        """
        scores = []

        for passage in passages:
            # 토큰화 - 쿼리와 문서를 함께 입력
            inputs = self.tokenizer(
                query,
                passage,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # 디바이스 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 점수 계산
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

                # 모델에 따라 점수 추출 방식 달라질 수 있음
                # 분류 모델인 경우 (일반적으로 positive 클래스의 logit 또는 확률 사용)
                if logits.shape[1] > 1:  # 다중 클래스인 경우
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    # 보통 1번 인덱스가 긍정적인 관련성을 의미함 (모델에 따라 다를 수 있음)
                    score = probs[0, 1].item()
                else:  # 단일 값인 경우 (회귀 출력)
                    score = logits[0, 0].item()

                scores.append(score)

        # 점수 정규화 (선택사항)
        if len(scores) > 0:
            min_score = min(scores)
            max_score = max(scores)
            if max_score > min_score:
                scores = [(score - min_score) / (max_score - min_score) for score in scores]

        return scores