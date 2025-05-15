# services/chunking_service.py
"""
마크다운 문서를 청킹하는 서비스

마크다운 문서를 헤더를 기준으로 청킹하는 서비스를 제공합니다.
"""

import logging
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

logger = logging.getLogger(__name__)

class ChunkingService:
    """마크다운 청킹 서비스"""

    def __init__(self):
        """청킹 서비스 초기화"""
        # 마크다운 헤더 스플리터 초기화
        self.headers_to_split_on = [
            ("#", "header1"),
            ("##", "header2"),
            ("###", "header3"),
        ]
        
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on
        )

    def chunk_markdown(self, markdown_content: str, file_path: str, filename: str) -> List[Document]:
        """
        마크다운 문서를 헤더 기반으로 청킹
        
        Args:
            markdown_content: 마크다운 내용
            file_path: 마크다운 파일 경로
            filename: 원본 파일명
            
        Returns:
            청킹된 문서 리스트
        """
        logger.info(f"마크다운 문서 청킹 시작: {filename}")
        
        try:
            # 마크다운 헤더 기반 청킹
            chunks = self.markdown_splitter.split_text(markdown_content)
            
            # 각 청크에 메타데이터 추가
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "file_name": filename,
                    "file_path": file_path,
                    "chunk_id": i
                })
            
            logger.info(f"{filename} 문서의 청크 {len(chunks)}개 생성 완료")
            return chunks
            
        except Exception as e:
            logger.error(f"마크다운 청킹 중 오류 발생: {e}", exc_info=True)
            raise
            
    def get_chunk_texts_and_metadata(self, chunks: List[Document]) -> tuple:
        """
        청크에서 텍스트와 메타데이터 추출
        
        Args:
            chunks: 청크 리스트
            
        Returns:
            (텍스트 리스트, 메타데이터 리스트) 튜플
        """
        texts = []
        metadata_list = []
        
        for chunk in chunks:
            texts.append(chunk.page_content)
            metadata_list.append(chunk.metadata)
            
        return texts, metadata_list
