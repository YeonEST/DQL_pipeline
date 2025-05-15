# ui/process_tab.py
"""
레거시 처리 탭 호환성 모듈

이전 버전과의 호환성을 위한 프로세스 함수를 제공합니다.
"""

import logging
from typing import Optional

from services.pdf_service import PDFService
from services.chunking_service import ChunkingService
from services.embedding_service import EmbeddingService
from services.qdrant_service import QdrantService

logger = logging.getLogger(__name__)

def process_file(
        file_path: str,
        original_filename: str,
        st_model: str,
        reranker_model: Optional[str],
        device: str,
        download_dir: str,
        qdrant_url: str,
        collection_name: str,
        use_cpu_for_pdf: bool = True
) -> bool:
    """
    파일 처리 파이프라인 실행 (레거시 호환성 함수)

    Args:
        file_path: 처리할 PDF 파일 경로
        original_filename: 원본 파일명
        st_model: 임베딩 모델 이름
        reranker_model: 리랭커 모델 이름 (선택사항)
        device: 사용할 디바이스 (cuda 또는 cpu)
        download_dir: 모델 다운로드 디렉토리
        qdrant_url: Qdrant URL
        collection_name: 컬렉션 이름
        use_cpu_for_pdf: PDF 처리에 CPU 모드 사용 여부

    Returns:
        성공 여부
    """
    try:
        logger.info(f"레거시 처리 모드로 파일 처리: {file_path}")

        # 1. PDF 서비스로 마크다운 변환 (CPU 모드 적용)
        pdf_service = PDFService(output_dir="output/md", use_cpu=use_cpu_for_pdf)
        result = pdf_service.convert_pdf_to_markdown(file_path, original_filename)

        if not result.get("success", False):
            logger.error(f"PDF 변환 실패: {result.get('error', '알 수 없는 오류')}")
            return False

        # 마크다운 파일 경로
        md_path = result.get("output_file")

        # 마크다운 파일 내용 읽기
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # 2. 임베딩 서비스 초기화
        embedding_service = EmbeddingService(
            model_name=st_model,
            device=device,
            download_dir=download_dir
        )

        # 3. 청킹 서비스 초기화
        chunking_service = ChunkingService()

        # 4. Qdrant 서비스 초기화
        qdrant_service = QdrantService(
            qdrant_url=qdrant_url,
            collection_name=collection_name
        )

        # 임베딩 차원 확인
        embedding_dim = embedding_service.get_embedding_dimension()

        # 컬렉션 생성 또는 확인
        if not qdrant_service.ensure_collection(collection_name, embedding_dim):
            logger.error(f"컬렉션 확인/생성 실패: {collection_name}")
            return False

        # 5. 마크다운 청킹
        chunks = chunking_service.chunk_markdown(
            markdown_content=md_content,
            file_path=md_path,
            filename=original_filename
        )

        # 6. 텍스트와 메타데이터 추출
        texts, metadata_list = chunking_service.get_chunk_texts_and_metadata(chunks)

        # 7. 임베딩 생성
        vectors = embedding_service.create_embeddings(texts)

        # 8. Qdrant에 저장
        success = qdrant_service.store_vectors(
            collection_name=collection_name,
            vectors=vectors,
            texts=texts,
            metadata_list=metadata_list
        )

        if not success:
            logger.error("Qdrant에 임베딩 저장 실패")
            return False

        logger.info(f"파일 처리 완료: {original_filename}, 청크 수: {len(chunks)}")
        return True

    except Exception as e:
        logger.error(f"파일 '{file_path}' 처리 중 오류 발생: {e}", exc_info=True)
        return False