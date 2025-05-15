# services/pdf_service.py
"""
PDF 문서를 마크다운으로 변환하는 서비스

docling 라이브러리를 사용하여 PDF 문서를 마크다운 또는 텍스트로 변환합니다.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Literal

from langchain_docling import DoclingLoader
# 문제가 되는 ExportType 임포트를 확인합니다
from langchain_docling.loader import ExportType  

logger = logging.getLogger(__name__)

class PDFService:
    """PDF 처리 서비스"""

    def __init__(self, output_dir: str = "output/md", use_cpu: bool = True):
        """
        PDFService 초기화

        Args:
            output_dir: 마크다운 파일 출력 디렉토리
            use_cpu: CPU 모드 사용 (GPU 비활성화)
        """
        self.output_dir = output_dir
        self.use_cpu = use_cpu

        # CPU 모드 설정이 활성화된 경우 환경 변수 설정
        if self.use_cpu:
            os.environ["DOCLING_FORCE_CPU"] = "1"
            logger.info("Docling이 CPU 모드로 설정되었습니다.")

        # 출력 디렉토리가 없으면 생성
        os.makedirs(self.output_dir, exist_ok=True)

    def convert_pdf_to_markdown(
            self,
            pdf_path: str,
            original_filename: Optional[str] = None,
            save_to_file: bool = True,
            output_format: Literal["markdown", "text"] = "markdown"
    ) -> Dict[str, Any]:
        """
        PDF 파일을 마크다운 또는 텍스트로 변환

        Args:
            pdf_path: 처리할 PDF 파일 경로
            original_filename: 원본 파일명 (제공되지 않으면 pdf_path에서 추출)
            save_to_file: 파일로 저장할지 여부
            output_format: 출력 형식 ("markdown" 또는 "text")

        Returns:
            변환 결과 정보 (success, content, output_file 등)
        """
        try:
            logger.info(f"PDF → {output_format.capitalize()} 변환 시작: {pdf_path}")

            # 원본 파일명이 제공되지 않으면 경로에서 추출
            if original_filename is None:
                original_filename = Path(pdf_path).name

            # 파일명에서 확장자 제거
            base_filename = os.path.splitext(original_filename)[0]

            # 출력 파일 경로
            ext = ".md" if output_format == "markdown" else ".txt"
            output_path = os.path.join(self.output_dir, f"{base_filename}{ext}")

            # Docling을 사용하여 PDF를 변환
            # 수정된 코드: ExportType.TEXT 대신 문자열로 전달
            if output_format == "markdown":
                export_type = ExportType.MARKDOWN
            else:
                # 'text'가 enum에 없으면 대소문자 변형 시도 또는 다른 이름일 수 있음
                # 여기서는 ExportType.PLAIN 또는 다른 값을 시도해볼 수 있음
                # export_type = ExportType.PLAIN  # 또는 Docling에서 제공하는 다른 값
                # 아래 방식으로 문자열 직접 전달도 시도 가능
                export_type = "text"  

            content = self._load_pdf_with_docling(pdf_path, export_type)

            # 파일로 저장 (선택사항)
            if save_to_file:
                with open(output_path, "w", encoding="utf-8") as out_file:
                    out_file.write(content)
                logger.info(f"PDF → {output_format.capitalize()} 저장 완료: {output_path}")
                file_size_kb = round(os.path.getsize(output_path) / 1024, 2)
            else:
                output_path = None
                file_size_kb = None

            return {
                "success": True,
                "original_file": original_filename,
                "output_file": output_path,
                "content": content,
                "file_size_kb": file_size_kb,
                "format": output_format
            }

        except Exception as e:
            logger.error(f"PDF → {output_format.capitalize()} 변환 실패: {e}", exc_info=True)
            return {
                "success": False,
                "original_file": original_filename,
                "error": str(e),
                "format": output_format
            }

    def _load_pdf_with_docling(self, file_path: str, export_type=ExportType.MARKDOWN) -> str:
        """
        Docling을 사용하여 PDF 파일 로드 및 지정된 형식으로 변환

        Args:
            file_path: PDF 파일 경로
            export_type: 출력 형식 (MARKDOWN 또는 "text")

        Returns:
            변환된 텍스트
        """
        from langchain_docling import DoclingLoader
        from pathlib import Path
        
        logger.info(f"Docling을 사용하여 PDF 로드 중: {file_path} (출력 타입: {export_type})")

        # Docling Loader를 사용하여 PDF 로드
        loader = DoclingLoader(file_path=file_path)
        docs = loader.load()
        
        # 예제에서처럼 export_type에 따라 적절한 변환 메서드 호출
        if isinstance(docs, list) and len(docs) > 0:
            # LangChain Document 객체 리스트인 경우
            combined_text = "\n\n".join([doc.page_content for doc in docs])
            return combined_text
        else:
            # 단일 문서 또는 Docling Document 객체인 경우
            # Docling 문서 예제처럼 export_to_text 또는 export_to_markdown 메서드 호출 시도
            try:
                if hasattr(docs, 'export_to_markdown') and export_type == ExportType.MARKDOWN:
                    return docs.export_to_markdown()
                elif hasattr(docs, 'export_to_text') and export_type == "text":
                    return docs.export_to_text()
                elif hasattr(docs, 'page_content'):
                    return docs.page_content
                elif isinstance(docs, str):
                    return docs
                else:
                    return str(docs)
            except AttributeError:
                # LangChain Document 객체의 경우
                if hasattr(docs, 'page_content'):
                    return docs.page_content
                return str(docs)