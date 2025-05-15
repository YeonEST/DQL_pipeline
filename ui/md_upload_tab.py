# ui/md_upload_tab.py
"""
마크다운을 임베딩하고 Qdrant에 업로드하는 UI 탭

마크다운 파일을 청킹하고 임베딩하여 Qdrant에 저장하는 UI를 제공합니다.
"""

import streamlit as st
import os
import pandas as pd
import time
from typing import Dict, Any, List
import logging
from pathlib import Path

from services.chunking_service import ChunkingService
from services.embedding_service import EmbeddingService
from services.qdrant_service import QdrantService

logger = logging.getLogger(__name__)

# ui/md_upload_tab.py (일부)

def render_md_upload_tab(config: Dict[str, Any]):
    """
    마크다운 임베딩 및 업로드 탭 렌더링

    Args:
        config: 앱 설정 딕셔너리
    """
    st.header("Markdown 임베딩 및 업로드")
    st.write("'output/md/' 디렉토리의 마크다운 파일을 선택하여 임베딩 후 Qdrant에 업로드합니다.")

    # 마크다운 디렉토리 정보 표시 - config에서 가져오기
    md_dir = config.get("markdown_dir", "output/md")
    st.info(f"마크다운 파일 디렉토리: {md_dir}")

    # Qdrant 컬렉션 선택 - config에서 가져오기
    collection_name = config.get("collection_name", "documents")
    st.subheader("벡터 데이터베이스 설정")

    # Qdrant URL 정보 표시
    qdrant_url = config.get("qdrant_url", "http://localhost:6333")
    st.caption(f"Qdrant URL: {qdrant_url}")

    # 컬렉션 이름 입력 필드
    collection_input = st.text_input("컬렉션 이름", value=collection_name)

    # 컬렉션 목록 가져오기 시도
    try:
        qdrant_service = QdrantService(qdrant_url=qdrant_url)
        collections = qdrant_service.get_collections()
        if collections:
            st.success(f"사용 가능한 컬렉션: {', '.join(collections)}")
            # 컬렉션이 존재하지 않으면 알림
            if collection_input and collection_input not in collections:
                st.warning(f"'{collection_input}' 컬렉션이 존재하지 않습니다. 처리 시 자동으로 생성됩니다.")
    except Exception as e:
        st.error(f"Qdrant 연결 오류: {str(e)}")

    # 마크다운 파일 목록 가져오기
    md_files = list(Path(md_dir).glob("*.md"))

    if not md_files:
        st.warning("마크다운 파일이 존재하지 않습니다. 먼저 'PDF → Markdown' 탭에서 PDF 파일을 변환해주세요.")
        return

    # 파일 정보 목록 생성
    file_info = []
    for md_file in md_files:
        file_info.append({
            "filename": md_file.name,
            "path": str(md_file),
            "size_kb": round(os.path.getsize(md_file) / 1024, 2),
            "modified": time.ctime(os.path.getmtime(md_file))
        })

    # 데이터프레임으로 변환
    df = pd.DataFrame(file_info)

    # 멀티셀렉트로 파일 선택
    selected_filenames = st.multiselect(
        "처리할 마크다운 파일 선택",
        options=df["filename"].tolist(),
        format_func=lambda x: x
    )

    if selected_filenames:
        # 선택된 파일의 정보 필터링
        selected_files = df[df["filename"].isin(selected_filenames)]

        # 선택된 파일 표시
        st.write("선택된 파일:")
        st.dataframe(selected_files, use_container_width=True)

        # 처리 옵션 설정
        st.subheader("처리 옵션")

        # 임베딩 모델 정보 표시 - config에서 가져오기
        current_model = config.get("st_model", "")
        if current_model:
            st.info(f"사용할 임베딩 모델: {current_model}")
        else:
            st.warning("임베딩 모델이 선택되지 않았습니다. 설정 탭에서 모델을 선택해주세요.")

        # 디바이스 정보 표시
        device = config.get("device", "cuda")
        st.caption(f"처리 디바이스: {device}")

        # ui/md_upload_tab.py (계속)

        with st.expander("고급 설정", expanded=False):
            batch_size = st.slider("임베딩 배치 크기", min_value=4, max_value=32, value=16, step=4)
            max_length = st.slider("최대 토큰 길이", min_value=128, max_value=1024, value=512, step=64)

        # 임베딩 및 업로드 버튼
        if st.button("선택한 파일 임베딩 및 업로드", type="primary"):
            # 서비스 초기화
            embedding_service = EmbeddingService(
                model_name=config.get("st_model", ""),
                device=config.get("device", ""),
                download_dir=config.get("embedding_model_dir", "models/embedding")
            )

            chunking_service = ChunkingService()

            qdrant_service = QdrantService(
                qdrant_url=config.get("qdrant_url", ""),
                collection_name=collection_input
            )

            # 임베딩 차원 확인
            embedding_dim = embedding_service.get_embedding_dimension()

            # 컬렉션 생성 또는 확인
            collection_status = st.empty()
            collection_status.info(f"컬렉션 '{collection_input}' 확인/생성 중...")

            if qdrant_service.ensure_collection(collection_input, embedding_dim):
                collection_status.success(f"컬렉션 '{collection_input}' 준비 완료")
            else:
                collection_status.error(f"컬렉션 '{collection_input}' 생성 실패!")
                st.error("벡터 데이터베이스 연결 또는 컬렉션 생성 중 오류가 발생했습니다.")
                return

            # 진행 상황 표시를 위한 컴포넌트
            progress_text = st.empty()
            progress_bar = st.progress(0)

            # 처리 결과 요약
            result_summary = {
                "total_files": len(selected_files),
                "processed_files": 0,
                "successful_files": 0,
                "failed_files": 0,
                "total_chunks": 0
            }

            # 각 파일 처리
            for i, (_, row) in enumerate(selected_files.iterrows()):
                file_path = row["path"]
                filename = row["filename"]

                # 진행상황 업데이트
                progress = i / len(selected_files)
                progress_bar.progress(progress)
                progress_text.info(f"({i+1}/{len(selected_files)}) '{filename}' 처리 중...")

                status_text = st.empty()
                status_text.info(f"'{filename}' 파일 읽기 및 청킹 중...")

                try:
                    # 시작 시간 기록
                    start_time = time.time()

                    # 마크다운 파일 읽기
                    with open(file_path, "r", encoding="utf-8") as f:
                        markdown_content = f.read()

                    # 마크다운 청킹
                    chunks = chunking_service.chunk_markdown(
                        markdown_content=markdown_content,
                        file_path=file_path,
                        filename=filename
                    )

                    # 텍스트와 메타데이터 추출
                    texts, metadata_list = chunking_service.get_chunk_texts_and_metadata(chunks)

                    # 청크 수 업데이트
                    total_chunks = len(texts)
                    result_summary["total_chunks"] += total_chunks

                    # 청크 정보 표시
                    status_text.info(f"'{filename}' 청킹 완료 - {total_chunks}개 청크 생성됨")

                    # 임베딩 생성 - 배치 처리
                    embed_status = st.empty()
                    embed_progress = st.progress(0)
                    embed_status.info(f"{total_chunks}개 청크 임베딩 생성 중...")

                    # 커스텀 배치 크기 사용
                    embeddings = []
                    for j in range(0, len(texts), batch_size):
                        batch_end = min(j + batch_size, len(texts))
                        batch_texts = texts[j:batch_end]

                        # 임베딩 생성
                        batch_embeddings = embedding_service.create_embeddings(
                            batch_texts,
                            max_length=max_length
                        )
                        embeddings.extend(batch_embeddings)

                        # 임베딩 진행 상황 업데이트
                        embed_progress.progress(batch_end / len(texts))
                        embed_status.info(f"임베딩 진행 중: {batch_end}/{len(texts)} 청크 완료")

                    # 임베딩 완료
                    embed_progress.progress(1.0)
                    embed_status.success(f"{len(embeddings)}개 임베딩 생성 완료!")

                    # Qdrant에 저장
                    db_status = st.empty()
                    db_status.info(f"Qdrant에 {len(embeddings)}개 벡터 저장 중...")

                    success = qdrant_service.store_vectors(
                        collection_name=collection_input,
                        vectors=embeddings,
                        texts=texts,
                        metadata_list=metadata_list
                    )

                    # 처리 시간 계산
                    elapsed_time = time.time() - start_time

                    if success:
                        result_summary["successful_files"] += 1
                        db_status.success(f"벡터 저장 완료!")
                        status_text.success(
                            f"✅ '{filename}' 처리 완료: {len(chunks)}개 청크, "
                            f"처리 시간: {elapsed_time:.2f}초"
                        )

                        # 세션 상태에 처리 이력 추가
                        if "processed_files" not in st.session_state:
                            st.session_state.processed_files = []

                        st.session_state.processed_files.append({
                            "filename": filename,
                            "collection": collection_input,
                            "chunks": len(chunks),
                            "type": "Markdown→임베딩",
                            "status": "완료",
                            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                    else:
                        result_summary["failed_files"] += 1
                        db_status.error(f"벡터 저장 실패!")
                        status_text.error(f"❌ '{filename}' 처리 실패: 벡터 저장 오류")

                        # 세션 상태에 처리 이력 추가
                        if "processed_files" not in st.session_state:
                            st.session_state.processed_files = []

                        st.session_state.processed_files.append({
                            "filename": filename,
                            "collection": collection_input,
                            "chunks": 0,
                            "type": "Markdown→임베딩",
                            "status": "실패",
                            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                        })

                except Exception as e:
                    result_summary["failed_files"] += 1
                    logger.error(f"마크다운 처리 중 오류 발생: {e}", exc_info=True)
                    status_text.error(f"❌ '{filename}' 처리 실패! 오류: {str(e)}")

                    # 세션 상태에 처리 이력 추가
                    if "processed_files" not in st.session_state:
                        st.session_state.processed_files = []

                    st.session_state.processed_files.append({
                        "filename": filename,
                        "collection": collection_input,
                        "chunks": 0,
                        "type": "Markdown→임베딩",
                        "status": "실패",
                        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

                # 처리된 파일 수 업데이트
                result_summary["processed_files"] += 1

                # 진행 상황 업데이트
                progress_bar.progress((i + 1) / len(selected_files))

            # 전체 작업 완료
            progress_bar.progress(1.0)
            progress_text.success("모든 파일 처리 완료!")

            # 결과 요약 표시
            st.success(
                f"✅ 처리 완료: {result_summary['successful_files']}/{result_summary['total_files']} 파일 성공, "
                f"총 {result_summary['total_chunks']}개 청크 처리됨"
            )

            # 실패한 파일이 있으면 경고 표시
            if result_summary["failed_files"] > 0:
                st.warning(f"⚠️ {result_summary['failed_files']}개 파일 처리 실패")

            # 결과 확인 안내
            st.info("'처리 이력' 탭에서 상세 처리 내역을 확인할 수 있습니다.")

            # 설정 업데이트 (사용된 컬렉션 이름 저장)
            if collection_input != config.get("collection_name", "documents"):
                # 세션 상태 업데이트
                if "config" in st.session_state:
                    st.session_state.config["collection_name"] = collection_input
                    st.success(f"'{collection_input}' 컬렉션이 기본 컬렉션으로 설정되었습니다.")

    else:
        st.info("처리할 마크다운 파일을 선택해주세요.")

    # 하단에 임베딩 모델 정보 표시
    st.markdown("---")
    st.caption(f"사용 중인 임베딩 모델: {os.path.basename(config.get('st_model', ''))}")