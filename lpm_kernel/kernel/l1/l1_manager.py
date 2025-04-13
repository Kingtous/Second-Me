from datetime import datetime
from typing import List, Optional

import numpy as np

from lpm_kernel.L1.bio import Note, Chunk, Bio
from lpm_kernel.L1.l1_generator import L1Generator
from lpm_kernel.common.repository.database_session import DatabaseSession
from lpm_kernel.file_data.document_service import document_service
from lpm_kernel.models.l1 import L1Bio
from lpm_kernel.models.l1 import (
    L1GenerationResult,
    L1Version,
    GlobalBioDTO,
    StatusBioDTO,
)
from lpm_kernel.models.status_biography import StatusBiography

from lpm_kernel.configs.logging import get_train_process_logger
logger = get_train_process_logger()


def extract_notes_from_documents(documents) -> tuple[List[Note], list]:
    """Extract Note objects and memory list from documents

    Args:
        documents: Document list containing L0 data

    Returns:
        tuple: (notes_list, memory_list)
            - notes_list: List of Note objects
            - memory_list: List of memory dictionaries for clustering
    """
    notes_list = []
    memory_list = []

    for doc in documents:
        doc_id = doc.get("id")
        doc_embedding = document_service.get_document_embedding(doc_id)
        chunks = document_service.get_document_chunks(doc_id)
        all_chunk_embeddings = document_service.get_chunk_embeddings_by_document_id(
            doc_id
        )

        if doc_embedding is None:
            logger.warning(f"Document {doc_id} missing document embedding")
            continue
        if chunks is None:
            logger.warning(f"Document {doc_id} missing chunks")
            continue
        if all_chunk_embeddings is None:
            logger.warning(f"Document {doc_id} missing chunk embeddings")
            continue

        # Ensure create_time is in string format
        create_time = doc.get("create_time")
        if isinstance(create_time, datetime):
            create_time = create_time.strftime("%Y-%m-%d %H:%M:%S")

        # Get document insight and summary
        insight_data = doc.get("insight", {})
        summary_data = doc.get("summary", {})

        if insight_data is None:
            insight_data = {}
        if summary_data is None:
            summary_data = {}

        # Build Note object
        note = Note(
            noteId=doc_id,
            content=doc.get("raw_content", ""),
            createTime=create_time,
            memoryType="TEXT",
            embedding=np.array(doc_embedding),
            chunks=[
                Chunk(
                    id=f"{chunk.id}",
                    document_id=doc_id,
                    content=chunk.content,
                    embedding=np.array(all_chunk_embeddings.get(chunk.id))
                    if all_chunk_embeddings.get(chunk.id) is not None
                    else None,
                    tags=chunk.tags if hasattr(chunk, "tags") else None,
                    topic=chunk.topic if hasattr(chunk, "topic") else None,
                )
                for chunk in chunks
                if all_chunk_embeddings.get(chunk.id) is not None
            ],
            title=insight_data.get("title", ""),
            summary=summary_data.get("summary", ""),
            insight=insight_data.get("insight", ""),
            tags=summary_data.get("keywords", []),
        )
        notes_list.append(note)
        memory_list.append({"memoryId": str(doc_id), "embedding": doc_embedding})

    return notes_list, memory_list


def generate_l1_from_l0() -> L1GenerationResult:
    """Generate L1 level knowledge representation from L0 data"""
    l1_generator = L1Generator()

    # 1. Prepare data
    documents = document_service.list_documents_with_l0()
    logger.info(f"Found {len(documents)} documents with L0 data")

    # 2. Extract notes and memories
    notes_list, memory_list = extract_notes_from_documents(documents)

    if not notes_list or not memory_list:
        logger.error("No valid documents found for processing")
        return None

    try:
        # 3. Generate L1 data
        # 3.1 Generate topics
        clusters = l1_generator.gen_topics_for_shades(
            old_cluster_list=[], old_outlier_memory_list=[], new_memory_list=memory_list
        )
        logger.info(f"Generated clusters: {bool(clusters)}")

        # 3.2 Generate chunk topics
        chunk_topics = l1_generator.generate_topics(notes_list)
        logger.info(f"Generated chunk topics: {bool(chunk_topics)}")

        # Add log in l1_manager.py
        logger.info(f"chunk_topics content: {chunk_topics}")

        # 3.3 Generate features for each cluster and merge them
        shades = []
        if clusters and "clusterList" in clusters:
            for cluster in clusters.get("clusterList", []):
                cluster_memory_ids = [
                    str(m.get("memoryId")) for m in cluster.get("memoryList", [])
                ]
                logger.info(
                    f"Processing cluster with {len(cluster_memory_ids)} memories"
                )

                cluster_notes = [
                    note for note in notes_list if str(note.id) in cluster_memory_ids
                ]
                if cluster_notes:
                    shade = l1_generator.gen_shade_for_cluster([], cluster_notes, [])
                    if shade:
                        shades.append(shade)
                        logger.info(
                            f"Generated shade for cluster: {shade.name if hasattr(shade, 'name') else 'Unknown'}"
                        )

        logger.info(f"Generated {len(shades)} shades")
        merged_shades = l1_generator.merge_shades(shades)
        logger.info(f"Merged shades success: {merged_shades.success}")
        logger.info(
            f"Number of merged shades: {len(merged_shades.merge_shade_list) if merged_shades.success else 0}"
        )

        # 3.4 Generate global biography
        bio = l1_generator.gen_global_biography(
            old_profile=Bio(
                shadesList=merged_shades.merge_shade_list
                if merged_shades.success
                else []
            ),
            cluster_list=clusters.get("clusterList", []),
        )
        logger.info(f"Generated global biography: {bio}")

        # 4. Build result object
        result = L1GenerationResult(
            bio=bio, clusters=clusters, chunk_topics=chunk_topics
        )

        logger.info("L1 generation completed successfully")
        return result

    except Exception as e:
        logger.error(f"Error in L1 generation: {str(e)}", exc_info=True)
        raise


def store_status_bio(status_bio: Bio) -> None:
    """Store status biography to database

    Args:
        status_bio (Bio): Generated status biography object
    """
    try:
        with DatabaseSession.session() as session:
            # Delete old status biography (if exists)
            session.query(StatusBiography).delete()

            # Insert new status biography
            new_bio = StatusBiography(
                content=status_bio.content_second_view,
                content_third_view=status_bio.content_third_view,
                summary=status_bio.summary_second_view,
                summary_third_view=status_bio.summary_third_view,
            )
            session.add(new_bio)
            session.commit()
    except Exception as e:
        logger.error(f"Error storing status biography: {str(e)}", exc_info=True)
        raise


def get_latest_status_bio() -> Optional[StatusBioDTO]:
    """Get the latest status biography

    Returns:
        Optional[StatusBioDTO]: Data transfer object for status biography, returns None if not found
    """
    try:
        with DatabaseSession.session() as session:
            # Get the latest status biography
            latest_bio = (
                session.query(StatusBiography)
                .order_by(StatusBiography.create_time.desc())
                .first()
            )

            if not latest_bio:
                return None

            # Convert to DTO and return
            return StatusBioDTO.from_model(latest_bio)
    except Exception as e:
        logger.error(f"Error getting status biography: {str(e)}", exc_info=True)
        return None


def get_latest_global_bio() -> Optional[GlobalBioDTO]:
    """Get the latest global biography

    Returns:
        Optional[GlobalBioDTO]: Data transfer object for global biography, returns None if not found
    """
    try:
        with DatabaseSession.session() as session:
            # Get the latest version of L1 data
            latest_version = (
                session.query(L1Version).order_by(L1Version.version.desc()).first()
            )

            if not latest_version:
                return None

            # Get bio data for this version
            bio = (
                session.query(L1Bio)
                .filter(L1Bio.version == latest_version.version)
                .first()
            )

            if not bio:
                return None

            # Convert to DTO and return
            return GlobalBioDTO.from_model(bio)
    except Exception as e:
        logger.error(f"Error getting global biography: {str(e)}", exc_info=True)
        return None


def generate_and_store_status_bio() -> Bio:
    """Generate and store status biography

    Returns:
        Bio: Generated status biography object
    """
    # Generate status biography
    status_bio = generate_status_bio()
    if status_bio:
        # Store to database
        store_status_bio(status_bio)
    return status_bio


def generate_status_bio() -> Bio:
    """Generate status biography

    Returns:
        Bio: Generated status biography
    """
    l1_generator = L1Generator()

    try:
        # 1. Get all documents and extract notes
        documents = document_service.list_documents_with_l0()
        notes_list, _ = extract_notes_from_documents(documents)

        if not notes_list:
            logger.error("No valid notes found for status bio generation")
            return None

        # 2. Generate status biography
        # Currently we only use notes, todos and chats are empty lists for now
        current_time = datetime.now().strftime("%Y-%m-%d")
        status_bio = l1_generator.gen_status_biography(
            cur_time=current_time,
            notes=notes_list,
            todos=[],  # Empty for now
            chats=[],  # Empty for now
        )

        logger.info("Status biography generated successfully")
        return status_bio

    except Exception as e:
        logger.error(f"Error generating status bio: {str(e)}", exc_info=True)
        raise
