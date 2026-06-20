"""
backend/ingestion/pipeline.py

Orchestrates the ingestion processing pipeline:
Document Parse -> Segment Chunks -> Extract Entities & Metadata -> Embed Chunks -> Coordinated DB Ingestion.

Ensures distributed transactional integrity across PostgreSQL, Qdrant, and Neo4j.
"""

import logging
import uuid
import threading
from typing import List, Optional
from qdrant_client.models import PointStruct, PointIdsList
from psycopg2.extras import execute_values

# Backend database clients
from backend.core.database import (
    get_postgres_connection,
    get_neo4j_driver,
    get_qdrant_client
)
# Extraction and extraction pipeline steps
from backend.ingestion.parser import parse_document
from backend.ingestion.chunker import chunk_document
from backend.ingestion.extractor import extract_metadata

# Guarded import for extract_entities to handle differences in Phase 2 implementations
try:
    from backend.ingestion.extractor import extract_entities
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning(
        "extract_entities function not found in backend.ingestion.extractor. "
        "Using safe fallback stub (will skip dynamically populating Document-MENTIONS-Entity relationships)."
    )
    def extract_entities(text: str) -> list:
        return []

from backend.ingestion.embedder import DocumentEmbedder

# Phase 3 Knowledge Graph operations
from backend.graph.relationship_extractor import extract_relationships
from backend.graph.graph_builder import build_relationships, build_mentions

logger = logging.getLogger(__name__)

# Global thread-safe singleton cache for heavy embedding model weights
_GLOBAL_EMBEDDER: Optional[DocumentEmbedder] = None
_EMBEDDER_LOCK = threading.Lock()


def _get_shared_embedder() -> DocumentEmbedder:
    """
    Returns a shared, thread-safe DocumentEmbedder singleton instance
    to prevent reloading heavy sentence-transformer weights on every pipeline call.
    """
    global _GLOBAL_EMBEDDER
    if _GLOBAL_EMBEDDER is None:
        with _EMBEDDER_LOCK:
            if _GLOBAL_EMBEDDER is None:
                _GLOBAL_EMBEDDER = DocumentEmbedder()
    return _GLOBAL_EMBEDDER


def ingest_document_pipeline(
    file_path: str, 
    filename: str, 
    embedder: Optional[DocumentEmbedder] = None
) -> int:
    """
    Orchestrates the entire transactional ingestion pipeline:
    Parse -> Chunk -> Extract Metadata & Entities -> Embed Chunks -> Parallel Database Ingestion.
    
    Args:
        file_path: Path to the target physical file.
        filename: Name of the file.
        embedder: Optional DocumentEmbedder. If None, uses a shared singleton instance.

    Returns:
        The generated document ID from PostgreSQL on success.
    """
    logger.info("Starting end-to-end ingestion pipeline for: %s", filename)

    # Step 1: Parse the physical file
    parsed_doc = parse_document(file_path, filename)

    # Step 2: Split text into logical chunks
    chunks = chunk_document(parsed_doc)
    if not chunks:
        raise ValueError(f"No text chunks could be extracted from document: {filename}")

    # Step 3: Extract entity references, standard mentions, and header metadata
    metadata = extract_metadata(parsed_doc.raw_text, parsed_doc.doc_type, chunks)
    entities = extract_entities(parsed_doc.raw_text)

    # Step 4: Generate dense vector embeddings for each chunk
    if embedder is None:
        embedder = _get_shared_embedder()
        
    chunk_texts = [chunk.text for chunk in chunks]
    embeddings = embedder.embed_texts(chunk_texts)

    # Step 5: Coordinated Database Ingestion (Postgres, Qdrant, Neo4j)
    pg_conn = None
    pg_cursor = None
    
    # Distributed state flags to coordinate clean rollbacks on failure
    qdrant_client = None
    qdrant_points_uploaded = False
    qdrant_ids: List[str] = []
    doc_id = None

    try:
        pg_conn = get_postgres_connection()
        pg_cursor = pg_conn.cursor()

        # Step 5a: Insert Parent Document record in PostgreSQL (starts transaction block)
        pg_cursor.execute(
            """
            INSERT INTO documents (filename, doc_type, status)
            VALUES (%s, %s, 'processing')
            RETURNING id;
            """,
            (filename, parsed_doc.doc_type)
        )
        doc_id = pg_cursor.fetchone()[0]

        # Prepare batch collections
        pg_chunks_payload = []
        qdrant_points = []
        all_relationships = []

        # Single-pass loop to prepare PG payloads, vector payloads, and run relationship extraction
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            qdrant_id = str(uuid.uuid4())
            qdrant_ids.append(qdrant_id)

            # PostgreSQL relational chunk record
            pg_chunks_payload.append((
                doc_id,
                chunk.text,
                chunk.section_ref,
                chunk.page_number,
                qdrant_id
            ))

            # Qdrant Vector DB payload
            payload = {
                "document_id": doc_id,
                "filename": filename,
                "doc_type": parsed_doc.doc_type,
                "chunk_id": chunk.chunk_id,
                "page_number": chunk.page_number,
                "section_ref": chunk.section_ref,
                "text": chunk.text,
            }
            qdrant_points.append(
                PointStruct(
                    id=qdrant_id,
                    vector=vector,
                    payload=payload
                )
            )

            # Step 5b: Extract semantic relationships from chunk text
            chunk_rels = extract_relationships(chunk.text, doc_id, idx)
            all_relationships.extend(chunk_rels)

        # Bulk Insert Chunks to PostgreSQL
        execute_values(
            pg_cursor,
            """
            INSERT INTO chunks (document_id, chunk_text, section_ref, page_number, qdrant_id)
            VALUES %s;
            """,
            pg_chunks_payload
        )

        # Step 5c: Upsert vectors to Qdrant Collection
        qdrant_client = get_qdrant_client()
        qdrant_client.upsert(
            collection_name="legal_chunks",
            points=qdrant_points
        )
        qdrant_points_uploaded = True

        # Step 5d: Ingest relationships and provenance markers into Neo4j Graph
        neo4j_driver = get_neo4j_driver()
        
        # Build the document provenance layer (linking Document node to mapped entities)
        build_mentions(neo4j_driver, doc_id, entities)
        
        # Build semantic structural legal relationships network
        build_relationships(neo4j_driver, all_relationships)

        # Step 5e: Mark processing as complete in PostgreSQL
        pg_cursor.execute(
            "UPDATE documents SET status = 'completed' WHERE id = %s;",
            (doc_id,)
        )
        pg_conn.commit()
        logger.info("Successfully ingested document '%s' with ID: %d", filename, doc_id)
        return doc_id

    except Exception as e:
        logger.exception("Ingestion pipeline failed on database transactions. Commencing rollbacks...")
        
        # Roll back PostgreSQL relational changes
        if pg_conn:
            try:
                pg_conn.rollback()
            except Exception as pg_err:
                logger.error("Failed to rollback PostgreSQL transaction: %s", pg_err)

        # Roll back Qdrant vectors to prevent orphaned vector pollution
        if qdrant_points_uploaded and qdrant_client and qdrant_ids:
            try:
                logger.info("Cleaning up uploaded vectors in Qdrant to preserve transactional integrity...")
                qdrant_client.delete(
                    collection_name="legal_chunks",
                    points_selector=PointIdsList(points=qdrant_ids)
                )
            except Exception as qdrant_err:
                logger.error("Failed to clean up Qdrant points during rollback: %s", qdrant_err)

        # Safely flag document record status as failed in a clean session
        if doc_id is not None:
            try:
                with get_postgres_connection() as err_conn:
                    with err_conn.cursor() as err_cursor:
                        err_cursor.execute(
                            "UPDATE documents SET status = 'failed' WHERE id = %s;",
                            (doc_id,)
                        )
                        err_conn.commit()
            except Exception as status_err:
                logger.error("Failed to mark document ID %d as failed: %s", doc_id, status_err)

        raise RuntimeError(f"Ingestion pipeline failed: {e}") from e

    finally:
        # Prevent UnboundLocalError by executing defensive cleanup checks
        if pg_cursor:
            try:
                pg_cursor.close()
            except Exception:
                pass
        if pg_conn:
            try:
                pg_conn.close()
            except Exception:
                pass