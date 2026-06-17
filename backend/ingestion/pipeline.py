import logging
import uuid
import threading
from typing import List, Optional
from qdrant_client.models import PointStruct, PointIdsList
from psycopg2.extras import execute_values

# Backend system imports
from backend.core.database import (
    get_postgres_connection,
    get_neo4j_driver,
    get_qdrant_client
)
from backend.ingestion.parser import parse_document
from backend.ingestion.chunker import chunk_document
from backend.ingestion.extractor import extract_metadata
from backend.ingestion.embedder import DocumentEmbedder

logger = logging.getLogger(__name__)

# Global fallback thread-safe cached embedder instance to prevent GPU OOM
_GLOBAL_EMBEDDER: Optional[DocumentEmbedder] = None
_EMBEDDER_LOCK = threading.Lock()


def _get_shared_embedder() -> DocumentEmbedder:
    """
    Returns a shared thread-safe DocumentEmbedder singleton instance
    to prevent reloading heavy model weights on every invocation.
    """
    global _GLOBAL_EMBEDDER
    if _GLOBAL_EMBEDDER is None:
        with _EMBEDDER_LOCK:
            if _GLOBAL_EMBEDDER is None:
                _GLOBAL_EMBEDDER = DocumentEmbedder()
    return _GLOBAL_EMBEDDER


def _ingest_to_neo4j(tx, doc_id: int, filename: str, doc_type: str, section_refs: List[str]):
    """
    Executes parameterized Cypher queries to merge entities into Neo4j.
    Safely groups reference labels to avoid Cypher injection and leverages batching.
    """
    doc_type_clean = doc_type.lower()
    
    # Strict label whitelist mapping to eliminate Cypher injection vulnerability
    label_whitelist = {
        "judgment": "Judgment",
        "circular": "Circular",
        "notification": "Notification",
        "amendment": "Amendment"
    }
    base_label = label_whitelist.get(doc_type_clean, "Document")

    # MERGE the Document node safely (label string interpolation uses whitelisted strings only)
    merge_doc_query = f"""
    MERGE (d:{base_label} {{id: $doc_id}})
    SET d.filename = $filename, d.doc_type = $doc_type
    """
    tx.run(merge_doc_query, doc_id=str(doc_id), filename=filename, doc_type=doc_type)

    # Group references to bypass loop-based sequential roundtrips.
    # Grouping matches pre-defined indexes securely without dynamic query interpolation.
    refs_by_label = {
        "Section": [],
        "Circular": [],
        "Notification": []
    }

    for ref in section_refs:
        ref_lower = ref.lower()
        if "circular" in ref_lower:
            refs_by_label["Circular"].append(ref)
        elif "notification" in ref_lower:
            refs_by_label["Notification"].append(ref)
        else:
            refs_by_label["Section"].append(ref)

    # Perform batch write for each specific target label using optimized UNWIND
    for ref_label, ref_ids in refs_by_label.items():
        if not ref_ids:
            continue
        
        # Batch query using parameterized references
        batch_query = f"""
        UNWIND $ref_ids AS ref_id
        MERGE (r:{ref_label} {{id: ref_id}})
        WITH r
        MATCH (d:{base_label} {{id: $doc_id}})
        MERGE (d)-[:MENTIONS]->(r)
        """
        tx.run(batch_query, ref_ids=ref_ids, doc_id=str(doc_id))


def ingest_document_pipeline(
    file_path: str, 
    filename: str, 
    embedder: Optional[DocumentEmbedder] = None
) -> int:
    """
    Orchestrates the entire ingestion pipeline:
    Parse -> Chunk -> Extract Metadata -> Embed Chunks -> Parallel Database Ingestion.
    
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

    # Step 3: Extract entity references and header metadata
    metadata = extract_metadata(parsed_doc.raw_text, parsed_doc.doc_type, chunks)
    section_refs = getattr(metadata, "section_refs", []) or []

    # Step 4: Generate dense vector embeddings for each chunk
    if embedder is None:
        embedder = _get_shared_embedder()
        
    chunk_texts = [chunk.text for chunk in chunks]
    embeddings = embedder.embed_texts(chunk_texts)

    # Step 5: Database Ingestion (Postgres, Qdrant, Neo4j)
    pg_conn = None
    pg_cursor = None
    
    # State flags to coordinate distributed rollbacks
    qdrant_client = None
    qdrant_points_uploaded = False
    qdrant_ids: List[str] = []
    doc_id = None

    try:
        pg_conn = get_postgres_connection()
        pg_cursor = pg_conn.cursor()

        # Step 5a: Insert Parent Document record in PostgreSQL
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

        # Bulk Insert Chunks to PostgreSQL
        execute_values(
            pg_cursor,
            """
            INSERT INTO chunks (document_id, chunk_text, section_ref, page_number, qdrant_id)
            VALUES %s;
            """,
            pg_chunks_payload
        )

        # Step 5b: Upsert vectors to Qdrant Collection
        qdrant_client = get_qdrant_client()
        qdrant_client.upsert(
            collection_name="legal_chunks",
            points=qdrant_points
        )
        qdrant_points_uploaded = True

        # Step 5c: Ingest relationships into Neo4j Graph
        neo4j_driver = get_neo4j_driver()
        with neo4j_driver.session() as neo4j_session:
            neo4j_session.execute_write(
                _ingest_to_neo4j,
                doc_id,
                filename,
                parsed_doc.doc_type,
                section_refs
            )

        # Step 5d: Mark processing as complete in PostgreSQL
        pg_cursor.execute(
            "UPDATE documents SET status = 'completed' WHERE id = %s;",
            (doc_id,)
        )
        pg_conn.commit()
        logger.info("Successfully ingested document '%s' with ID: %d", filename, doc_id)
        return doc_id

    except Exception as e:
        logger.exception("Ingestion pipeline failed on database transactions. Commencing rollbacks...")
        
        # Roll back PostgreSQL transaction
        if pg_conn:
            try:
                pg_conn.rollback()
            except Exception as pg_err:
                logger.error("Failed to rollback PostgreSQL connection: %s", pg_err)

        # Roll back Qdrant to prevent orphaned vector pollution
        if qdrant_points_uploaded and qdrant_client and qdrant_ids:
            try:
                logger.info("Cleaning up uploaded vectors in Qdrant to preserve transactional integrity...")
                qdrant_client.delete(
                    collection_name="legal_chunks",
                    points_selector=PointIdsList(points=qdrant_ids)
                )
            except Exception as qdrant_err:
                logger.error("Failed to clean up Qdrant points during rollback: %s", qdrant_err)

        # Mark document as failed in a clean database session
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
        # Prevent UnboundLocalError by executing safe cleanup checks
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