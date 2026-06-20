"""
backend/scripts/backfill_graph.py

Database migration and backfill utility.
Scans PostgreSQL for all completed documents, reconstructs their raw texts
and chunks, extracts legal entities/relationships, and populates Neo4j.

Avoids duplicate embedding charges by leveraging existing relational text chunks.
"""

import sys
import logging
import argparse
from typing import Optional

# Core Database and Ingestion configurations
from backend.core.database import get_postgres_connection, get_neo4j_driver
from backend.ingestion.extractor import extract_entities

# Phase 3 Knowledge Graph operations
from backend.graph.relationship_extractor import extract_relationships
from backend.graph.graph_builder import build_relationships, build_mentions

# Setup structured logging for standalone script execution
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("backfill_graph")


def backfill_database(database: Optional[str] = None) -> None:
    """
    Executes the backfill migration pipeline.
    Reconstructs documents from relational database chunks and merges them to Neo4j.
    """
    logger.info("Initializing Neo4j Knowledge Graph backfill routine...")

    pg_conn = None
    pg_cursor = None
    neo4j_driver = None

    try:
        pg_conn = get_postgres_connection()
        pg_cursor = pg_conn.cursor()
        neo4j_driver = get_neo4j_driver()

        # Step 1: Retrieve all successfully processed parent documents
        pg_cursor.execute(
            "SELECT id, filename, doc_type FROM documents WHERE status = 'completed';"
        )
        documents = pg_cursor.fetchall()

        if not documents:
            logger.info("No completed documents found in PostgreSQL to migrate. Backfill terminated.")
            return

        logger.info(f"Identified {len(documents)} completed documents to backfill.")

        for doc_id, filename, doc_type in documents:
            logger.info(f"Processing Document ID {doc_id}: '{filename}' ({doc_type})...")

            # Step 2: Fetch and sort relational chunks to reconstruct the full document text
            pg_cursor.execute(
                "SELECT chunk_text FROM chunks WHERE document_id = %s ORDER BY id;",
                (doc_id,)
            )
            chunk_rows = pg_cursor.fetchall()

            if not chunk_rows:
                logger.warning(f"Skipping Document ID {doc_id} — no relational chunks found.")
                continue

            chunk_texts = [row[0] for row in chunk_rows]
            reconstructed_raw_text = "\n\n".join(chunk_texts)

            # Step 3: Extract entities for the document provenance mentions layer
            logger.info(f"Running entity extraction for Document ID {doc_id}...")
            entities = extract_entities(reconstructed_raw_text)

            # Step 4: Extract semantic legal relationships chunk-by-chunk
            logger.info(f"Running relationship extraction for Document ID {doc_id}...")
            all_relationships = []
            for idx, chunk_text in enumerate(chunk_texts):
                chunk_rels = extract_relationships(chunk_text, doc_id, idx)
                all_relationships.extend(chunk_rels)

            # Step 5: Merge backfilled components into Neo4j
            logger.info(
                f"Writing graph elements to Neo4j (Entities to merge: {len(entities)}, "
                f"Relationships to merge: {len(all_relationships)})..."
            )
            
            # Write mentions layer
            build_mentions(neo4j_driver, doc_id, entities, database=database)
            
            # Write relationship layer
            build_relationships(neo4j_driver, all_relationships, database=database)

            logger.info(f"Successfully backfilled Document ID {doc_id} into the Knowledge Graph.")

        logger.info("Knowledge Graph backfill routine executed successfully.")

    except Exception as e:
        logger.exception("An error occurred during the Knowledge Graph backfill execution.")
        sys.exit(1)

    finally:
        # Prevent resource leakages
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill missing Neo4j structural nodes and relations from Postgres.")
    parser.add_argument(
        "--database",
        type=str,
        default=None,
        help="Target Neo4j database name (defaults to default DBMS target)."
    )
    args = parser.parse_args()

    backfill_database(database=args.database)