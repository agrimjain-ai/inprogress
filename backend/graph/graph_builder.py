"""
backend/graph/graph_builder.py

Takes entities (from extractor.py) and relationships (from
relationship_extractor.py) and writes them into Neo4j as nodes + edges.

Responsibilities:
- Compute composite keys for nodes that don't have one natural unique field
  (Section, Amendment) — see schema.py for why these exist.
- MERGE nodes (never CREATE directly) so re-ingesting a document never
  duplicates graph data.
- MERGE relationships with provenance properties (doc_id, chunk_index,
  confidence, extraction_method) so every edge can be traced back to the
  chunk that produced it.
- UNWIND grouping strategies for high-performance batch writes.

Known key-collision fixes:
- Section nodes are keyed by (act_name, section_text). When no act name is
  captured from text, they fall back to DEFAULT_ACT_NAME instead of a
  generic "Unknown Act" bucket — a domain-informed default, not a guarantee
  of correctness. Verify against real documents before trusting cross-Act
  dedup.
- Amendment nodes are keyed on citation text alone (no doc_id scoping), so
  the same Notification/Act cited across multiple documents converges on
  one canonical node instead of fragmenting per document.
"""

import logging
from typing import List, Dict, Any, Optional

from neo4j import Driver, Transaction

from backend.graph.schema import NodeLabel, RelType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public Composite Key Helpers (Exposed to prevent schema drift with queries.py)
# ---------------------------------------------------------------------------

# GST law in India is structured as CGST Act + per-state SGST Acts + IGST
# Act, and CGST/SGST provisions are pari materia (mirror text) by design.
# In practice, legal commentary and case law overwhelmingly cite an
# unqualified "Section X" as a CGST Act reference. This is a heuristic, not
# a guarantee — it can still merge sections that are technically from a
# different (but textually identical) state Act.
DEFAULT_ACT_NAME = "CGST Act"


def get_section_key(section_text: str, act_name: Optional[str] = None) -> str:
    """Computes a standardized composite section key."""
    if not act_name:
        logger.debug(
            f"No act name captured for '{section_text}' — defaulting to "
            f"'{DEFAULT_ACT_NAME}'."
        )
        act_name = DEFAULT_ACT_NAME
    return f"{act_name.strip()}::{section_text.strip()}"


def get_amendment_key(amendment_text: str) -> str:
    """Computes a standardized global amendment key based purely on citation text."""
    return amendment_text.strip()


def get_node_key_field(node_type: str) -> str:
    """Returns the property name that uniquely identifies each node type."""
    return {
        NodeLabel.SECTION: "section_key",
        NodeLabel.AMENDMENT: "amendment_key",
        NodeLabel.JUDGMENT: "case_number",
        NodeLabel.NOTIFICATION: "notification_number",
        NodeLabel.CIRCULAR: "circular_number",
        NodeLabel.PERSON: "name",
        NodeLabel.ORGANIZATION: "name",
        NodeLabel.DOCUMENT: "doc_id",
    }[node_type]


def get_node_key(
    node_type: str, raw_text: str, act_name: Optional[str] = None
) -> str:
    """
    Computes the unique key value for any given node type.
    Note: doc_id parameter was safely removed as amendments are now globally scoped.
    """
    if node_type == NodeLabel.SECTION:
        return get_section_key(raw_text, act_name)
    if node_type == NodeLabel.AMENDMENT:
        return get_amendment_key(raw_text)
    return raw_text.strip()


# ---------------------------------------------------------------------------
# High-Performance Batch Write Queries (UNWIND)
# ---------------------------------------------------------------------------

def _get_batch_merge_query(
    source_type: str, source_key_field: str,
    rel_type: str,
    target_type: str, target_key_field: str,
) -> str:
    """
    Generates an optimized Cypher query to write a batch of relationships of a specific signature.
    Safely handles node endpoint merges and relationship generation in a single database round-trip.
    """
    return (
        f"UNWIND $batch AS item\n"
        f"MERGE (s:{source_type} {{{source_key_field}: item.source_key}}) "
        f"ON CREATE SET s.created_at = datetime(), s.raw_text = item.source_raw_text "
        f"ON MATCH SET s.updated_at = datetime()\n"
        f"MERGE (t:{target_type} {{{target_key_field}: item.target_key}}) "
        f"ON CREATE SET t.created_at = datetime(), t.raw_text = item.target_raw_text "
        f"ON MATCH SET t.updated_at = datetime()\n"
        f"MERGE (s)-[r:{rel_type}]->(t) "
        f"ON CREATE SET r.confidence = item.confidence, "
        f"              r.extraction_method = item.extraction_method, "
        f"              r.doc_id = item.doc_id, "
        f"              r.chunk_index = item.chunk_index, "
        f"              r.created_at = datetime() "
        f"ON MATCH SET  r.confidence = CASE WHEN item.confidence > r.confidence "
        f"                                  THEN item.confidence ELSE r.confidence END, "
        f"              r.updated_at = datetime()"
    )


# ---------------------------------------------------------------------------
# Node + relationship writes
# ---------------------------------------------------------------------------

def build_relationships(
    driver: Driver, relationships: List[Dict[str, Any]], database: Optional[str] = None
) -> int:
    """
    Writes a batch of extracted relationships into Neo4j using managed transaction
    retries and batch performance processing via Cypher UNWIND.

    Returns the number of relationships successfully written.
    """
    if not relationships:
        return 0

    session_params = {"database": database} if database else {}
    written_count = 0

    valid_types = {
        NodeLabel.DOCUMENT, NodeLabel.SECTION, NodeLabel.JUDGMENT,
        NodeLabel.NOTIFICATION, NodeLabel.CIRCULAR, NodeLabel.AMENDMENT,
        NodeLabel.PERSON, NodeLabel.ORGANIZATION
    }

    with driver.session(**session_params) as session:
        # Run everything inside a managed write transaction function
        def _write_relationships_tx(tx: Transaction) -> int:
            # 1. Pre-resolve case numbers for CITED_BY relationships to prevent N+1 queries.
            citing_keys_cache: Dict[int, Optional[str]] = {}
            target_citing_doc_ids = {
                r["doc_id"] for r in relationships 
                if r["rel_type"] == RelType.CITED_BY and r["source_text"] is None
            }
            for doc_id in target_citing_doc_ids:
                citing_keys_cache[doc_id] = _resolve_citing_judgment_key_tx(tx, doc_id)

            # 2. Group relationships by signature: (source_type, rel_type, target_type)
            grouped_batches: Dict[tuple, List[Dict[str, Any]]] = {}
            skipped_count = 0

            for rel in relationships:
                s_type = rel.get("source_type")
                r_type = rel.get("rel_type")
                t_type = rel.get("target_type")

                if s_type not in valid_types or t_type not in valid_types:
                    logger.warning(f"Skipping relationship with invalid types: {s_type} -> {t_type}")
                    skipped_count += 1
                    continue

                source_act = rel.get("source_act")
                target_act = rel.get("target_act")

                # Unpack and compute source keys
                if r_type == RelType.CITED_BY and rel["source_text"] is None:
                    source_key = citing_keys_cache.get(rel["doc_id"])
                    if not source_key:
                        logger.warning(
                            f"Skipping CITED_BY relationship — no Judgment node "
                            f"found for doc_id {rel['doc_id']}"
                        )
                        skipped_count += 1
                        continue
                else:
                    source_key = get_node_key(s_type, rel["source_text"], source_act)

                target_key = get_node_key(t_type, rel["target_text"], target_act)

                signature = (s_type, r_type, t_type)
                if signature not in grouped_batches:
                    grouped_batches[signature] = []

                grouped_batches[signature].append({
                    "source_key": source_key,
                    "source_raw_text": rel["source_text"] if rel["source_text"] else source_key,
                    "target_key": target_key,
                    "target_raw_text": rel["target_text"],
                    "confidence": rel["confidence"],
                    "extraction_method": rel["extraction_method"],
                    "doc_id": rel["doc_id"],
                    "chunk_index": rel["chunk_index"],
                })

            if skipped_count:
                logger.warning(f"Skipped {skipped_count} relationships during batch build")

            # 3. Perform batch writes for each signature group
            total_written = 0
            for signature, batch in grouped_batches.items():
                s_type, r_type, t_type = signature
                s_key_field = get_node_key_field(s_type)
                t_key_field = get_node_key_field(t_type)

                query = _get_batch_merge_query(s_type, s_key_field, r_type, t_type, t_key_field)
                tx.run(query, batch=batch)
                total_written += len(batch)

            return total_written

        try:
            written_count = session.execute_write(_write_relationships_tx)
        except Exception as e:
            logger.error(f"Failed transaction while building relationships: {e}")
            raise e

    logger.info(f"Successfully processed and merged {written_count} relationships into Neo4j")
    return written_count


def _resolve_citing_judgment_key_tx(tx: Transaction, doc_id: int) -> Optional[str]:
    """
    Internal transactional helper. Resolves a citing judgment case number inside 
    an active transaction without spawning a nested driver session.
    """
    query = (
        f"MATCH (d:{NodeLabel.DOCUMENT} {{doc_id: $doc_id}}) "
        f"OPTIONAL MATCH (j:{NodeLabel.JUDGMENT} {{case_number: d.case_number}}) "
        f"RETURN j.case_number AS case_number"
    )
    record = tx.run(query, doc_id=doc_id).single()
    if record and record["case_number"]:
        return record["case_number"]
    return None


def build_mentions(
    driver: Driver, doc_id: int, entities: List[Dict[str, Any]], database: Optional[str] = None
) -> int:
    """
    Links a Document node to every entity extracted from it via MENTIONS.
    This is the provenance layer — lets us answer "which documents talk
    about Section 16(4)?" without going through relationship extraction.

    Optimized to perform single-transaction batched merges grouped by node labels.

    Note: entities here come from extractor.py's extract_entities(), which
    doesn't currently capture an act name per entity. Section entities
    mentioned via MENTIONS will therefore always fall back to
    DEFAULT_ACT_NAME — only relationship_extractor.py's AMENDED_BY pattern
    captures a real act name today.
    """
    if not entities:
        return 0

    session_params = {"database": database} if database else {}
    written_count = 0

    valid_types = {
        NodeLabel.PERSON, NodeLabel.ORGANIZATION,
        NodeLabel.SECTION, NodeLabel.JUDGMENT,
        NodeLabel.NOTIFICATION, NodeLabel.CIRCULAR,
    }

    # Group entities by label type to facilitate UNWIND parameterization
    grouped_entities: Dict[str, List[Dict[str, Any]]] = {}
    for entity in entities:
        n_type = entity.get("type")
        if n_type not in valid_types:
            logger.warning(f"Skipping unknown entity type during mentions build: {n_type}")
            continue

        if n_type not in grouped_entities:
            grouped_entities[n_type] = []

        grouped_entities[n_type].append({
            "entity_name": entity["entity"],
            "key": get_node_key(n_type, entity["entity"])
        })

    with driver.session(**session_params) as session:
        def _write_mentions_tx(tx: Transaction) -> int:
            # 1. Verify/merge parent Document first
            tx.run(
                f"MERGE (d:{NodeLabel.DOCUMENT} {{doc_id: $doc_id}}) "
                f"ON CREATE SET d.created_at = datetime() "
                f"ON MATCH SET d.updated_at = datetime()",
                doc_id=doc_id
            )

            # 2. Write batches
            total_mentions_written = 0
            for n_type, batch in grouped_entities.items():
                key_field = get_node_key_field(n_type)

                query = (
                    f"UNWIND $batch AS item\n"
                    f"MERGE (e:{n_type} {{{key_field}: item.key}}) "
                    f"ON CREATE SET e.created_at = datetime(), e.raw_text = item.entity_name\n"
                    f"MATCH (d:{NodeLabel.DOCUMENT} {{doc_id: $doc_id}})\n"
                    f"MERGE (d)-[r:{RelType.MENTIONS}]->(e) "
                    f"ON CREATE SET r.created_at = datetime()"
                )
                tx.run(query, batch=batch, doc_id=doc_id)
                total_mentions_written += len(batch)

            return total_mentions_written

        try:
            written_count = session.execute_write(_write_mentions_tx)
        except Exception as e:
            logger.error(f"Transaction failed while building mentions for doc_id {doc_id}: {e}")
            raise e

    logger.info(f"Successfully processed and merged {written_count} MENTIONS links for doc_id {doc_id}")
    return written_count