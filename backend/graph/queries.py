"""
backend/graph/queries.py

Traverses and queries the Neo4j Knowledge Graph for GST Copilot.

Provides the query layer for graph-augmented legal retrieval (RAG). Helps answer 
complex structural legal questions such as:
- Is Section X amended or affected by Notification Y?
- Has Judgment A been overruled, directly or transitively?
- What are the cross-referenced sections and cited precedents for a given case?

Implements transaction-safe execution and leverages key-derivation abstractions
to ensure exact-match synchronization with the graph builder.
"""

import logging
from typing import List, Dict, Any, Optional
from neo4j import Driver, Transaction
from neo4j.exceptions import Neo4jError

from backend.graph.schema import NodeLabel, RelType
from backend.graph.graph_builder import (
    get_section_key,
    get_node_key,
    get_node_key_field,
    DEFAULT_ACT_NAME
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section Queries
# ---------------------------------------------------------------------------

def get_section_context_for_rag(
    driver: Driver, section_text: str, act_name: Optional[str] = None, database: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Retrieves the entire structural surrounding context for a specific GST Section.
    Finds directly related notifications, amendments, and cross-references.
    
    This context can be injected directly into LLM prompt templates to ground answers.
    """
    section_key = get_section_key(section_text, act_name)
    session_params = {"database": database} if database else {}

    query = (
        f"MATCH (s:{NodeLabel.SECTION} {{section_key: $section_key}})\n"
        f"OPTIONAL MATCH (s)-[:{RelType.AMENDED_BY}]->(a:{NodeLabel.AMENDMENT})\n"
        f"WITH s, collect(distinct a.raw_text) AS amendments\n"
        f"OPTIONAL MATCH (s)-[:{RelType.CROSS_REFERENCED}]-(other_s:{NodeLabel.SECTION})\n"
        f"WITH s, amendments, collect(distinct other_s.raw_text) AS cross_references\n"
        f"OPTIONAL MATCH (n:{NodeLabel.NOTIFICATION})-[:{RelType.AFFECTS}]->(s)\n"
        f"WITH s, amendments, cross_references, collect(distinct n.notification_number) AS notifications\n"
        f"RETURN s.raw_text AS section_name, "
        f"       s.section_key AS section_key, "
        f"       s.status AS status, "
        f"       amendments, "
        f"       cross_references, "
        f"       notifications"
    )

    def _read_tx(tx: Transaction) -> Optional[Dict[str, Any]]:
        record = tx.run(query, section_key=section_key).single()
        if record:
            return dict(record)
        return None

    try:
        with driver.session(**session_params) as session:
            return session.execute_read(_read_tx)
    except Neo4jError as e:
        logger.error(f"Error querying section context for key '{section_key}': {e}")
        return None


# ---------------------------------------------------------------------------
# Precedent & Overruling Queries
# ---------------------------------------------------------------------------

def check_judgment_status(
    driver: Driver, case_number: str, database: Optional[str] = None
) -> Dict[str, Any]:
    """
    Checks if a judgment has been overruled, either directly or transitively.
    Traces a bounded path of OVERRULED_BY relationships up to a depth of 3.
    
    Returns a status dictionary detailing if the case is safe to cite.
    """
    session_params = {"database": database} if database else {}

    # Traces overruling lineage: j -> overruling_case_1 -> overruling_case_2...
    query = (
        f"MATCH (j:{NodeLabel.JUDGMENT} {{case_number: $case_number}})\n"
        f"OPTIONAL MATCH path = (j)-[:{RelType.OVERRULED_BY}*1..3]->(overruling_case:{NodeLabel.JUDGMENT})\n"
        f"RETURN j.case_number AS case_number, "
        f"       j.court_name AS court, "
        f"       j.date AS date, "
        f"       length(path) > 0 AS is_overruled, "
        f"       [node in nodes(path) | node.case_number] AS overruling_lineage"
    )

    def _read_tx(tx: Transaction) -> Dict[str, Any]:
        record = tx.run(query, case_number=case_number.strip()).single()
        if record:
            return dict(record)
        return {
            "case_number": case_number,
            "is_overruled": False,
            "overruling_lineage": [],
            "error": "Judgment node not found in graph."
        }

    try:
        with driver.session(**session_params) as session:
            return session.execute_read(_read_tx)
    except Neo4jError as e:
        logger.error(f"Error checking overruling status for case '{case_number}': {e}")
        return {"case_number": case_number, "is_overruled": False, "overruling_lineage": [], "error": str(e)}


def get_judgment_citation_network(
    driver: Driver, case_number: str, database: Optional[str] = None
) -> Dict[str, List[str]]:
    """
    Finds the citation network surrounding a given judgment case:
    1. What other judgments/notifications does this case rely on (CITED_BY out)?
    2. What other judgments cite this case (CITED_BY in)?
    """
    session_params = {"database": database} if database else {}

    query = (
        f"MATCH (j:{NodeLabel.JUDGMENT} {{case_number: $case_number}})\n"
        f"OPTIONAL MATCH (cited:{NodeLabel.JUDGMENT})-[:{RelType.CITED_BY}]->(j)\n"
        f"WITH j, collect(distinct cited.case_number) AS cited_by_this_case\n"
        f"OPTIONAL MATCH (j)-[:{RelType.CITED_BY}]->(citing:{NodeLabel.JUDGMENT})\n"
        f"RETURN j.case_number AS case_number, "
        f"       cited_by_this_case AS precedents_used, "
        f"       collect(distinct citing.case_number) AS subsequent_citations"
    )

    def _read_tx(tx: Transaction) -> Dict[str, List[str]]:
        record = tx.run(query, case_number=case_number.strip()).single()
        if record:
            return {
                "precedents_used": record["precedents_used"],
                "subsequent_citations": record["subsequent_citations"]
            }
        return {"precedents_used": [], "subsequent_citations": []}

    try:
        with driver.session(**session_params) as session:
            return session.execute_read(_read_tx)
    except Neo4jError as e:
        logger.error(f"Error checking citation network for '{case_number}': {e}")
        return {"precedents_used": [], "subsequent_citations": []}


# ---------------------------------------------------------------------------
# Generic Subgraph / Neighborhood Traversal
# ---------------------------------------------------------------------------

def get_entity_neighborhood(
    driver: Driver, node_type: str, raw_text: str, act_name: Optional[str] = None, limit: int = 25, database: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    A generic neighborhood expansion query. Useful for exploration or graph-based RAG
    when the exact relationship patterns are unknown.
    
    Expanse is bounded strictly by label verification to prevent parameter injection.
    """
    valid_types = {
        NodeLabel.SECTION, NodeLabel.JUDGMENT, NodeLabel.NOTIFICATION,
        NodeLabel.CIRCULAR, NodeLabel.AMENDMENT, NodeLabel.PERSON,
        NodeLabel.ORGANIZATION, NodeLabel.DOCUMENT
    }

    if node_type not in valid_types:
        logger.error(f"Invalid node label received for expansion: {node_type}")
        return []

    # Safe computation of the key field
    key_field = get_node_key_field(node_type)
    key_value = get_node_key(node_type, raw_text, act_name)
    session_params = {"database": database} if database else {}

    # Construct clean parameterized query with strict type strings
    query = (
        f"MATCH (n:{node_type} {{{key_field}: $key}})-[r]-(target)\n"
        f"RETURN type(r) AS rel_type, "
        f"       startNode(r) = n AS is_outgoing, "
        f"       labels(target)[0] AS target_label, "
        f"       coalesce(target.raw_text, target.case_number, target.notification_number, target.name) AS target_name, "
        f"       properties(r) AS rel_properties "
        f"LIMIT $limit"
    )

    def _read_tx(tx: Transaction) -> List[Dict[str, Any]]:
        results = []
        cursor = tx.run(query, key=key_value, limit=limit)
        for record in cursor:
            results.append(dict(record))
        return results

    try:
        with driver.session(**session_params) as session:
            return session.execute_read(_read_tx)
    except Neo4jError as e:
        logger.error(f"Neighborhood query failed for ({node_type}: {key_value}): {e}")
        return []


# ---------------------------------------------------------------------------
# Hybrid Retrieval (Vector to Graph Link)
# ---------------------------------------------------------------------------

def get_graph_context_for_documents(
    driver: Driver, doc_ids: List[int], database: Optional[str] = None
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Acts as the vital bridge between Qdrant/PostgreSQL and Neo4j.
    Takes a set of retrieved Document IDs (resolved during vector retrieval)
    and pulls all corresponding structural entities mentioned in those sources.
    
    Allows RAG systems to synthesize: "Chunk X mentions Section 16(4) which is affected by..."
    """
    if not doc_ids:
        return {}

    session_params = {"database": database} if database else {}

    query = (
        f"UNWIND $doc_ids AS doc_id\n"
        f"MATCH (d:{NodeLabel.DOCUMENT} {{doc_id: doc_id}})-[r:{RelType.MENTIONS}]->(e)\n"
        f"RETURN d.doc_id AS doc_id, "
        f"       labels(e)[0] AS entity_type, "
        f"       coalesce(e.raw_text, e.case_number, e.notification_number, e.name) AS entity_name"
    )

    def _read_tx(tx: Transaction) -> Dict[int, List[Dict[str, Any]]]:
        structured_context: Dict[int, List[Dict[str, Any]]] = {d: [] for d in doc_ids}
        cursor = tx.run(query, doc_ids=doc_ids)
        for record in cursor:
            d_id = record["doc_id"]
            if d_id in structured_context:
                structured_context[d_id].append({
                    "entity_type": record["entity_type"],
                    "entity_name": record["entity_name"]
                })
        return structured_context

    try:
        with driver.session(**session_params) as session:
            return session.execute_read(_read_tx)
    except Neo4jError as e:
        logger.error(f"Error bridging Document context from Graph: {e}")
        return {}