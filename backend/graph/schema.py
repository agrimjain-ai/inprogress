"""
backend/graph/schema.py

Neo4j schema definitions for GST Copilot — Phase 3 (Knowledge Graph).

Single source of truth for node labels, relationship types, and the
constraints/indexes that keep the graph clean. Import NodeLabel / RelType
from here in relationship_extractor.py and graph_builder.py instead of
hardcoding strings.
"""

import logging
from typing import Optional
from neo4j import Driver
from neo4j.exceptions import Neo4jError

# Initialize module-level logger
logger = logging.getLogger(__name__)


class NodeLabel:
    DOCUMENT = "Document"
    SECTION = "Section"
    JUDGMENT = "Judgment"
    NOTIFICATION = "Notification"
    CIRCULAR = "Circular"
    AMENDMENT = "Amendment"
    PERSON = "Person"
    ORGANIZATION = "Organization"


class RelType:
    AMENDED_BY = "AMENDED_BY"          # Section -> Amendment
    OVERRULED_BY = "OVERRULED_BY"      # Judgment -> Judgment
    CITED_BY = "CITED_BY"              # Judgment/Circular -> Judgment
    CROSS_REFERENCED = "CROSS_REFERENCED"  # Section -> Section
    AFFECTS = "AFFECTS"                # Notification -> Section
    MENTIONS = "MENTIONS"              # Document -> any entity (provenance)
    PRESIDED_OVER = "PRESIDED_OVER"    # Person -> Judgment
    DECIDED = "DECIDED"                # Organization -> Judgment


# Uniqueness constraints. These are what make MERGE safe — re-ingesting the
# same document reuses existing nodes instead of creating duplicates.
#
# Note on composite keys: Section and Amendment don't have a single natural
# unique field (e.g. "Section 16" exists in multiple Acts), so we build a
# composite `*_key` property at write time, e.g.
#   section_key = f"{act_name}::{section_number}"
#   amendment_key = f"{act_name}::{section_number}::{amendment_date}"
# This module only declares the constraint; graph_builder.py is responsible
# for computing and setting that property before MERGE.
CONSTRAINTS = [
    f"CREATE CONSTRAINT document_id IF NOT EXISTS "
    f"FOR (d:{NodeLabel.DOCUMENT}) REQUIRE d.doc_id IS UNIQUE",

    f"CREATE CONSTRAINT section_key IF NOT EXISTS "
    f"FOR (s:{NodeLabel.SECTION}) REQUIRE s.section_key IS UNIQUE",

    f"CREATE CONSTRAINT judgment_case_number IF NOT EXISTS "
    f"FOR (j:{NodeLabel.JUDGMENT}) REQUIRE j.case_number IS UNIQUE",

    f"CREATE CONSTRAINT notification_number IF NOT EXISTS "
    f"FOR (n:{NodeLabel.NOTIFICATION}) REQUIRE n.notification_number IS UNIQUE",

    f"CREATE CONSTRAINT circular_number IF NOT EXISTS "
    f"FOR (c:{NodeLabel.CIRCULAR}) REQUIRE c.circular_number IS UNIQUE",

    f"CREATE CONSTRAINT amendment_key IF NOT EXISTS "
    f"FOR (a:{NodeLabel.AMENDMENT}) REQUIRE a.amendment_key IS UNIQUE",

    f"CREATE CONSTRAINT person_name IF NOT EXISTS "
    f"FOR (p:{NodeLabel.PERSON}) REQUIRE p.name IS UNIQUE",

    f"CREATE CONSTRAINT org_name IF NOT EXISTS "
    f"FOR (o:{NodeLabel.ORGANIZATION}) REQUIRE o.name IS UNIQUE",
]

# Non-unique indexes, for query speed on fields we'll filter/sort by often.
INDEXES = [
    f"CREATE INDEX section_status IF NOT EXISTS "
    f"FOR (s:{NodeLabel.SECTION}) ON (s.status)",

    f"CREATE INDEX judgment_court IF NOT EXISTS "
    f"FOR (j:{NodeLabel.JUDGMENT}) ON (j.court_name)",

    f"CREATE INDEX judgment_date IF NOT EXISTS "
    f"FOR (j:{NodeLabel.JUDGMENT}) ON (j.date)",
]


def create_schema(driver: Driver, database: Optional[str] = None) -> None:
    """
    Idempotent — safe to call on every app startup.
    Creates all constraints and indexes if they don't already exist.
    Uses managed write transactions and explicit database parameters.
    """
    session_params = {}
    if database:
        session_params["database"] = database

    with driver.session(**session_params) as session:
        logger.info("Verifying and enforcing Neo4j schema constraints and indexes...")

        for stmt in CONSTRAINTS:
            try:
                session.execute_write(lambda tx: tx.run(stmt))
                name = stmt.split("IF NOT EXISTS")[0].replace("CREATE CONSTRAINT", "").strip()
                logger.info(f"Verified constraint: {name}")
            except Neo4jError as e:
                logger.error(f"Failed to apply constraint statement: {stmt}. Error: {e}")
                raise e

        for stmt in INDEXES:
            try:
                session.execute_write(lambda tx: tx.run(stmt))
                name = stmt.split("IF NOT EXISTS")[0].replace("CREATE INDEX", "").strip()
                logger.info(f"Verified index: {name}")
            except Neo4jError as e:
                logger.error(f"Failed to apply index statement: {stmt}. Error: {e}")
                raise e

        logger.info("Neo4j schema verification completed.")


def drop_schema(driver: Driver, database: Optional[str] = None) -> None:
    """
    Dev-only helper. Drops all constraints/indexes defined in the database.
    Does NOT delete any nodes or relationships — schema rules only.
    """
    session_params = {}
    if database:
        session_params["database"] = database

    with driver.session(**session_params) as session:
        logger.warning("Dropping Neo4j constraints and indexes...")

        # Retrieve and drop custom constraints
        try:
            constraints = session.execute_read(lambda tx: tx.run("SHOW CONSTRAINTS").data())
            for record in constraints:
                name = record.get("name")
                if name:
                    session.execute_write(lambda tx: tx.run(f"DROP CONSTRAINT {name} IF EXISTS"))
                    logger.info(f"Dropped constraint: {name}")
        except Neo4jError as e:
            logger.error(f"Failed to retrieve or drop constraints: {e}")

        # Retrieve and drop custom indexes (excluding default system-defined lookup indexes)
        try:
            indexes = session.execute_read(lambda tx: tx.run("SHOW INDEXES").data())
            for record in indexes:
                name = record.get("name")
                idx_type = record.get("type")
                if name and idx_type != "LOOKUP":
                    session.execute_write(lambda tx: tx.run(f"DROP INDEX {name} IF EXISTS"))
                    logger.info(f"Dropped index: {name}")
        except Neo4jError as e:
            logger.error(f"Failed to retrieve or drop indexes: {e}")


def verify_connectivity(driver: Driver) -> bool:
    """
    Performs a direct connection health check to the database.
    """
    try:
        driver.verify_connectivity()
        return True
    except Exception as e:
        logger.error(f"Neo4j connection verification failed: {e}")
        return False