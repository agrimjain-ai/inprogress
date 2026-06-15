import psycopg2
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from backend.core.config import get_settings

settings = get_settings()

# PostgreSQL Connection
def get_postgres_connection():
    return psycopg2.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        dbname=settings.postgres_db
    )

# Neo4j Driver
def get_neo4j_driver():
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password)
    )

# Qdrant Client
def get_qdrant_client():
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port
    )