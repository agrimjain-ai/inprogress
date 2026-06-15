import psycopg2
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from backend.core.config import get_settings

settings = get_settings()

def init_postgres():
    print("Initializing PostgreSQL...")
    conn = psycopg2.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        dbname=settings.postgres_db
    )
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            filename TEXT NOT NULL,
            doc_type TEXT,          -- 'act', 'judgment', 'circular', 'notification'
            uploaded_at TIMESTAMP DEFAULT NOW(),
            status TEXT DEFAULT 'pending'
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            document_id INTEGER REFERENCES documents(id),
            chunk_text TEXT NOT NULL,
            section_ref TEXT,       -- e.g. 'Section 16(4)'
            page_number INTEGER,
            qdrant_id TEXT          -- links to vector in Qdrant
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id SERIAL PRIMARY KEY,
            query_text TEXT NOT NULL,
            response TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print("PostgreSQL initialized.")

def init_neo4j():
    print("Initializing Neo4j...")
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password)
    )
    with driver.session() as session:
        # Unique constraints — prevents duplicate nodes
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Section) REQUIRE n.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Judgment) REQUIRE n.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Circular) REQUIRE n.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Notification) REQUIRE n.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Amendment) REQUIRE n.id IS UNIQUE")

    driver.close()
    print("Neo4j initialized.")

def init_qdrant():
    print("Initializing Qdrant...")
    client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port
    )
    
    # 768 = vector size from sentence-transformers (all-mpnet-base-v2)
    client.recreate_collection(
        collection_name="legal_chunks",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )
    print("Qdrant initialized.")

if __name__ == "__main__":
    init_postgres()
    init_neo4j()
    init_qdrant()
    print("All databases initialized successfully.")