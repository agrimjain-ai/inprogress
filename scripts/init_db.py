import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from qdrant_client.models import VectorParams, Distance
from backend.core.database import (
    get_postgres_connection,
    get_neo4j_driver,
    get_qdrant_client
)

def init_postgres():
    print("Initializing PostgreSQL...")
    conn = get_postgres_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            filename TEXT NOT NULL,
            doc_type TEXT,
            uploaded_at TIMESTAMP DEFAULT NOW(),
            status TEXT DEFAULT 'pending'
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            document_id INTEGER REFERENCES documents(id),
            chunk_text TEXT NOT NULL,
            section_ref TEXT,
            page_number INTEGER,
            qdrant_id TEXT
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
    print("PostgreSQL done.")

def init_neo4j():
    print("Initializing Neo4j...")
    driver = get_neo4j_driver()

    with driver.session() as session:
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Section) REQUIRE n.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Judgment) REQUIRE n.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Circular) REQUIRE n.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Notification) REQUIRE n.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Amendment) REQUIRE n.id IS UNIQUE")

    driver.close()
    print("Neo4j done.")

def init_qdrant():
    print("Initializing Qdrant...")
    client = get_qdrant_client()

    if not client.collection_exists("legal_chunks"):
        client.create_collection(
            collection_name="legal_chunks",
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
    print("Qdrant done.")

if __name__ == "__main__":
    init_postgres()
    init_neo4j()
    init_qdrant()
    print("\nAll databases initialized successfully.")