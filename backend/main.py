"""
GST Copilot FastAPI Application
Main entry point for all API endpoints
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import get_settings
from backend.api.routes.upload import router as upload_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="GST Copilot API",
    description="AI-powered GST legal document analysis and retrieval system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware
settings = get_settings()
origins = getattr(settings, "ALLOWED_ORIGINS", ["*"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(upload_router)

logger.info("GST Copilot API initialized")


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    Verifies that the API is running and databases are accessible
    """
    try:
        from backend.core.database import get_postgres_connection, get_neo4j_driver, get_qdrant_client
        
        # Test PostgreSQL connection
        pg_conn = get_postgres_connection()
        pg_conn.close()
        
        # Test Neo4j connection
        neo4j_driver = get_neo4j_driver()
        neo4j_driver.verify_connectivity()
        
        # Test Qdrant connection
        qdrant_client = get_qdrant_client()
        qdrant_client.get_collections()
        
        logger.info("Health check passed: all databases accessible")
        return {
            "status": "ok",
            "message": "All databases connected",
            "postgres": "connected",
            "neo4j": "connected",
            "qdrant": "connected"
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "error",
            "message": f"Database connection failed: {str(e)}"
        }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "GST Copilot API",
        "version": "1.0.0",
        "status": "running",
        "docs": "http://localhost:8000/docs",
        "endpoints": {
            "upload": "POST /api/upload",
            "list_documents": "GET /api/documents",
            "delete_document": "DELETE /api/documents/{doc_id}",
            "health": "GET /health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run with: python -m backend.main
    # Or: python backend/main.py
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )