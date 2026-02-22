"""
Main FastAPI application for REXI.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from typing import List, Optional

from rexi.config.settings import get_settings
from rexi.core.ingestion import IngestionEngine
from rexi.core.reasoning import ReasoningEngine
from rexi.core.knowledge_graph import KnowledgeGraph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="REXI API",
    description="Self-Learning Personal Knowledge Graph Assistant API",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services
ingestion_engine: Optional[IngestionEngine] = None
reasoning_engine: Optional[ReasoningEngine] = None
knowledge_graph: Optional[KnowledgeGraph] = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global ingestion_engine, reasoning_engine, knowledge_graph
    
    try:
        ingestion_engine = IngestionEngine()
        reasoning_engine = ReasoningEngine()
        knowledge_graph = KnowledgeGraph()
        logger.info("REXI services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global ingestion_engine, reasoning_engine, knowledge_graph
    
    try:
        if ingestion_engine:
            ingestion_engine.close()
        if reasoning_engine:
            reasoning_engine.close()
        if knowledge_graph:
            knowledge_graph.close()
        logger.info("REXI services shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "REXI - Self-Learning Personal Knowledge Graph Assistant",
        "version": "0.1.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check if services are available
        stats = knowledge_graph.get_statistics() if knowledge_graph else {}
        
        return {
            "status": "healthy",
            "services": {
                "ingestion": ingestion_engine is not None,
                "reasoning": reasoning_engine is not None,
                "knowledge_graph": knowledge_graph is not None
            },
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    """Ingest a file into the knowledge graph."""
    if not ingestion_engine:
        raise HTTPException(status_code=503, detail="Ingestion service not available")
    
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_extension = f".{file.filename.split('.')[-1].lower()}"
        if file_extension not in ingestion_engine.get_supported_extensions():
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_extension}"
            )
        
        # Save uploaded file
        import os
        import aiofiles
        
        upload_dir = settings.upload_dir
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Ingest file
        document = await ingestion_engine.ingest_file(file_path, "upload")
        
        return {
            "message": "File ingested successfully",
            "document_id": document.id,
            "title": document.title,
            "word_count": document.word_count,
            "entities_extracted": document.entities_extracted
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/text")
async def ingest_text(data: dict):
    """Ingest text directly into the knowledge graph."""
    if not ingestion_engine:
        raise HTTPException(status_code=503, detail="Ingestion service not available")
    
    try:
        text = data.get("text", "")
        title = data.get("title", "")
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text provided")
        
        document = await ingestion_engine.ingest_text(text, title, "api")
        
        return {
            "message": "Text ingested successfully",
            "document_id": document.id,
            "title": document.title,
            "word_count": document.word_count,
            "entities_extracted": document.entities_extracted
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_knowledge_graph(query_data: dict):
    """Query the knowledge graph."""
    if not reasoning_engine:
        raise HTTPException(status_code=503, detail="Reasoning service not available")
    
    try:
        query = query_data.get("query", "")
        max_hops = query_data.get("max_hops", 3)
        temperature = query_data.get("temperature", 0.3)
        
        if not query.strip():
            raise HTTPException(status_code=400, detail="No query provided")
        
        result = reasoning_engine.answer_query(query, max_hops, temperature)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/entities")
async def get_entities(
    entity_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """Get entities from the knowledge graph."""
    if not knowledge_graph:
        raise HTTPException(status_code=503, detail="Knowledge graph service not available")
    
    try:
        # This would need to be implemented in the knowledge graph service
        # For now, return empty list
        return {
            "entities": [],
            "total": 0,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Get entities failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/entities/{entity_id}")
async def get_entity(entity_id: str):
    """Get a specific entity by ID."""
    if not knowledge_graph:
        raise HTTPException(status_code=503, detail="Knowledge graph service not available")
    
    try:
        entity = knowledge_graph.get_entity(entity_id)
        
        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")
        
        return entity.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get entity failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/entities/{entity_id}/neighbors")
async def get_entity_neighbors(entity_id: str, depth: int = 1):
    """Get neighbors of an entity."""
    if not knowledge_graph:
        raise HTTPException(status_code=503, detail="Knowledge graph service not available")
    
    try:
        neighbors = knowledge_graph.get_entity_neighbors(entity_id, depth)
        
        return {
            "entity_id": entity_id,
            "depth": depth,
            "neighbors": [neighbor.to_dict() for neighbor in neighbors]
        }
        
    except Exception as e:
        logger.error(f"Get neighbors failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/relationships")
async def get_relationships(
    source_id: Optional[str] = None,
    target_id: Optional[str] = None,
    relationship_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """Get relationships from the knowledge graph."""
    if not knowledge_graph:
        raise HTTPException(status_code=503, detail="Knowledge graph service not available")
    
    try:
        relationships = knowledge_graph.get_relationships(source_id, relationship_type)
        
        return {
            "relationships": [rel.to_dict() for rel in relationships],
            "total": len(relationships),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Get relationships failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/entities/search")
async def search_entities(search_data: dict):
    """Search for entities by text similarity."""
    if not knowledge_graph:
        raise HTTPException(status_code=503, detail="Knowledge graph service not available")
    
    try:
        query_text = search_data.get("query", "")
        entity_type = search_data.get("entity_type")
        limit = search_data.get("limit", 10)
        
        if not query_text.strip():
            raise HTTPException(status_code=400, detail="No query provided")
        
        results = knowledge_graph.find_similar_entities(query_text, entity_type, limit)
        
        return {
            "query": query_text,
            "results": [
                {
                    "entity": result["entity"].to_dict(),
                    "similarity_score": result["similarity_score"]
                }
                for result in results
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Entity search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/insights")
async def get_insights(entity_types: Optional[List[str]] = None):
    """Get insights from the knowledge graph."""
    if not reasoning_engine:
        raise HTTPException(status_code=503, detail="Reasoning service not available")
    
    try:
        # Convert string types to EntityType enums
        from rexi.models.entities import EntityType
        types = None
        if entity_types:
            types = [EntityType(t) for t in entity_types if t in EntityType.__members__]
        
        insights = reasoning_engine.find_insights(types)
        
        return {
            "insights": insights,
            "timestamp": knowledge_graph.get_statistics().get("last_updated")
        }
        
    except Exception as e:
        logger.error(f"Get insights failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
async def explain_relationship(explain_data: dict):
    """Explain relationship between two entities."""
    if not reasoning_engine:
        raise HTTPException(status_code=503, detail="Reasoning service not available")
    
    try:
        source_entity = explain_data.get("source_entity")
        target_entity = explain_data.get("target_entity")
        relationship_type = explain_data.get("relationship_type")
        
        if not source_entity or not target_entity:
            raise HTTPException(status_code=400, detail="Both source and target entities required")
        
        explanation = reasoning_engine.explain_relationship(
            source_entity, target_entity, relationship_type
        )
        
        return explanation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Relationship explanation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def get_statistics():
    """Get knowledge graph statistics."""
    if not knowledge_graph:
        raise HTTPException(status_code=503, detail="Knowledge graph service not available")
    
    try:
        stats = knowledge_graph.get_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"Get statistics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
