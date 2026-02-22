# REXI API Documentation

## Overview

REXI (Self-Learning Personal Knowledge Graph Assistant) provides a RESTful API for ingesting documents, managing knowledge graphs, and performing reasoning queries.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Currently, the API does not require authentication. This will be implemented in future versions.

## Endpoints

### Health Check

#### GET /health

Check the health status of all services.

**Response:**
```json
{
  "status": "healthy",
  "services": {
    "ingestion": true,
    "reasoning": true,
    "knowledge_graph": true
  },
  "statistics": {
    "entity_count": 150,
    "relationship_count": 300,
    "vector_count": 150,
    "last_updated": "2024-02-22T20:00:00.000Z"
  }
}
```

### Document Ingestion

#### POST /ingest/file

Upload and ingest a file into the knowledge graph.

**Request:** `multipart/form-data`
- `file`: File to ingest (required)

**Supported file types:**
- PDF (.pdf)
- Text (.txt)
- Markdown (.md)
- JSON (.json)
- CSV (.csv)
- Word documents (.docx, .doc)
- Rich Text Format (.rtf)
- EPUB (.epub)
- MOBI (.mobi)

**Response:**
```json
{
  "message": "File ingested successfully",
  "document_id": "doc_123",
  "title": "example.pdf",
  "word_count": 1500,
  "entities_extracted": true
}
```

#### POST /ingest/text

Ingest text directly into the knowledge graph.

**Request:**
```json
{
  "text": "This is the text content to ingest...",
  "title": "Optional title"
}
```

**Response:**
```json
{
  "message": "Text ingested successfully",
  "document_id": "doc_124",
  "title": "Optional title",
  "word_count": 25,
  "entities_extracted": true
}
```

### Knowledge Graph Queries

#### POST /query

Ask questions and get answers from the knowledge graph.

**Request:**
```json
{
  "query": "What is machine learning?",
  "max_hops": 3,
  "temperature": 0.3
}
```

**Response:**
```json
{
  "query": "What is machine learning?",
  "answer": "Machine learning is a subset of artificial intelligence...",
  "reasoning_path": ["entity_1", "entity_2", "entity_3"],
  "confidence": 0.85,
  "evidence": ["doc_123", "doc_124"],
  "entities_used": [
    {
      "id": "entity_1",
      "name": "Machine Learning",
      "type": "concept",
      "description": "A field of AI..."
    }
  ],
  "timestamp": "2024-02-22T20:00:00.000Z"
}
```

#### GET /entities

Retrieve entities from the knowledge graph.

**Query Parameters:**
- `entity_type` (optional): Filter by entity type
- `limit` (default: 50): Maximum number of entities to return
- `offset` (default: 0): Number of entities to skip

**Response:**
```json
{
  "entities": [
    {
      "id": "entity_1",
      "name": "Machine Learning",
      "type": "concept",
      "description": "A subset of AI...",
      "confidence": 0.9,
      "created_at": "2024-02-22T19:00:00.000Z",
      "updated_at": "2024-02-22T19:00:00.000Z"
    }
  ],
  "total": 150,
  "limit": 50,
  "offset": 0
}
```

#### GET /entities/{entity_id}

Get a specific entity by ID.

**Response:**
```json
{
  "id": "entity_1",
  "name": "Machine Learning",
  "type": "concept",
  "description": "A subset of artificial intelligence...",
  "confidence": 0.9,
  "created_at": "2024-02-22T19:00:00.000Z",
  "updated_at": "2024-02-22T19:00:00.000Z",
  "source_references": ["doc_123"],
  "privacy_level": "private"
}
```

#### GET /entities/{entity_id}/neighbors

Get neighboring entities within specified depth.

**Query Parameters:**
- `depth` (default: 1): Maximum depth for neighbor search

**Response:**
```json
{
  "entity_id": "entity_1",
  "depth": 1,
  "neighbors": [
    {
      "id": "entity_2",
      "name": "Neural Networks",
      "type": "concept",
      "description": "Computing systems inspired by biological neural networks"
    }
  ]
}
```

#### GET /relationships

Retrieve relationships from the knowledge graph.

**Query Parameters:**
- `source_id` (optional): Filter by source entity ID
- `target_id` (optional): Filter by target entity ID
- `relationship_type` (optional): Filter by relationship type
- `limit` (default: 50): Maximum number of relationships
- `offset` (default: 0): Number of relationships to skip

**Response:**
```json
{
  "relationships": [
    {
      "id": "rel_1",
      "source_entity_id": "entity_1",
      "target_entity_id": "entity_2",
      "type": "enables",
      "strength_score": 0.8,
      "confidence": 0.9,
      "created_at": "2024-02-22T19:00:00.000Z",
      "updated_at": "2024-02-22T19:00:00.000Z"
    }
  ],
  "total": 300,
  "limit": 50,
  "offset": 0
}
```

#### POST /entities/search

Search for entities using semantic similarity.

**Request:**
```json
{
  "query": "artificial intelligence",
  "entity_type": "concept",
  "limit": 10
}
```

**Response:**
```json
{
  "query": "artificial intelligence",
  "results": [
    {
      "entity": {
        "id": "entity_1",
        "name": "Artificial Intelligence",
        "type": "concept",
        "description": "The simulation of human intelligence..."
      },
      "similarity_score": 0.95
    }
  ]
}
```

### Insights and Analysis

#### GET /insights

Get insights and patterns from the knowledge graph.

**Query Parameters:**
- `entity_types` (optional): Array of entity types to analyze

**Response:**
```json
{
  "insights": [
    {
      "type": "hub_entities",
      "description": "Highly connected entities in the graph",
      "data": [
        {
          "entity": "Machine Learning",
          "connection_count": 25,
          "centrality_score": 0.85
        }
      ]
    }
  ],
  "timestamp": "2024-02-22T20:00:00.000Z"
}
```

#### POST /explain

Explain the relationship between two entities.

**Request:**
```json
{
  "source_entity": "entity_1",
  "target_entity": "entity_2",
  "relationship_type": "enables"
}
```

**Response:**
```json
{
  "explanation": "Machine Learning enables AI Systems by providing algorithms and techniques...",
  "confidence": 0.8,
  "path": [
    {
      "from": {
        "id": "entity_1",
        "name": "Machine Learning",
        "type": "concept"
      },
      "relationship": {
        "type": "enables",
        "confidence": 0.8
      },
      "to": {
        "id": "entity_2",
        "name": "AI Systems",
        "type": "concept"
      }
    }
  ],
  "path_length": 1
}
```

#### GET /statistics

Get knowledge graph statistics.

**Response:**
```json
{
  "entity_count": 150,
  "relationship_count": 300,
  "vector_count": 150,
  "last_updated": "2024-02-22T20:00:00.000Z"
}
```

## Entity Types

- `person`: People or individuals
- `concept`: Abstract concepts or ideas
- `skill`: Abilities or capabilities
- `topic`: Subject areas or topics
- `project`: Projects or initiatives
- `tool`: Tools or software
- `paper`: Research papers or documents
- `book`: Books or publications
- `event`: Events or occurrences
- `organization`: Companies or groups
- `idea`: Ideas or proposals
- `task`: Tasks or activities
- `goal`: Goals or objectives
- `emotion`: Emotional states
- `habit`: Habits or routines

## Relationship Types

- `enables`: One thing enables another
- `causes`: One thing causes another
- `improves`: One thing improves another
- `depends_on`: One thing depends on another
- `contradicts`: One thing contradicts another
- `learned_from`: Learning relationship
- `used_in`: Usage relationship
- `part_of`: Part-of relationship
- `related_to`: General relatedness
- `precedes`: Temporal precedence
- `follows`: Temporal succession
- `inspired_by`: Inspiration relationship
- `supports`: Support relationship
- `applied_to`: Application relationship

## Error Responses

All endpoints return error responses in the following format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

Common HTTP status codes:
- `400`: Bad Request - Invalid input parameters
- `404`: Not Found - Resource not found
- `413`: Payload Too Large - File size exceeds limit
- `500`: Internal Server Error - Server-side error
- `503`: Service Unavailable - Required service not available

## Rate Limiting

Currently not implemented, but will be added in future versions.

## Pagination

List endpoints support pagination using `limit` and `offset` parameters.

## WebSocket Support

Real-time updates will be supported via WebSocket in future versions.

## SDKs

Client SDKs will be provided for:
- Python
- JavaScript/TypeScript
- React components

## Examples

### Python Client Example

```python
import requests

# Ingest a file
with open('document.pdf', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/api/v1/ingest/file', files=files)
    print(response.json())

# Query the knowledge graph
query_data = {
    "query": "What is machine learning?",
    "max_hops": 3
}
response = requests.post('http://localhost:8000/api/v1/query', json=query_data)
print(response.json())
```

### JavaScript Client Example

```javascript
// Query the knowledge graph
const response = await fetch('http://localhost:8000/api/v1/query', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        query: 'What is machine learning?',
        max_hops: 3
    })
});

const result = await response.json();
console.log(result.answer);
```
