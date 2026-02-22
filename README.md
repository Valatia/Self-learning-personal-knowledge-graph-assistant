# REXI - Self-Learning Personal Knowledge Graph Assistant

A sophisticated AI system that transforms personal data into structured knowledge with reasoning capabilities.

## Overview

REXI is a research-level neuro-symbolic AI system that acts as a persistent external brain, continuously ingesting unstructured personal data and converting it into an evolving knowledge graph with multi-hop reasoning, temporal memory, and insight generation capabilities.

## Features

- **Multi-source Data Ingestion**: PDFs, markdown notes, emails, chat logs, code repositories, audio, and more
- **Knowledge Graph Construction**: Automatic entity and relation extraction with temporal modeling
- **Hybrid Retrieval**: Combines graph traversal, vector similarity, and keyword search
- **Advanced Reasoning**: Multi-hop, causal, temporal, and analogical reasoning
- **Self-Learning**: Knowledge gap detection, exploration suggestions, and memory evolution
- **Explainable AI**: Evidence-based answers with source citations
- **Interactive Visualization**: Graph explorer, timeline views, and analytics dashboard

## Architecture

- **Backend**: Python + FastAPI
- **AI/ML**: LLMs, Sentence Transformers, spaCy, HuggingFace models
- **Databases**: Neo4j (graph), Qdrant (vector), PostgreSQL (metadata)
- **Frontend**: React + Next.js + D3.js
- **Infrastructure**: Docker + Kubernetes

## Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd rexi

# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set up frontend
cd frontend
npm install
npm run dev

# Start backend services
cd ../backend
uvicorn main:app --reload
```

## Development Status

This project is currently in active development. See the [implementation plan](docs/implementation-plan.md) for detailed progress.

## Documentation

- [Product Requirements](docs/prd.md)
- [Technical Design](docs/design.md)
- [Tech Stack](docs/tech-stack.md)
- [API Documentation](docs/api.md)
- [Development Guide](docs/development.md)

## License

MIT License - see LICENSE file for details.
