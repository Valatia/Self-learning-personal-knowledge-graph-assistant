# REXI Development Guide

## Getting Started

This guide covers setting up the REXI development environment and contributing to the project.

## Prerequisites

- Python 3.11+
- Node.js 18+
- Docker and Docker Compose
- Git

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/rexi.git
cd rexi
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### 3. Set Up Frontend

```bash
cd frontend
npm install
```

### 4. Start Development Services

#### Option A: Using Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

#### Option B: Manual Setup

Start each service individually:

```bash
# Start Neo4j
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/neo4j_password \
  -e NEO4J_PLUGINS='["apoc", "graph-data-science"]' \
  neo4j:5.15-community

# Start Qdrant
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  qdrant/qdrant:v1.7.0

# Start PostgreSQL
docker run -d --name postgres \
  -p 5432:5432 \
  -e POSTGRES_DB=rexi \
  -e POSTGRES_USER=rexi \
  -e POSTGRES_PASSWORD=rexi_password \
  postgres:15

# Start Redis
docker run -d --name redis \
  -p 6379:6379 \
  redis:7-alpine
```

### 5. Configure Environment

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# Database connections
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

QDRANT_HOST=localhost
QDRANT_PORT=6333

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rexi
POSTGRES_USER=rexi
POSTGRES_PASSWORD=your_postgres_password

REDIS_HOST=localhost
REDIS_PORT=6379

# AI services
OPENAI_API_KEY=your_openai_api_key
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=gpt-3.5-turbo
```

### 6. Start Development Servers

```bash
# Backend (in root directory)
uvicorn rexi.api.main:app --reload --host 0.0.0.0 --port 8000

# Frontend (in frontend directory)
cd frontend
npm run dev
```

## Project Structure

```
rexi/
├── src/
│   └── rexi/
│       ├── api/           # FastAPI endpoints
│       ├── agents/        # AI agents
│       ├── core/          # Core business logic
│       ├── models/        # Data models
│       ├── services/      # External service integrations
│       ├── utils/         # Utility functions
│       └── config/        # Configuration
├── frontend/             # React frontend
├── tests/               # Test suite
├── docs/                # Documentation
├── scripts/             # Utility scripts
├── data/                # Data directory
└── docker-compose.yml    # Development environment
```

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Follow the coding standards
- Write tests for new features
- Update documentation

### 3. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/rexi --cov-report=html

# Run specific test file
pytest tests/test_services.py
```

### 4. Code Quality

```bash
# Format code
black src/ tests/
ruff check --fix src/ tests/
mypy src/

# Run pre-commit hooks
pre-commit run --all-files
```

### 5. Commit Changes

```bash
git add .
git commit -m "feat: add new feature description"
```

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
# Create pull request on GitHub
```

## Coding Standards

### Python

- Use Black for code formatting
- Follow PEP 8 style guide
- Use type hints
- Write docstrings for all functions and classes
- Maximum line length: 88 characters

### JavaScript/TypeScript

- Use Prettier for formatting
- Follow ESLint rules
- Use TypeScript for type safety
- Maximum line length: 100 characters

### Documentation

- Use Markdown for documentation
- Include examples in API docs
- Keep README files up to date

## Testing

### Test Structure

```
tests/
├── test_services.py     # Service layer tests
├── test_core.py        # Core logic tests
├── test_api.py         # API endpoint tests
└── test_integration.py  # Integration tests
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration    # Integration tests only
pytest -m "not slow"    # Skip slow tests

# Run with coverage
pytest --cov=src/rexi --cov-report=term-missing
```

### Test Coverage

- Aim for >80% code coverage
- Focus on critical paths
- Test edge cases and error conditions

## Debugging

### Backend Debugging

```bash
# Enable debug mode
export DEBUG=true

# Run with debugger
python -m pdb -m uvicorn rexi.api.main:app --reload
```

### Frontend Debugging

```bash
# Start with debugging
npm run dev

# Use browser dev tools
# Set breakpoints in VS Code
```

### Database Debugging

#### Neo4j Browser

Access at: http://localhost:7474

#### Qdrant Management

```bash
# Check collection info
curl http://localhost:6333/collections

# Search vectors
curl -X POST http://localhost:6333/collections/knowledge_embeddings/points/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, ...], "limit": 10}'
```

## Performance Monitoring

### Backend Performance

```bash
# Monitor with built-in tools
curl http://localhost:8000/metrics

# Check response times
curl -w "@curl-format.txt" http://localhost:8000/health
```

### Database Performance

#### Neo4j Queries

```bash
# Enable query logging
# In Neo4j configuration
dbms.logs.query.enabled=true
dbms.logs.query.threshold=1s
```

#### Vector Search Performance

Monitor:
- Search latency
- Index size
- Memory usage

## Common Issues

### Port Conflicts

If ports are already in use:

```bash
# Check what's using ports
netstat -tulpn | grep :8000
netstat -tulpn | grep :3000

# Kill processes
kill -9 <PID>

# Or change ports in .env
API_PORT=8001
```

### Memory Issues

```bash
# Monitor memory usage
docker stats

# Clear Docker cache
docker system prune -a
```

### Database Connection Issues

```bash
# Check Neo4j status
curl http://localhost:7474/db/manage/server/jmx/domain=org.neo4j

# Check Qdrant status
curl http://localhost:6333/health
```

## Deployment

### Development Deployment

```bash
# Build and deploy to development
docker-compose -f docker-compose.dev.yml up -d
```

### Production Deployment

```bash
# Build production images
docker build -t rexi-backend .
docker build -t rexi-frontend ./frontend

# Deploy with production compose
docker-compose -f docker-compose.prod.yml up -d
```

## Contributing Guidelines

### Before Contributing

1. Read the project documentation
2. Set up development environment
3. Run existing tests to ensure they pass
4. Create an issue for your feature/bug fix

### Making Changes

1. Fork the repository
2. Create feature branch
3. Implement your changes
4. Add tests
5. Update documentation
6. Ensure all tests pass
7. Submit pull request

### Pull Request Requirements

- Clear description of changes
- Link to relevant issues
- Tests pass
- Code follows style guidelines
- Documentation updated

### Code Review Process

1. Automated checks pass
2. Manual review by maintainers
3. Address feedback
4. Approval and merge

## Resources

### Documentation

- [API Documentation](api.md)
- [Architecture Overview](../Project_description/design.md)
- [Technical Stack](../Project_description/tech-stack.md)

### Tools

- [Neo4j Browser](http://localhost:7474)
- [FastAPI Docs](http://localhost:8000/docs)
- [React Dev Tools](chrome://extensions/)

### Community

- GitHub Issues: Report bugs and request features
- GitHub Discussions: Ask questions and share ideas
- Wiki: Additional documentation and examples

## Troubleshooting

### Common Solutions

1. **Import errors**: Ensure virtual environment is activated
2. **Database connection**: Check Docker containers are running
3. **Permission errors**: Check file permissions in data directory
4. **Memory errors**: Increase Docker memory limits

### Getting Help

1. Check existing issues and documentation
2. Search troubleshooting guides
3. Create new issue with detailed information
4. Join community discussions
