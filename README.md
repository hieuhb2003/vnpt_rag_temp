# Enterprise RAG System with Multi-Agent Architecture

A production-ready Retrieval-Augmented Generation (RAG) system for internal knowledge base and customer support. Features a multi-agent architecture with tree-structured document indexing, hybrid retrieval, and two-tier verification.

## Features

- **Multi-Agent Architecture**: Router, Planner, Retriever, Synthesizer, Verifier, and Grader agents
- **Tree-Structured Document Indexing**: Preserves document hierarchy for better context
- **Hybrid Retrieval**: Combines vector search (semantic) + BM25 (keyword)
- **Two-Tier Verification**: Fast similarity check + LLM-based groundedness evaluation
- **Multi-Level Caching**: Semantic cache, embedding cache, and retrieval cache
- **Tech Stack**: FastAPI, LangGraph, Qdrant, PostgreSQL, Redis, MinIO

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Gateway (FastAPI)                     │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                      Agent Orchestrator                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  Router  │→ │ Planner  │→ │Retriever │→ │   Synthesizer    │ │
│  │  Agent   │  │  Agent   │  │  Agent   │  │      Agent       │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
│                                    │                 │           │
│                              ┌─────┴─────┐    ┌─────┴─────┐     │
│                              │  Verifier │    │  Grader   │     │
│                              │   Agent   │    │   Agent   │     │
│                              └───────────┘    └───────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Anthropic API key or OpenAI API key

### 1. Clone and Configure

```bash
# Clone the repository
git clone <repository-url>
cd tmp_vnpt_rag

# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env
```

### 2. Start Services

```bash
# Start all services (API, Qdrant, PostgreSQL, Redis, MinIO)
docker-compose up -d

# Check service status
docker-compose ps
```

### 3. Verify Setup

```bash
# Test API health
curl http://localhost:8000/health

# Access API documentation
open http://localhost:8000/docs

# Access Qdrant dashboard
open http://localhost:6333/dashboard

# Access MinIO console
open http://localhost:9001
```

## Project Structure

```
rag-system/
├── src/
│   ├── main.py                 # FastAPI application entry
│   ├── config/                 # Configuration and settings
│   ├── api/                    # API routes and middleware
│   ├── agents/                 # LangGraph agents
│   ├── tools/                  # LangChain tools
│   ├── indexing/               # Document parsing and indexing
│   ├── storage/                # Database connections (Qdrant, PostgreSQL, Redis, MinIO)
│   ├── models/                 # Pydantic data models
│   └── utils/                  # Logging and utilities
├── scripts/                    # Utility scripts
├── tests/                      # Unit and integration tests
├── docs/                       # Documentation
├── docker-compose.yml          # Container orchestration
├── Dockerfile                  # API container image
├── pyproject.toml              # Python dependencies
└── CLAUDE.md                   # Project documentation for Claude Code
```

## Environment Variables

Key environment variables (see `.env.example` for full list):

```bash
# LLM Configuration
LLM_PROVIDER=anthropic  # or openai
ANTHROPIC_API_KEY=sk-ant-...
LLM_MODEL=claude-3-5-sonnet-20241022

# Embedding Configuration
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536

# Database Connections
QDRANT_HOST=qdrant
POSTGRES_HOST=postgres
REDIS_HOST=redis
MINIO_HOST=minio
```

## Development

### Running Tests

```bash
# Run all tests
docker-compose exec api pytest

# Run with coverage
docker-compose exec api pytest --cov=src --cov-report=html

# Run specific test file
docker-compose exec api pytest tests/unit/test_vector_store.py
```

### Code Quality

```bash
# Format code with black
docker-compose exec api black src/ tests/

# Lint with ruff
docker-compose exec api ruff check src/ tests/

# Type check with mypy
docker-compose exec api mypy src/
```

### Database Initialization

```bash
# Initialize PostgreSQL schema
docker exec -i rag-postgres psql -U raguser -d ragdb < scripts/init_postgres.sql

# Or run the init script
docker-compose exec api python scripts/init_db.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/` | GET | API information |
| `/api/v1/query` | POST | Query the RAG system |
| `/api/v1/documents` | POST | Upload and index documents |
| `/docs` | GET | Swagger UI documentation |

## Service URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| API | http://localhost:8000 | - |
| API Docs | http://localhost:8000/docs | - |
| Qdrant Dashboard | http://localhost:6333/dashboard | - |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin123 |
| PostgreSQL | localhost:5432 | raguser / ragpass123 |
| Redis | localhost:6379 | - |

## Documentation

- [CLAUDE.md](CLAUDE.md) - Project architecture and implementation guide
- [docs/api.md](docs/api.md) - API documentation
- [docs/architecture.md](docs/architecture.md) - System architecture details
- [docs/deployment.md](docs/deployment.md) - Deployment guide

## Implementation Status

- ✅ TODO-01: Project Setup (Docker, dependencies, basic structure)
- ⏳ TODO-02: Storage Layer (Database connections, vector store)
- ⏳ TODO-03: Indexing Pipeline (Document parsing, chunking, embedding)
- ⏳ TODO-04: Tools (Individual tool implementations)
- ⏳ TODO-05: Agents (Agent definitions and orchestrator)
- ⏳ TODO-06: API (FastAPI routes and middleware)
- ⏳ TODO-07: Testing (Unit and integration tests)
- ⏳ TODO-08: Optimization (Caching, performance tuning)

## Contributing

1. Follow the existing code structure and patterns
2. Write tests for new features
3. Update documentation as needed
4. Follow PEP 8 with Black formatting

## License

[Specify your license here]

## Support

For issues and questions, please open a GitHub issue.
