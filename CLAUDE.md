# CLAUDE.md - Enterprise RAG System with Multi-Agent Architecture

## Project Overview

This project implements a production-ready RAG (Retrieval-Augmented Generation) system for internal knowledge base and customer support. The system uses a multi-agent architecture with tree-structured document indexing, hybrid retrieval, and two-tier verification.

## Architecture Summary

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
                                │
┌─────────────────────────────────────────────────────────────────┐
│                         Tool Layer                               │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────┐  │
│  │   Query    │ │  Hybrid    │ │    Tree    │ │   Verify     │  │
│  │  Rewriter  │ │   Search   │ │  Navigator │ │  Groundedness│  │
│  └────────────┘ └────────────┘ └────────────┘ └──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                       Storage Layer                              │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────┐  │
│  │  Qdrant    │ │ PostgreSQL │ │   Redis    │ │  MinIO/S3    │  │
│  │  (Vector)  │ │  (Metadata)│ │  (Cache)   │ │  (Documents) │  │
│  └────────────┘ └────────────┘ └────────────┘ └──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Tech Stack

- **Language**: Python 3.11+
- **API Framework**: FastAPI with async support
- **Agent Framework**: LangGraph (for complex workflows) + LangChain (tools)
- **Vector Database**: Qdrant (hybrid search support)
- **Relational Database**: PostgreSQL with pgvector extension
- **Cache**: Redis (semantic cache + embedding cache)
- **Object Storage**: MinIO (S3-compatible) for documents
- **Embedding Model**: Configurable (default: text-embedding-3-small)
- **LLM**: Configurable (default: Claude 3.5 Sonnet via Anthropic API)
- **Containerization**: Docker + Docker Compose

## Project Structure

```
rag-system/
├── CLAUDE.md                    # This file
├── docker-compose.yml           # Container orchestration
├── Dockerfile                   # Main application image
├── .env.example                 # Environment template
├── pyproject.toml              # Python dependencies
├── src/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py         # Pydantic settings from env
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── query.py        # Query endpoints
│   │   │   ├── documents.py    # Document management
│   │   │   └── health.py       # Health checks
│   │   └── middleware/
│   │       ├── __init__.py
│   │       └── rate_limit.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── orchestrator.py     # LangGraph workflow
│   │   ├── router_agent.py     # Query routing
│   │   ├── planner_agent.py    # Query decomposition
│   │   ├── retriever_agent.py  # Retrieval execution
│   │   ├── synthesizer_agent.py # Answer generation
│   │   ├── verifier_agent.py   # Groundedness check
│   │   └── grader_agent.py     # Relevance grading
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── query_rewriter.py
│   │   ├── query_decomposer.py
│   │   ├── hybrid_search.py
│   │   ├── tree_navigator.py
│   │   ├── section_retriever.py
│   │   ├── cross_reference.py
│   │   ├── synthesize_answer.py
│   │   ├── verify_groundedness.py
│   │   └── check_freshness.py
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── document_parser.py   # Parse docs to tree structure
│   │   ├── chunker.py           # Semantic chunking
│   │   ├── embedder.py          # Embedding generation
│   │   ├── tree_builder.py      # Build document trees
│   │   └── index_manager.py     # Multi-level index management
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── vector_store.py      # Qdrant operations
│   │   ├── metadata_store.py    # PostgreSQL operations
│   │   ├── cache.py             # Redis caching
│   │   └── document_store.py    # MinIO operations
│   ├── models/
│   │   ├── __init__.py
│   │   ├── document.py          # Document schemas
│   │   ├── query.py             # Query schemas
│   │   ├── chunk.py             # Chunk schemas
│   │   └── response.py          # Response schemas
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       ├── metrics.py
│       └── helpers.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   └── integration/
├── scripts/
│   ├── init_db.py              # Database initialization
│   ├── index_documents.py      # Bulk indexing script
│   └── benchmark.py            # Performance benchmarks
└── docs/
    ├── api.md
    ├── deployment.md
    └── architecture.md
```

## Key Design Decisions

### 1. Tree-Structured Document Indexing
Documents are parsed into hierarchical trees preserving heading structure:
- **Document Level**: Title + auto-generated summary → for topic routing
- **Section Level**: H1/H2 + first paragraph → for navigation
- **Chunk Level**: 400-512 token chunks → for precise retrieval

### 2. Multi-Level Indexing
Three separate index collections in Qdrant:
- `documents`: Document-level embeddings
- `sections`: Section-level embeddings  
- `chunks`: Chunk-level embeddings with parent references

### 3. Hybrid Retrieval
Combine vector search + BM25 keyword search:
- Vector: semantic similarity
- BM25: exact term matching (product names, error codes)
- Fusion: Reciprocal Rank Fusion (RRF)

### 4. Two-Tier Verification (DoorDash Pattern)
- **Tier 1**: Fast semantic similarity check (cosine > 0.85)
- **Tier 2**: LLM-based groundedness evaluation (only if Tier 1 fails)

### 5. Caching Strategy
```
Query → Check Semantic Cache (Redis)
  ↓ (miss)
Embed → Check Embedding Cache (Redis)  
  ↓ (miss)
Retrieve → Check Retrieval Cache (Redis, 30min TTL)
  ↓ (miss)
Generate → Store in Semantic Cache
```

## Environment Variables

All configuration via environment variables (see `.env.example`):

```bash
# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# LLM
LLM_PROVIDER=anthropic  # or openai
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
LLM_MODEL=claude-3-5-sonnet-20241022

# Embeddings  
EMBEDDING_PROVIDER=openai  # or local
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536

# Qdrant
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_API_KEY=

# PostgreSQL
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=ragdb
POSTGRES_USER=raguser
POSTGRES_PASSWORD=

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=

# MinIO
MINIO_HOST=minio
MINIO_PORT=9000
MINIO_ACCESS_KEY=
MINIO_SECRET_KEY=
MINIO_BUCKET=documents

# Indexing
CHUNK_SIZE=512
CHUNK_OVERLAP=50
MAX_TREE_DEPTH=5

# Retrieval
HYBRID_ALPHA=0.7  # 0=keyword, 1=vector
TOP_K_DOCUMENTS=5
TOP_K_SECTIONS=10
TOP_K_CHUNKS=20

# Caching
CACHE_EMBEDDING_TTL=3600
CACHE_RETRIEVAL_TTL=1800
CACHE_SEMANTIC_TTL=3600
SEMANTIC_CACHE_THRESHOLD=0.85

# Verification
GROUNDEDNESS_THRESHOLD=0.85
ENABLE_TIER2_VERIFICATION=true
```

## Coding Standards

### Python Style
- Use Python 3.11+ features (type hints, match statements)
- Follow PEP 8 with Black formatting (line length 88)
- Use Pydantic v2 for all data models
- Async/await for all I/O operations
- Dependency injection via FastAPI's Depends

### Error Handling
```python
from src.utils.exceptions import RAGException, RetryableError

try:
    result = await tool.execute()
except RetryableError as e:
    # Automatic retry with exponential backoff
    result = await retry_with_backoff(tool.execute, max_retries=3)
except RAGException as e:
    # Log and return graceful degradation
    logger.error(f"Tool failed: {e}")
    return fallback_response()
```

### Logging
```python
from src.utils.logging import get_logger

logger = get_logger(__name__)
logger.info("Processing query", extra={"query_id": query_id, "user_id": user_id})
```

### Testing
- Unit tests: pytest with pytest-asyncio
- Integration tests: testcontainers for databases
- Coverage target: 80%+


## Git workflow & commit policy (GitHub)

- Repo https://github.com/hieuhb2003/vnpt_rag_temp.git

### Branching
- Never commit directly to `main`.
- Create a new branch per task: `feat/<short-scope>` or `fix/<short-scope>`.

### Before commit
- Always run: `pytest -q` (or the project test command) and fix failures.
- Show `git diff` summary in the final response before committing.
- Stage only relevant files (avoid unrelated formatting/lockfile noise unless required).

### Commit message standard (Conventional Commits)
Use: <type>(<scope>): <subject>
- type: feat | fix | docs | refactor | perf | test | build | ci | chore
- subject: imperative, <= 72 chars, no trailing period
- body: explain “what/why”, include breaking changes if any
- footer: reference issue/PR if provided
- Commit each feature not, push one commit with alot of changes

### Commit steps
1. `git status`
2. `git diff`
3. `git add <files>`
4. `git commit -m "<type>(<scope>): <subject>"` (add body via editor if needed)
5. `git push -u origin <branch>`

### PR (if requested)
- Create a PR from the branch to `main` with a clear description and testing notes.

### Attribution
- If you add any “Generated with …” or “Co-Authored-By …” trailers by default, keep/remove them based on this repo policy: <WRITE YOUR POLICY HERE>.


## Common Commands

```bash
# Development
docker-compose up -d                    # Start all services
docker-compose logs -f api              # Follow API logs
docker-compose exec api pytest          # Run tests

# Indexing
docker-compose exec api python scripts/index_documents.py --path /data/docs

# Database
docker-compose exec api python scripts/init_db.py

# Benchmarks
docker-compose exec api python scripts/benchmark.py --queries 100
```

## Implementation Order

Follow the TODO files in order:
1. `TODO-01-project-setup.md` - Project structure, Docker, dependencies
2. `TODO-02-storage-layer.md` - Database connections, vector store
3. `TODO-03-indexing-pipeline.md` - Document parsing, chunking, embedding
4. `TODO-04-tools.md` - Individual tool implementations
5. `TODO-05-agents.md` - Agent definitions and orchestrator
6. `TODO-06-api.md` - FastAPI routes and middleware
7. `TODO-07-testing.md` - Unit and integration tests
8. `TODO-08-optimization.md` - Caching, performance tuning

## Important Notes for Claude Code

1. **Always check environment variables** - Never hardcode credentials
2. **Use async everywhere** - All database and API calls must be async
3. **Implement graceful degradation** - If a tool fails, provide fallback
4. **Log extensively** - Include query_id in all log messages for tracing
5. **Cache aggressively** - Check cache before any expensive operation
6. **Type everything** - Full type hints for all functions
7. **Test incrementally** - Write tests as you implement each component
8. **Docker-first** - All code must work in containerized environment
