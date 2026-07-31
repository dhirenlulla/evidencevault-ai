<div align="center">

# EvidenceVault AI

### Production-Aware Document Intelligence and Citation-Ready RAG Platform

A backend-first platform for secure PDF ingestion, document classification, page-aware extraction, deterministic chunking, durable PostgreSQL persistence, hybrid retrieval, cross-encoder reranking, and measurable Retrieval-Augmented Generation.

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Async_API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC244C)](https://qdrant.tech/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-220_Passing-brightgreen)](#testing)
[![Status](https://img.shields.io/badge/Status-Active_Development-orange)](#development-status)

[Overview](#project-overview) ·
[Features](#implemented-features) ·
[Architecture](#system-architecture) ·
[Setup](#local-development-setup) ·
[API](#available-api-endpoints) ·
[Roadmap](#roadmap)

</div>

---

## Project Overview

**EvidenceVault AI** is a production-aware document intelligence platform designed to help teams securely ingest enterprise documents, preserve page-level evidence, and answer questions using grounded, measurably-evaluated Retrieval-Augmented Generation (RAG).

The platform is being built for document-heavy workflows such as:

- Legal and contract analysis
- HR policies and employee handbooks
- Compliance and audit documentation
- Research papers and technical reports
- Finance and operations documents
- Internal knowledge bases

Unlike a basic "chat with PDF" prototype, EvidenceVault AI is being developed as a production-oriented system with explicit lifecycle states, secure file handling, PDF classification, page-aware extraction, deterministic chunks, transactional persistence, database migrations, embeddings, hybrid dense + lexical retrieval fused with Reciprocal Rank Fusion, cross-encoder reranking, LLM-grounded generation, a dedicated retrieval evaluation framework, and automated tests at every layer.

> **Current scope:** Secure ingestion, classification, extraction, chunking, durable PostgreSQL persistence, embedding generation, Qdrant vector indexing, dense retrieval, BM25 lexical retrieval, RRF-fused hybrid retrieval, cross-encoder reranking, prompt construction, Groq LLM generation, and a standalone retrieval evaluation framework (Recall/Precision/Hit Rate/MRR/NDCG + latency stats) are all implemented and unit-tested. **Streaming generation, a live query/chat API endpoint wiring the full retrieval → rerank → generate pipeline together, page-level citations, and observability come next.**

---

## Problem Statement

Organizations store critical knowledge across contracts, policies, reports, manuals, research papers, and standard operating procedures. Finding reliable answers inside these files is slow, while general-purpose LLMs may generate unsupported responses.

EvidenceVault AI aims to solve this by building a pipeline that:

1. Validates and stores uploaded PDF documents.
2. Records metadata and lifecycle state in PostgreSQL.
3. Detects encrypted, malformed, empty, scanned, and partially extractable PDFs.
4. Extracts text page by page while preserving citation boundaries.
5. Cleans extracted text conservatively.
6. Generates overlapping, citation-ready chunks.
7. Produces deterministic UUIDs and content hashes.
8. Persists chunks transactionally and idempotently.
9. Generates embeddings and indexes chunks in Qdrant.
10. Retrieves evidence using dense semantic search fused with BM25 keyword search via Reciprocal Rank Fusion.
11. Reranks the fused candidate set with a cross-encoder for higher-precision ordering.
12. Produces grounded answers with page-level citations.
13. Measures retrieval quality (Recall, Precision, Hit Rate, MRR, NDCG) and refuses unsupported claims.

---

## Why This Project Is Different

Many beginner RAG applications follow this flow:

```text
Upload PDF → Split text → Store vectors → Ask an LLM
```

EvidenceVault AI introduces production-oriented concerns first:

- Secure filename sanitization
- File-extension, MIME-type, and PDF-signature validation
- Configurable upload-size enforcement
- Asynchronous chunked file writes
- Partial-file and compensating cleanup
- Encrypted and malformed PDF detection
- Image-only and OCR-required classification
- Page-level extraction and source metadata
- Deterministic UUID5 chunk IDs
- SHA-256 chunk fingerprints
- Transactional chunk replacement
- Idempotent reprocessing
- Alembic migrations
- PostgreSQL constraints and indexes
- Dependency-aware health checks
- Dense retrieval decoupled from lexical retrieval, fused by rank rather than raw score (RRF)
- A two-stage retrieve-then-rerank pipeline (cheap bi-encoder shortlist, accurate cross-encoder reordering)
- A standalone, dependency-free retrieval evaluation framework, not just "it feels better"
- Automated unit and API tests for every service, using dependency-injected fakes instead of live models or databases

---

## Development Status

### Overall Progress

| Stage | Status | Summary |
|---|---:|---|
| Step 1 | ✅ Complete | FastAPI foundation, settings, OpenAPI, health API, and tests |
| Step 2 | ✅ Complete | Docker Compose infrastructure with PostgreSQL and Qdrant |
| Step 3 | ✅ Complete | Async SQLAlchemy, Alembic, Qdrant client, models, and dependency health checks |
| Step 4 | ✅ Complete | Secure PDF upload, local storage, repository, and document APIs |
| Step 5A | ✅ Complete | PyMuPDF installation and PDF inspection tooling |
| Step 5B | ✅ Complete | Page-level extraction and conservative text cleaning |
| Step 5C | ✅ Complete | Text, partial, scanned, empty, encrypted, and malformed classification |
| Step 5D | ✅ Complete | Processing workflow and PostgreSQL status transitions |
| Step 5E | ✅ Complete | Deterministic page-aware chunking with overlap and metadata |
| Step 5F | ✅ Complete | Transactional chunk persistence, processing API, and paginated chunk-retrieval API |
| Step 6 | ✅ Complete | Embedding generation and Qdrant vector indexing |
| Step 7 | ✅ Complete | Dense retrieval, prompt construction, and Groq LLM generation |
| Step 8 | ✅ Complete | BM25 lexical retrieval and hybrid retrieval orchestration |
| Step 9 | ✅ Complete | Reciprocal Rank Fusion (RRF) — fuses dense + lexical rankings |
| Step 10 | ✅ Complete | Cross-encoder reranking (top-K reordering of the fused candidate set) |
| Step 11 | ✅ Complete | Retrieval evaluation framework (Recall, Precision, Hit Rate, MRR, NDCG, latency) |
| Step 12 | ⏳ Planned | Streaming generation |
| Step 13 | ⏳ Planned | Wire hybrid retrieval → reranker → generation into one live query/chat endpoint |
| Step 14 | ⏳ Planned | Page-level citations and "not enough evidence" fallback |
| Step 15 | ⏳ Planned | Observability (structured logging, tracing, metrics) |

> Note: Hybrid retrieval, RRF, reranking, and evaluation are implemented and independently unit-tested, but **not yet wired into a single live API endpoint** — each service is built and tested in isolation first, then composed together in a later integration phase. This is intentional: it keeps every piece independently testable before it's combined into the full pipeline.

### Verified Development Baseline

The current implementation has been validated with:

- **220 passing automated tests**, zero failing, zero skipped
- A real **16-page PDF**
- **64,210 extracted characters**
- **10,096 extracted words**
- **73 deterministic chunks**
- Matching PostgreSQL `chunk_count` and physical chunk-row counts
- Stable chunk IDs across repeated runs
- Idempotent reprocessing without duplicated rows
- Every retrieval-layer service (embedding, dense retrieval, BM25, RRF fusion, reranking, evaluation) tested against fakes/doubles, with zero dependency on a live database, Qdrant instance, or downloaded ML model at test time

---

## Implemented Features

### 1. FastAPI Foundation

- Async FastAPI backend
- Versioned `/api/v1` routes
- Environment-based settings with `pydantic-settings`
- Swagger UI and OpenAPI schema
- Application lifespan cleanup
- Root and health endpoints
- Automated Pytest suite

### 2. Dockerized Infrastructure

Docker Compose provisions:

- **PostgreSQL 16** for relational application data
- **Qdrant** for the vector-search layer
- Persistent Docker volumes
- Isolated service networking
- PostgreSQL health checks

The FastAPI backend currently runs locally during development.

### 3. PostgreSQL and Alembic

- Async SQLAlchemy 2.0
- Psycopg 3 async driver
- Alembic migrations
- `documents` table
- `document_chunks` table
- UUID primary keys
- Foreign-key integrity
- `ON DELETE CASCADE`
- Unique chunk-order constraints
- Document-and-page indexes
- Commit, rollback, and refresh handling

### 4. Secure PDF Upload

The upload service supports:

- Filename sanitization
- `.pdf` extension validation
- MIME-type validation
- `%PDF-` signature verification
- Configurable maximum file size
- Chunked asynchronous disk writes
- UUID-based internal filenames
- Partial-file cleanup after failure
- File deletion if database persistence fails

### 5. Document APIs

Implemented endpoints support:

- Uploading a PDF
- Listing documents
- Retrieving a document by UUID
- Running the processing pipeline (classification → extraction → chunking → persistence)
- Retrieving paginated, persisted chunks
- Pagination validation
- Structured HTTP errors

### 6. PDF Inspection and Extraction

PyMuPDF is used to:

- Open and inspect PDFs
- Count pages
- Extract page-level text
- Preserve page numbers
- Approximate natural reading order
- Calculate character and word counts
- Detect text-empty pages
- Produce terminal inspection reports

### 7. Conservative Text Cleaning

The cleaning layer handles:

- Unicode NFC normalization
- Common ligature conversion
- Non-breaking spaces
- Soft hyphens
- Zero-width spaces
- Control characters
- Repeated horizontal whitespace
- Excessive blank lines
- Windows and Unix newline normalization

### 8. PDF Classification

Documents are classified as:

| Classification | Meaning |
|---|---|
| `text_based` | Every page contains extractable text |
| `partially_extractable` | Some pages contain text while others do not |
| `scanned_or_image_only` | No usable text, but large page images are present |
| `empty` | No extractable text or significant page images |
| `encrypted` | A password is required |
| `malformed` | The PDF cannot be parsed safely |

Page-level analysis records:

- Text character count
- Word count
- Image count
- Estimated image coverage
- Page classification
- OCR requirement

### 9. Processing State Machine

```text
uploaded
   ↓
processing
   ├── extracted
   ├── extracted_with_warnings
   ├── ocr_required
   ├── password_required
   └── failed
```

After durable chunk persistence:

```text
extracted
   ↓
chunked
```

### 10. Page-Aware Deterministic Chunking

Default settings:

| Option | Default |
|---|---:|
| Maximum chunk size | 1,200 characters |
| Overlap | 200 characters |
| Minimum page content | 40 characters |

Behavior:

- Never crosses a page boundary
- Prefers paragraph boundaries
- Falls back to sentence boundaries
- Falls back to word-aligned splitting
- Hard-splits only abnormally long tokens
- Preserves one page number per chunk
- Reports skipped empty or short pages
- Prevents consecutive duplicate chunks

Every chunk stores:

- Deterministic UUID
- Parent document UUID
- Global chunk index
- Page number
- Page-local chunk index
- Text
- Character count
- Word count
- SHA-256 content hash

### 11. Deterministic Chunk Identity

```text
document UUID
+ page number
+ page-local chunk index
+ SHA-256 text hash
= deterministic UUID5 chunk ID
```

Benefits:

- Stable IDs for unchanged content
- Changed IDs when content changes
- Safe retries
- Idempotent reprocessing
- Stable Qdrant point IDs
- Easier incremental indexing

### 12. Transactional Chunk Persistence

```text
DELETE previous chunks
INSERT complete new chunk set
UPDATE document status and chunk count
COMMIT
```

On failure:

```text
ROLLBACK
```

This prevents partial writes and duplicate chunk sets.

### 13. Embeddings and Vector Indexing

- Sentence-transformer embedding model (bi-encoder), lazily loaded and cached
- Configurable model name, batch size, device, and query instruction prefix
- Qdrant collection management and point upserts
- Deterministic point IDs derived from chunk UUIDs, keeping the vector index rebuildable from PostgreSQL

### 14. Dense Retrieval

- Embeds the query with the same model used at indexing time
- Searches Qdrant, scoped to a single document
- Returns ranked, similarity-scored chunks

### 15. BM25 Lexical Retrieval

- Classic BM25 scoring over the dense-retrieved candidate pool
- Complements dense retrieval by rewarding exact keyword/term overlap that embeddings can undervalue

### 16. Hybrid Retrieval with Reciprocal Rank Fusion (RRF)

- `HybridRetrievalService` orchestrates dense retrieval and BM25 lexical retrieval over the same candidate set
- `RRFService` fuses the two rankings by position (`1 / (k + rank)`), not by raw score — the industry-standard approach for combining retrievers whose scores live on incompatible scales
- Fully generic and reusable: `RRFService` fuses any ranked list of hashable identifiers, not just chunk IDs
- `k` is configurable (`rrf_k`, default 60)

### 17. Cross-Encoder Reranking

- `CrossEncoderReranker` re-scores a short candidate list by feeding `(query, chunk)` pairs jointly into a cross-encoder model — more accurate than bi-encoder similarity, deliberately used only on a small shortlist since it doesn't scale to a full corpus
- Configurable model name, device, and `top_k`
- Lazily loaded and cached, following the same pattern as the embedding service

### 18. Retrieval Evaluation Framework

- `RetrievalEvaluator`: computes Recall@K, Precision@K, Hit Rate@K, MRR, and NDCG@K for a single query against binary relevance labels
- `RetrievalMetricsAggregator`: averages metrics across a query set
- `compare_metrics`: before/after delta comparison between two retrieval configurations
- `LatencyTimer` and `compute_latency_stats`: wall-clock timing with mean/p50/p95 — a separate, reusable concern independent of relevance scoring
- Every metric operates on plain hashable IDs, with zero dependency on retrieval, embeddings, or the database — any ranked list can be evaluated

### 19. LLM Generation

- Provider-abstracted LLM service (currently Groq)
- Prompt builder assembles retrieved context into a grounded prompt
- Configurable temperature, max tokens, and timeout

---

## System Architecture

### Current Architecture

```mermaid
flowchart TD
    A[Client or Swagger UI] --> B[FastAPI API]
    B --> C[Secure Upload Service]
    C --> D[Local Upload Storage]
    C --> E[(PostgreSQL documents)]

    E --> F[Processing Workflow]
    D --> F
    F --> G[PDF Classifier]
    G --> H{Document Type}

    H -->|Text or Partial| I[Page-Aware Extraction]
    H -->|Image Only| J[OCR Required]
    H -->|Encrypted| K[Password Required]
    H -->|Malformed or Empty| L[Failed]

    I --> M[Text Cleaning]
    M --> N[Deterministic Chunker]
    N --> O[Transactional Chunk Repository]
    O --> P[(PostgreSQL document_chunks)]

    P --> Q[Embedding Service]
    Q --> R[(Qdrant Vector Index)]

    S[User Query] --> T[Dense Retrieval]
    S --> U[BM25 Lexical Retrieval]
    T --> V[RRF Fusion]
    U --> V
    V --> W[Cross-Encoder Reranker]
    W --> X[Prompt Builder]
    X --> Y[Groq LLM]
    Y --> Z[Answer]

    B --> AA[Health Service]
    AA --> E
    AA --> R
```

> `T` through `Z` are implemented and unit-tested as standalone services but not yet wired together behind a single API route — that composition is the next milestone.

### Current Ingestion Flow

```text
PDF upload
    ↓
Validate extension, MIME type, size, and signature
    ↓
Stream file to local storage
    ↓
Create PostgreSQL document record
    ↓
Classify PDF
    ↓
Extract and clean text page by page
    ↓
Generate page-aware overlapping chunks
    ↓
Create deterministic IDs and hashes
    ↓
Replace previous chunks transactionally
    ↓
Set document status to "chunked"
    ↓
Generate embeddings and upsert into Qdrant
```

### Current Retrieval Flow

```text
User question
    ↓
Dense retrieval (Qdrant, cosine similarity)  +  BM25 lexical retrieval (same candidate pool)
    ↓
Reciprocal Rank Fusion (fuse by rank, not raw score)
    ↓
Cross-encoder reranking (top-K re-scored jointly with the query)
    ↓
Prompt construction with retrieved context
    ↓
Groq LLM generation
```

### Planned: Full Query Pipeline (wired end-to-end)

```mermaid
flowchart TD
    D[User Question] --> E[Query Processing]
    E --> F[Dense Retrieval]
    E --> G[BM25 Retrieval]

    F --> H[Reciprocal Rank Fusion]
    G --> H
    H --> I[Cross-Encoder Reranker]
    I --> J[Top Evidence Chunks]
    J --> K[Grounded LLM]
    K --> L[Answer with Page Citations]
    L --> M[Faithfulness Evaluation]
```

---

## Technology Stack

### Implemented

| Layer | Technology |
|---|---|
| Language | Python 3.10 |
| API | FastAPI |
| Validation and settings | Pydantic, pydantic-settings |
| ORM | SQLAlchemy 2.0 async |
| PostgreSQL driver | Psycopg 3 |
| Migrations | Alembic |
| Relational database | PostgreSQL 16 |
| Vector database | Qdrant |
| PDF processing | PyMuPDF |
| Upload parsing | python-multipart |
| Async file I/O | aiofiles |
| Embeddings | sentence-transformers (bi-encoder) |
| Dense retrieval | Qdrant |
| Sparse retrieval | BM25 (rank_bm25) |
| Fusion | Reciprocal Rank Fusion (custom implementation) |
| Reranking | sentence-transformers CrossEncoder |
| LLM layer | Groq |
| Evaluation | Custom retrieval metrics (Recall, Precision, Hit Rate, MRR, NDCG) + latency stats |
| Testing | Pytest, HTTPX |
| Infrastructure | Docker Compose |

### Planned

| Layer | Planned Technology |
|---|---|
| Streaming | Server-sent events / chunked LLM streaming |
| Extended evaluation | RAGAS and faithfulness scoring |
| Frontend | Streamlit or Next.js |
| Cloud storage | AWS S3 |
| Deployment | Docker and AWS |
| Observability | Structured logging, tracing, metrics |

---

## Repository Structure

```text
evidencevault-ai/
├── backend/
│   ├── alembic/
│   │   └── versions/                  # Migration history
│   ├── app/
│   │   ├── api/routes/                # FastAPI routes
│   │   ├── clients/                   # External service clients
│   │   ├── core/                      # Settings, exceptions, dependency wiring
│   │   ├── db/models/                 # SQLAlchemy models
│   │   ├── repositories/              # Database access
│   │   ├── schemas/                   # Pydantic schemas
│   │   ├── services/                  # Processing, retrieval, reranking, evaluation logic
│   │   └── main.py                    # Application entry point
│   ├── scripts/                       # Development utilities
│   ├── tests/                         # Test suite
│   ├── uploads/                       # Local uploads; ignored by Git
│   ├── .env.example
│   ├── alembic.ini
│   ├── Dockerfile
│   ├── requirements.txt
│   └── requirements.lock.txt
├── docs/
├── evaluation/
├── frontend/
├── .env.example
├── compose.yaml
└── README.md
```

---

## Prerequisites

Install:

- Git
- Python 3.10
- Docker Desktop on Windows or macOS
- Docker Engine with the Compose plugin on Linux

### Verify Prerequisites

#### Windows PowerShell

```powershell
git --version
py -3.10 --version
docker --version
docker compose version
```

#### macOS

```bash
git --version
python3.10 --version
docker --version
docker compose version
```

#### Linux

```bash
git --version
python3.10 --version
docker --version
docker compose version
```

---

## Local Development Setup

### 1. Clone the Repository

#### Windows PowerShell

```powershell
git clone https://github.com/dhirenlulla/evidencevault-ai.git
Set-Location evidencevault-ai
```

#### macOS

```bash
git clone https://github.com/dhirenlulla/evidencevault-ai.git
cd evidencevault-ai
```

#### Linux

```bash
git clone https://github.com/dhirenlulla/evidencevault-ai.git
cd evidencevault-ai
```

### 2. Create Environment Files

#### Windows PowerShell

```powershell
Copy-Item .env.example .env
Copy-Item backend\.env.example backend\.env
```

#### macOS

```bash
cp .env.example .env
cp backend/.env.example backend/.env
```

#### Linux

```bash
cp .env.example .env
cp backend/.env.example backend/.env
```

### 3. Create and Activate a Virtual Environment

#### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

#### macOS

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

#### Linux

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

### 4. Start PostgreSQL and Qdrant

#### Windows PowerShell

```powershell
docker compose up -d
docker compose ps
```

#### macOS

```bash
docker compose up -d
docker compose ps
```

#### Linux

```bash
docker compose up -d
docker compose ps
```

### 5. Verify PostgreSQL

#### Windows PowerShell

```powershell
docker compose exec postgres pg_isready `
  -U evidencevault_user `
  -d evidencevault_db
```

#### macOS

```bash
docker compose exec postgres pg_isready \
  -U evidencevault_user \
  -d evidencevault_db
```

#### Linux

```bash
docker compose exec postgres pg_isready \
  -U evidencevault_user \
  -d evidencevault_db
```

### 6. Verify Qdrant

Open:

- REST API: `http://localhost:6333`
- Dashboard: `http://localhost:6333/dashboard`
- Health endpoint: `http://localhost:6333/healthz`

### 7. Apply Database Migrations

#### Windows PowerShell

```powershell
Set-Location backend
python -m alembic upgrade head
python -m alembic current
```

#### macOS

```bash
cd backend
python -m alembic upgrade head
python -m alembic current
```

#### Linux

```bash
cd backend
python -m alembic upgrade head
python -m alembic current
```

### 8. Start the FastAPI Server

Run from `backend`.

#### Windows PowerShell

```powershell
python -m uvicorn app.main:app --reload
```

#### macOS

```bash
python -m uvicorn app.main:app --reload
```

#### Linux

```bash
python -m uvicorn app.main:app --reload
```

Open:

- API root: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- OpenAPI schema: `http://127.0.0.1:8000/openapi.json`
- Health API: `http://127.0.0.1:8000/api/v1/health`

---

## Environment Configuration

### Root `.env`

```env
POSTGRES_USER=evidencevault_user
POSTGRES_PASSWORD=evidencevault_local_password
POSTGRES_DB=evidencevault_db
POSTGRES_PORT=5432

QDRANT_HTTP_PORT=6333
QDRANT_GRPC_PORT=6334
```

### Backend `.env`

```env
APP_NAME=EvidenceVault AI API
APP_VERSION=0.3.0
ENVIRONMENT=development
API_V1_PREFIX=/api/v1

DATABASE_URL=postgresql+psycopg://evidencevault_user:evidencevault_local_password@localhost:5432/evidencevault_db
SQL_ECHO=false

QDRANT_URL=http://localhost:6333
QDRANT_TIMEOUT_SECONDS=5
QDRANT_COLLECTION_NAME=evidencevault_chunks

UPLOAD_DIRECTORY=uploads
MAX_UPLOAD_SIZE_MB=20
UPLOAD_CHUNK_SIZE_BYTES=1048576

EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5
EMBEDDING_DEVICE=cpu

RETRIEVAL_TOP_K=5
RRF_K=60

RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_TOP_K=5
RERANKER_DEVICE=cpu

EVALUATION_DEFAULT_K=5

GROQ_API_KEY=
GROQ_MODEL_NAME=llama-3.3-70b-versatile
```

> Never commit production secrets. These values are local-development examples.

---

## Available API Endpoints

### Implemented

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | API entry point |
| `GET` | `/api/v1/health` | PostgreSQL and Qdrant health |
| `POST` | `/api/v1/documents/upload` | Validate, store, and register a PDF |
| `GET` | `/api/v1/documents` | List documents |
| `GET` | `/api/v1/documents/{document_id}` | Retrieve document metadata |
| `POST` | `/api/v1/documents/{document_id}/process` | Run classification, extraction, chunking, and persistence |
| `GET` | `/api/v1/documents/{document_id}/chunks` | Return paginated persisted chunks |

### Planned

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/documents/{document_id}/query` | Run the full retrieve → fuse → rerank → generate pipeline and return a grounded, cited answer |
| `GET` | `/api/v1/documents/{document_id}/evaluate` | Run the retrieval evaluation framework against a labeled query set |

---

## API Usage Examples

### Upload a PDF

#### Windows PowerShell

```powershell
curl.exe -X POST `
  "http://127.0.0.1:8000/api/v1/documents/upload" `
  -H "accept: application/json" `
  -F "file=@C:\path\to\document.pdf"
```

#### macOS

```bash
curl -X POST \
  "http://127.0.0.1:8000/api/v1/documents/upload" \
  -H "accept: application/json" \
  -F "file=@/path/to/document.pdf"
```

#### Linux

```bash
curl -X POST \
  "http://127.0.0.1:8000/api/v1/documents/upload" \
  -H "accept: application/json" \
  -F "file=@/path/to/document.pdf"
```

### List Documents

#### Windows PowerShell

```powershell
curl.exe "http://127.0.0.1:8000/api/v1/documents?limit=20&offset=0"
```

#### macOS

```bash
curl "http://127.0.0.1:8000/api/v1/documents?limit=20&offset=0"
```

#### Linux

```bash
curl "http://127.0.0.1:8000/api/v1/documents?limit=20&offset=0"
```

### Retrieve One Document

Replace `<DOCUMENT_UUID>` with a real UUID.

#### Windows PowerShell

```powershell
curl.exe "http://127.0.0.1:8000/api/v1/documents/<DOCUMENT_UUID>"
```

#### macOS

```bash
curl "http://127.0.0.1:8000/api/v1/documents/<DOCUMENT_UUID>"
```

#### Linux

```bash
curl "http://127.0.0.1:8000/api/v1/documents/<DOCUMENT_UUID>"
```

---

## Testing

### Run the Complete Suite

#### Windows PowerShell

```powershell
python -m pytest -v
```

#### macOS

```bash
python -m pytest -v
```

#### Linux

```bash
python -m pytest -v
```

Current verified result:

```text
220 passed
```

A Starlette TestClient deprecation warning may appear. It is currently non-blocking.

### Focused Tests

#### Windows PowerShell

```powershell
python -m pytest tests\test_local_storage.py -v
python -m pytest tests\test_pdf_extraction.py -v
python -m pytest tests\test_pdf_classification.py -v
python -m pytest tests\test_document_processing.py -v
python -m pytest tests\test_text_chunking.py -v
python -m pytest tests\test_document_chunk_persistence.py -v
python -m pytest tests\test_embedding_service.py -v
python -m pytest tests\test_qdrant_indexing.py -v
python -m pytest tests\test_qdrant_search.py -v
python -m pytest tests\test_retrieval_service.py -v
python -m pytest tests\test_bm25_service.py -v
python -m pytest tests\test_rrf_service.py -v
python -m pytest tests\test_hybrid_retrieval.py -v
python -m pytest tests\test_reranker_service.py -v
python -m pytest tests\test_evaluation_service.py -v
python -m pytest tests\test_latency.py -v
python -m pytest tests\test_generation_service.py -v
python -m pytest tests\test_groq_llm.py -v
```

#### macOS / Linux

```bash
python -m pytest tests/test_local_storage.py -v
python -m pytest tests/test_pdf_extraction.py -v
python -m pytest tests/test_pdf_classification.py -v
python -m pytest tests/test_document_processing.py -v
python -m pytest tests/test_text_chunking.py -v
python -m pytest tests/test_document_chunk_persistence.py -v
python -m pytest tests/test_embedding_service.py -v
python -m pytest tests/test_qdrant_indexing.py -v
python -m pytest tests/test_qdrant_search.py -v
python -m pytest tests/test_retrieval_service.py -v
python -m pytest tests/test_bm25_service.py -v
python -m pytest tests/test_rrf_service.py -v
python -m pytest tests/test_hybrid_retrieval.py -v
python -m pytest tests/test_reranker_service.py -v
python -m pytest tests/test_evaluation_service.py -v
python -m pytest tests/test_latency.py -v
python -m pytest tests/test_generation_service.py -v
python -m pytest tests/test_groq_llm.py -v
```

---

## Database Verification

### List Tables

#### Windows PowerShell

```powershell
docker compose exec postgres psql `
  -U evidencevault_user `
  -d evidencevault_db `
  -c "\dt"
```

#### macOS

```bash
docker compose exec postgres psql \
  -U evidencevault_user \
  -d evidencevault_db \
  -c "\dt"
```

#### Linux

```bash
docker compose exec postgres psql \
  -U evidencevault_user \
  -d evidencevault_db \
  -c "\dt"
```

### Inspect `document_chunks`

#### Windows PowerShell

```powershell
docker compose exec postgres psql `
  -U evidencevault_user `
  -d evidencevault_db `
  -c "\d document_chunks"
```

#### macOS

```bash
docker compose exec postgres psql \
  -U evidencevault_user \
  -d evidencevault_db \
  -c "\d document_chunks"
```

#### Linux

```bash
docker compose exec postgres psql \
  -U evidencevault_user \
  -d evidencevault_db \
  -c "\d document_chunks"
```

### Compare Document Metadata and Chunk Rows

Replace `<DOCUMENT_UUID>`.

#### Windows PowerShell

```powershell
docker compose exec postgres psql `
  -U evidencevault_user `
  -d evidencevault_db `
  -c "SELECT id, status, page_count, chunk_count FROM documents WHERE id = '<DOCUMENT_UUID>';"

docker compose exec postgres psql `
  -U evidencevault_user `
  -d evidencevault_db `
  -c "SELECT COUNT(*) AS actual_chunk_rows FROM document_chunks WHERE document_id = '<DOCUMENT_UUID>';"
```

#### macOS

```bash
docker compose exec postgres psql \
  -U evidencevault_user \
  -d evidencevault_db \
  -c "SELECT id, status, page_count, chunk_count FROM documents WHERE id = '<DOCUMENT_UUID>';"

docker compose exec postgres psql \
  -U evidencevault_user \
  -d evidencevault_db \
  -c "SELECT COUNT(*) AS actual_chunk_rows FROM document_chunks WHERE document_id = '<DOCUMENT_UUID>';"
```

#### Linux

```bash
docker compose exec postgres psql \
  -U evidencevault_user \
  -d evidencevault_db \
  -c "SELECT id, status, page_count, chunk_count FROM documents WHERE id = '<DOCUMENT_UUID>';"

docker compose exec postgres psql \
  -U evidencevault_user \
  -d evidencevault_db \
  -c "SELECT COUNT(*) AS actual_chunk_rows FROM document_chunks WHERE document_id = '<DOCUMENT_UUID>';"
```

---

## Infrastructure Commands

### Status

```bash
docker compose ps
```

### Logs

```bash
docker compose logs -f
```

Press `Ctrl+C` to stop following logs.

### Stop Without Deleting Data

```bash
docker compose stop
```

### Remove Containers but Keep Volumes

```bash
docker compose down
```

> Avoid `docker compose down -v` unless you intentionally want to delete PostgreSQL and Qdrant data.

---

## Safe Development Shutdown

### Windows PowerShell

```powershell
# Stop Uvicorn with Ctrl+C, then:
deactivate
Set-Location "D:\Projects\EvidenceVault-AI"
docker compose stop
```

### macOS / Linux

```bash
# Stop Uvicorn with Ctrl+C, then:
deactivate
cd /path/to/evidencevault-ai
docker compose stop
```

### Resume Development

#### Windows PowerShell

```powershell
Set-Location "D:\Projects\EvidenceVault-AI"
.\.venv\Scripts\Activate.ps1
docker compose up -d
Set-Location backend
python -m alembic current
python -m pytest -v
```

#### macOS / Linux

```bash
cd /path/to/evidencevault-ai
source .venv/bin/activate
docker compose up -d
cd backend
python -m alembic current
python -m pytest -v
```

---

## Database Design

### `documents`

Stores:

- Document UUID
- Original filename
- Internal filename
- Storage path
- File size
- MIME type
- Processing status
- Page count
- Chunk count
- Error message
- Timestamps

### `document_chunks`

Stores:

- Deterministic chunk UUID
- Parent document UUID
- Global chunk index
- Page number
- Page-local chunk index
- Text
- Character count
- Word count
- SHA-256 content hash
- Creation timestamp

Protections:

- UUID primary key
- Foreign key to `documents`
- `ON DELETE CASCADE`
- Unique document-wide chunk index
- Unique page-local chunk index
- Composite document-and-page index

---

## Design Decisions

### Why PostgreSQL and Qdrant?

PostgreSQL is the durable source of truth for metadata, processing states, chunk text, ordering, and hashes.

Qdrant is the retrieval index for embedding vectors, similarity search, and stable point upserts.

This separation allows the vector index to be rebuilt from PostgreSQL at any time.

### Why Preserve Page Boundaries?

Every chunk belongs to one page, enabling citations such as:

```text
Source: employee-handbook.pdf, page 7
```

Cross-page chunks would make attribution ambiguous.

### Why Deterministic IDs?

Random IDs change on every run. Deterministic IDs stay stable when content and chunking settings are unchanged.

They support:

- Safe reprocessing
- Idempotent upserts
- Incremental indexing
- Content-change detection
- Easier debugging

### Why Transactional Replacement?

The system removes old chunks and inserts the complete new set inside one transaction. A failure rolls everything back, preventing incomplete or duplicated data.

### Why Character-Based Chunking?

The final embedding model is not locked yet. Character-based limits remain predictable, deterministic, model-independent, dependency-light, and easy to test.

### Why Fuse Rankings Instead of Averaging Scores?

Dense retrieval (cosine similarity) and BM25 (lexical score) live on incompatible scales — a `0.83` from one and a `3.1` from the other mean nothing comparable side by side. Reciprocal Rank Fusion sidesteps this by only using rank position (`1 / (k + rank)`), making it robust to retrievers with completely different scoring distributions, and it's the standard approach used by production hybrid-search systems.

### Why Rerank Only a Shortlist, Not the Whole Corpus?

A cross-encoder reads the query and a chunk together in one forward pass, which is far more accurate than comparing two separately-computed embeddings — but it can't be pre-computed or indexed, so running it against every chunk in a document doesn't scale. The standard pattern is: cheap retrieval finds a shortlist fast, then the cross-encoder reorders just that shortlist accurately.

### Why a Standalone Evaluation Framework Instead of "It Feels Better"?

Recall, Precision, Hit Rate, MRR, and NDCG each catch different failure modes a retriever can have — high recall with poor ordering, or a perfect first hit but a small overall net. Measuring them explicitly, before and after each retrieval change, turns "I added reranking" into "I added reranking and it improved NDCG@5 by a measurable amount," which is a fundamentally stronger and more defensible engineering claim.

---

## Roadmap

### Phase 1: Backend and Infrastructure

- [x] FastAPI foundation
- [x] Environment-based settings
- [x] Health endpoint
- [x] PostgreSQL and Qdrant infrastructure
- [x] Async SQLAlchemy
- [x] Alembic migrations
- [x] Dependency health checks

### Phase 2: Secure Ingestion

- [x] Secure PDF upload
- [x] Streamed local storage
- [x] File-type and signature validation
- [x] Document metadata persistence
- [x] Upload, list, and detail endpoints
- [x] Storage consistency verification

### Phase 3: PDF Intelligence

- [x] PDF inspection
- [x] Page-level extraction
- [x] Conservative cleaning
- [x] Encrypted-PDF detection
- [x] Malformed-PDF detection
- [x] Blank-PDF detection
- [x] Scanned or image-only classification
- [x] Partial-extraction classification

### Phase 4: Chunking and Persistence

- [x] Page-aware chunking
- [x] Paragraph and sentence boundary preservation
- [x] Configurable overlap
- [x] UUID5 chunk IDs
- [x] SHA-256 hashes
- [x] Transactional PostgreSQL persistence
- [x] Idempotent replacement
- [x] Processing endpoint
- [x] Paginated chunk retrieval

### Phase 5: Embeddings and Retrieval

- [x] Embedding model integration
- [x] Qdrant collection management
- [x] Vector indexing
- [x] Dense retrieval
- [x] BM25 retrieval
- [x] Reciprocal Rank Fusion
- [x] Cross-encoder reranking
- [x] Retrieval evaluation framework (Recall, Precision, Hit Rate, MRR, NDCG, latency)
- [ ] Metadata filtering

### Phase 6: Grounded Generation

- [x] LLM provider abstraction (Groq)
- [x] Grounded prompt templates
- [ ] Wire hybrid retrieval → reranker → generation into one live endpoint
- [ ] Page-level citations
- [ ] "Not enough evidence" fallback
- [ ] Streaming responses
- [ ] Conversation history

### Phase 7: Evaluation and Product Features

- [x] Retrieval evaluation (Recall, Precision, Hit Rate, MRR, NDCG)
- [x] Latency tracking
- [ ] RAGAS metrics
- [ ] Faithfulness scoring
- [ ] Feedback collection
- [ ] Analytics dashboard
- [ ] Document comparison
- [ ] Risk-clause extraction
- [ ] Human-review flags

### Phase 8: Production Deployment

- [ ] Authentication and authorization
- [ ] AWS S3 storage
- [ ] Dockerized API service
- [ ] CI/CD
- [ ] Cloud deployment
- [ ] Monitoring
- [ ] Rate limiting
- [ ] Security hardening

---

## Planned Query Pipeline

```text
User question
    ↓
Optional query rewriting
    ↓
BM25 retrieval + Qdrant dense retrieval
    ↓
Reciprocal Rank Fusion
    ↓
Cross-encoder reranking
    ↓
Top evidence chunks
    ↓
Grounded LLM answer
    ↓
Page-level citations
    ↓
Faithfulness and unsupported-claim checks
```

Every stage up to "Grounded LLM answer" is implemented and independently tested. Wiring them together behind one endpoint, plus citations and faithfulness checks, is the next milestone.

Simple questions will bypass unnecessary agent behavior. Agentic tools will be reserved for multi-step workflows such as document comparison, calculations, and structured risk analysis.

---

## Example Future Response

```json
{
  "answer": "The termination notice period is 30 days.",
  "sources": [
    {
      "document_id": "7f8c9a2e-0000-0000-0000-000000000000",
      "filename": "vendor-contract.pdf",
      "page_number": 7,
      "chunk_id": "e2a3c866-0000-5000-8000-000000000000"
    }
  ],
  "faithfulness_score": 0.93,
  "latency_ms": 1840,
  "insufficient_evidence": false
}
```

This format is planned and is not part of the current API yet.

---

## Skills Demonstrated

- Python backend engineering
- FastAPI
- Async programming
- REST API design
- PostgreSQL
- SQLAlchemy
- Alembic
- Docker Compose
- Qdrant infrastructure
- PDF processing
- Document ingestion
- Text normalization
- Chunking strategy
- Deterministic identifiers
- Database transactions
- Embeddings and vector search
- Hybrid retrieval (dense + lexical)
- Reciprocal Rank Fusion
- Cross-encoder reranking
- Information retrieval evaluation (Recall, Precision, MRR, NDCG, Hit Rate)
- LLM integration
- Error handling
- Automated testing with dependency injection and test doubles
- RAG system design
- Production-aware engineering

---

## Contribution Workflow

### Create a Branch

```bash
git checkout -b feature/<feature-name>
```

### Run Tests Before Committing

```bash
cd backend
python -m pytest -v
cd ..
```

### Commit and Push

```bash
git add .
git commit -m "feat: describe the change"
git push -u origin feature/<feature-name>
```

---

## Known Limitations

Not implemented yet:

- OCR execution for image-only PDFs
- DOCX, HTML, CSV, and image ingestion
- A live query/chat endpoint wiring retrieval, reranking, and generation together
- Page-level citations in API responses
- Streaming responses
- Faithfulness / RAGAS-style scoring
- Authentication
- Cloud object storage
- Production deployment
- Frontend application

These are tracked transparently in the roadmap.

---

## Author

**Dhiren Lulla**

- GitHub: [@dhirenlulla](https://github.com/dhirenlulla)
- Repository: [evidencevault-ai](https://github.com/dhirenlulla/evidencevault-ai)

---

## Project Vision

EvidenceVault AI is being built to become more than a PDF chatbot.

The goal is a reliable document intelligence system that:

- Shows where every answer came from
- Preserves page-level evidence
- Knows when the document does not contain an answer
- Measures retrieval and answer quality, not just "vibes"
- Supports safe reprocessing
- Remains explainable in interviews and system-design discussions
- Can evolve from a local MVP into a deployable enterprise-style platform

> **Evidence before answers. Grounding before generation. Measurement before claims.**
