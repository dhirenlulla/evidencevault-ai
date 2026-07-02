<div align="center">

# EvidenceVault AI

### Production-Aware Document Intelligence and Citation-Ready RAG Platform

A backend-first platform for secure PDF ingestion, document classification, page-aware extraction, deterministic chunking, durable PostgreSQL persistence, and grounded Retrieval-Augmented Generation.

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Async_API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC244C)](https://qdrant.tech/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-57_Passing-brightgreen)](#testing)
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

**EvidenceVault AI** is a production-aware document intelligence platform designed to help teams securely ingest enterprise documents, preserve page-level evidence, and eventually answer questions using grounded Retrieval-Augmented Generation (RAG).

The platform is being built for document-heavy workflows such as:

- Legal and contract analysis
- HR policies and employee handbooks
- Compliance and audit documentation
- Research papers and technical reports
- Finance and operations documents
- Internal knowledge bases

Unlike a basic “chat with PDF” prototype, EvidenceVault AI is being developed as a production-oriented system with explicit lifecycle states, secure file handling, PDF classification, page-aware extraction, deterministic chunks, transactional persistence, database migrations, automated tests, and infrastructure for future hybrid retrieval and cited answers.

> **Current scope:** Secure ingestion, classification, extraction, chunking, and durable PostgreSQL persistence are implemented. Step 5F-B, which exposes processing and chunk-retrieval APIs, is the current milestone. Embeddings, Qdrant indexing, retrieval, reranking, and LLM generation come next.

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
10. Retrieves evidence using semantic and keyword search.
11. Produces grounded answers with page-level citations.
12. Evaluates faithfulness and refuses unsupported claims.

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
- Automated unit and API tests

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
| Step 5F-A | ✅ Complete | Durable transactional chunk persistence in PostgreSQL |
| Step 5F-B | 🚧 In progress | Processing API and paginated chunk-retrieval API |
| Step 6 | ⏳ Planned | Embeddings and Qdrant vector indexing |
| Step 7 | ⏳ Planned | Citation-aware RAG question answering |

### Verified Development Baseline

The current implementation has been validated with:

- **57 passing automated tests**
- A real **16-page PDF**
- **64,210 extracted characters**
- **10,096 extracted words**
- **73 deterministic chunks**
- Matching PostgreSQL `chunk_count` and physical chunk-row counts
- Stable chunk IDs across repeated runs
- Idempotent reprocessing without duplicated rows

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
- **Qdrant** for the upcoming vector-search layer
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

Future indexing states:

```text
chunked
   ↓
indexing
   ↓
ready
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
- Stable future Qdrant point IDs
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

    B --> Q[Health Service]
    Q --> E
    Q --> R[(Qdrant Infrastructure)]
```

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
```

### Planned RAG Architecture

```mermaid
flowchart TD
    A[Persisted PostgreSQL Chunks] --> B[Embedding Model]
    B --> C[(Qdrant Vector Index)]

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
| Vector infrastructure | Qdrant |
| PDF processing | PyMuPDF |
| Upload parsing | python-multipart |
| Async file I/O | aiofiles |
| Testing | Pytest, HTTPX |
| Infrastructure | Docker Compose |

### Planned

| Layer | Planned Technology |
|---|---|
| Embeddings | Sentence Transformers or BGE |
| Dense retrieval | Qdrant |
| Sparse retrieval | BM25 |
| Fusion | Reciprocal Rank Fusion |
| Reranking | Cross-encoder |
| LLM layer | Model-agnostic provider abstraction |
| Evaluation | RAGAS and custom metrics |
| Frontend | Streamlit or Next.js |
| Cloud storage | AWS S3 |
| Deployment | Docker and AWS |

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
│   │   ├── core/                      # Settings and exceptions
│   │   ├── db/models/                 # SQLAlchemy models
│   │   ├── repositories/              # Database access
│   │   ├── schemas/                   # Pydantic schemas
│   │   ├── services/                  # Processing logic
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
Copy-Item backend\\.env.example backend\\.env
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
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\\requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\\.venv\\Scripts\\Activate.ps1
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

UPLOAD_DIRECTORY=uploads
MAX_UPLOAD_SIZE_MB=20
UPLOAD_CHUNK_SIZE_BYTES=1048576
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

### Current Step 5F-B Milestone

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/documents/{document_id}/process` | Run classification, extraction, chunking, and persistence |
| `GET` | `/api/v1/documents/{document_id}/chunks` | Return paginated persisted chunks |

These two routes are in progress and should only be marked complete after their tests pass.

---

## API Usage Examples

### Upload a PDF

#### Windows PowerShell

```powershell
curl.exe -X POST `
  "http://127.0.0.1:8000/api/v1/documents/upload" `
  -H "accept: application/json" `
  -F "file=@C:\\path\\to\\document.pdf"
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

## Processing and Inspection Scripts

Run from `backend` with the virtual environment active.

### Inspect a PDF

#### Windows PowerShell

```powershell
python -m scripts.inspect_pdf ".\\uploads\\<PDF_FILENAME>.pdf"
```

#### macOS

```bash
python -m scripts.inspect_pdf "./uploads/<PDF_FILENAME>.pdf"
```

#### Linux

```bash
python -m scripts.inspect_pdf "./uploads/<PDF_FILENAME>.pdf"
```

### Classify a PDF

#### Windows PowerShell

```powershell
python -m scripts.classify_pdf ".\\uploads\\<PDF_FILENAME>.pdf"
```

#### macOS

```bash
python -m scripts.classify_pdf "./uploads/<PDF_FILENAME>.pdf"
```

#### Linux

```bash
python -m scripts.classify_pdf "./uploads/<PDF_FILENAME>.pdf"
```

### Process a Registered Document

#### Windows PowerShell

```powershell
python -m scripts.process_document <DOCUMENT_UUID>
```

#### macOS

```bash
python -m scripts.process_document <DOCUMENT_UUID>
```

#### Linux

```bash
python -m scripts.process_document <DOCUMENT_UUID>
```

### Preview Chunks

#### Windows PowerShell

```powershell
python -m scripts.preview_chunks <DOCUMENT_UUID> `
  --limit 5 `
  --max-characters 1200 `
  --overlap-characters 200 `
  --minimum-page-characters 40
```

#### macOS

```bash
python -m scripts.preview_chunks <DOCUMENT_UUID> \
  --limit 5 \
  --max-characters 1200 \
  --overlap-characters 200 \
  --minimum-page-characters 40
```

#### Linux

```bash
python -m scripts.preview_chunks <DOCUMENT_UUID> \
  --limit 5 \
  --max-characters 1200 \
  --overlap-characters 200 \
  --minimum-page-characters 40
```

### Persist Chunks

#### Windows PowerShell

```powershell
python -m scripts.persist_chunks <DOCUMENT_UUID> `
  --max-characters 1200 `
  --overlap-characters 200 `
  --minimum-page-characters 40
```

#### macOS

```bash
python -m scripts.persist_chunks <DOCUMENT_UUID> \
  --max-characters 1200 \
  --overlap-characters 200 \
  --minimum-page-characters 40
```

#### Linux

```bash
python -m scripts.persist_chunks <DOCUMENT_UUID> \
  --max-characters 1200 \
  --overlap-characters 200 \
  --minimum-page-characters 40
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
57 passed
```

A Starlette TestClient deprecation warning may appear. It is currently non-blocking.

### Focused Tests

#### Windows PowerShell

```powershell
python -m pytest tests\\test_local_storage.py -v
python -m pytest tests\\test_pdf_extraction.py -v
python -m pytest tests\\test_pdf_classification.py -v
python -m pytest tests\\test_document_processing.py -v
python -m pytest tests\\test_text_chunking.py -v
python -m pytest tests\\test_document_chunk_persistence.py -v
```

#### macOS

```bash
python -m pytest tests/test_local_storage.py -v
python -m pytest tests/test_pdf_extraction.py -v
python -m pytest tests/test_pdf_classification.py -v
python -m pytest tests/test_document_processing.py -v
python -m pytest tests/test_text_chunking.py -v
python -m pytest tests/test_document_chunk_persistence.py -v
```

#### Linux

```bash
python -m pytest tests/test_local_storage.py -v
python -m pytest tests/test_pdf_extraction.py -v
python -m pytest tests/test_pdf_classification.py -v
python -m pytest tests/test_document_processing.py -v
python -m pytest tests/test_text_chunking.py -v
python -m pytest tests/test_document_chunk_persistence.py -v
```

---

## Database Verification

### List Tables

#### Windows PowerShell

```powershell
docker compose exec postgres psql `
  -U evidencevault_user `
  -d evidencevault_db `
  -c "\\dt"
```

#### macOS

```bash
docker compose exec postgres psql \
  -U evidencevault_user \
  -d evidencevault_db \
  -c "\\dt"
```

#### Linux

```bash
docker compose exec postgres psql \
  -U evidencevault_user \
  -d evidencevault_db \
  -c "\\dt"
```

### Inspect `document_chunks`

#### Windows PowerShell

```powershell
docker compose exec postgres psql `
  -U evidencevault_user `
  -d evidencevault_db `
  -c "\\d document_chunks"
```

#### macOS

```bash
docker compose exec postgres psql \
  -U evidencevault_user \
  -d evidencevault_db \
  -c "\\d document_chunks"
```

#### Linux

```bash
docker compose exec postgres psql \
  -U evidencevault_user \
  -d evidencevault_db \
  -c "\\d document_chunks"
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

#### Windows PowerShell

```powershell
docker compose ps
```

#### macOS

```bash
docker compose ps
```

#### Linux

```bash
docker compose ps
```

### Logs

#### Windows PowerShell

```powershell
docker compose logs -f
```

#### macOS

```bash
docker compose logs -f
```

#### Linux

```bash
docker compose logs -f
```

Press `Ctrl+C` to stop following logs.

### Stop Without Deleting Data

#### Windows PowerShell

```powershell
docker compose stop
```

#### macOS

```bash
docker compose stop
```

#### Linux

```bash
docker compose stop
```

### Remove Containers but Keep Volumes

#### Windows PowerShell

```powershell
docker compose down
```

#### macOS

```bash
docker compose down
```

#### Linux

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
Set-Location "D:\\Projects\\EvidenceVault-AI"
docker compose stop
```

### macOS

```bash
# Stop Uvicorn with Ctrl+C, then:
deactivate
cd /path/to/evidencevault-ai
docker compose stop
```

### Linux

```bash
# Stop Uvicorn with Ctrl+C, then:
deactivate
cd /path/to/evidencevault-ai
docker compose stop
```

### Resume Development

#### Windows PowerShell

```powershell
Set-Location "D:\\Projects\\EvidenceVault-AI"
.\\.venv\\Scripts\\Activate.ps1
docker compose up -d
Set-Location backend
python -m alembic current
python -m pytest -v
```

#### macOS

```bash
cd /path/to/evidencevault-ai
source .venv/bin/activate
docker compose up -d
cd backend
python -m alembic current
python -m pytest -v
```

#### Linux

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

Qdrant will become the retrieval index for embedding vectors, similarity search, metadata filters, and stable point upserts.

This separation allows the vector index to be rebuilt from PostgreSQL.

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
- [ ] Processing endpoint
- [ ] Paginated chunk retrieval

### Phase 5: Embeddings and Retrieval

- [ ] Embedding model integration
- [ ] Qdrant collection management
- [ ] Vector indexing
- [ ] Dense retrieval
- [ ] BM25 retrieval
- [ ] Reciprocal Rank Fusion
- [ ] Cross-encoder reranking
- [ ] Metadata filtering

### Phase 6: Grounded Generation

- [ ] LLM provider abstraction
- [ ] Grounded prompt templates
- [ ] Page-level citations
- [ ] “Not enough evidence” fallback
- [ ] Streaming responses
- [ ] Conversation history

### Phase 7: Evaluation and Product Features

- [ ] Retrieval evaluation
- [ ] RAGAS metrics
- [ ] Faithfulness scoring
- [ ] Latency tracking
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
- Error handling
- Automated testing
- RAG system design
- Production-aware engineering

---

## Contribution Workflow

### Create a Branch

#### Windows PowerShell

```powershell
git checkout -b feature/<feature-name>
```

#### macOS

```bash
git checkout -b feature/<feature-name>
```

#### Linux

```bash
git checkout -b feature/<feature-name>
```

### Run Tests Before Committing

#### Windows PowerShell

```powershell
Set-Location backend
python -m pytest -v
Set-Location ..
```

#### macOS

```bash
cd backend
python -m pytest -v
cd ..
```

#### Linux

```bash
cd backend
python -m pytest -v
cd ..
```

### Commit and Push

#### Windows PowerShell

```powershell
git add .
git commit -m "feat: describe the change"
git push -u origin feature/<feature-name>
```

#### macOS

```bash
git add .
git commit -m "feat: describe the change"
git push -u origin feature/<feature-name>
```

#### Linux

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
- Embedding generation
- Qdrant vector indexing
- Semantic or hybrid retrieval
- Cross-encoder reranking
- LLM answer generation
- Source-citation responses
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
- Measures retrieval and answer quality
- Supports safe reprocessing
- Remains explainable in interviews and system-design discussions
- Can evolve from a local MVP into a deployable enterprise-style platform

> **Evidence before answers. Grounding before generation.**
