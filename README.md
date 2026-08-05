<div align="center">

# EvidenceVault AI

### Production-Aware Document Intelligence and Retrieval-Augmented Generation Platform

A backend-first platform for secure PDF ingestion, hybrid retrieval, cross-encoder reranking, measurable retrieval evaluation, and a live, streaming, grounded question-answering API.

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Async_API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC244C)](https://qdrant.tech/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-277_Passing-brightgreen)](#testing)
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

**EvidenceVault AI** is a production-aware document intelligence platform that securely ingests PDF documents, preserves page-level evidence, and answers questions using grounded, hybrid-retrieval-based, measurably-evaluated Retrieval-Augmented Generation (RAG).

Built for document-heavy workflows: legal/contract analysis, HR policies, compliance documentation, research papers, finance/operations documents, and internal knowledge bases.

> **Current scope:** Secure ingestion → classification → extraction → chunking → PostgreSQL persistence → embeddings → Qdrant indexing → hybrid retrieval (dense + BM25, fused via Reciprocal Rank Fusion) → cross-encoder reranking → streaming Groq LLM generation, wired end-to-end behind a live, streaming API. A standalone retrieval evaluation framework and structured observability (logging, request tracing) are also implemented. **Docker deployment for the backend, model-readiness health checks, and a frontend demo are next.**

---

## Why This Project Is Different

Most beginner RAG projects follow: `Upload PDF → Split text → Store vectors → Ask an LLM`. This project adds the production-oriented concerns that are usually skipped:

- Secure filename/MIME/signature validation, chunked async uploads, compensating cleanup on failure
- Encrypted, malformed, scanned, and partially-extractable PDF detection
- Deterministic UUID5 chunk IDs + SHA-256 fingerprints → idempotent reprocessing, stable Qdrant point IDs
- Transactional chunk replacement (delete + insert + commit, or full rollback)
- Dense retrieval decoupled from lexical (BM25) retrieval, fused by **rank** via Reciprocal Rank Fusion — not by averaging incompatible raw scores
- A two-stage retrieve-then-rerank pipeline (cheap bi-encoder shortlist → accurate cross-encoder reordering)
- A standalone, dependency-free retrieval evaluation framework (Recall/Precision/Hit Rate/MRR/NDCG) — measured, not assumed
- Real token-level streaming generation over Server-Sent Events, with provider-aware HTTP error mapping (429/502/504), not just a happy-path demo
- Structured JSON logging and request tracing via context-scoped correlation IDs
- Every service unit-tested against dependency-injected fakes — zero live DB/Qdrant/model dependency at test time

---

## Development Status

| Phase | Status | Summary |
|---|---:|---|
| 1–3 | ✅ Complete | FastAPI foundation, Docker Compose (Postgres + Qdrant), async SQLAlchemy, Alembic, health checks |
| 4 | ✅ Complete | Secure PDF upload/storage, processing state machine, PDF classification, page-aware extraction & cleaning |
| 5 | ✅ Complete | Deterministic chunking, transactional persistence, embeddings, Qdrant indexing, dense retrieval |
| 6 | ✅ Complete | BM25 lexical retrieval, hybrid retrieval orchestration, Reciprocal Rank Fusion |
| 7 | ✅ Complete | Cross-encoder reranking |
| 8 | ✅ Complete | Retrieval evaluation framework (Recall, Precision, Hit Rate, MRR, NDCG, latency stats) |
| 9 | ✅ Complete | Streaming generation (Groq, token-level) |
| 10 | ✅ Complete | Structured JSON logging + request tracing (correlation IDs) |
| 11 | ✅ Complete | Full pipeline wired live: `POST /query` and `POST /query/stream` (hybrid retrieval → RRF → rerank → generate) |
| 12 | 🚧 In progress | Deployment hardening — Dockerized backend, model-readiness health checks, fail-fast config validation, closing test-coverage gaps on `/answer` and `/retrieve` |
| 13 | ⏳ Planned | Frontend demo (Streamlit or Next.js) with a live, clickable link |
| 14 | ⏳ Planned | Page-level citations, "not enough evidence" fallback, faithfulness scoring |

### Verified Baseline

- **277+ passing automated tests**, zero failing, zero skipped — verify with `pytest -v`
- Validated against a real 16-page PDF: 64,210 characters, 10,096 words, 73 deterministic chunks
- Every retrieval-layer service (embedding, dense retrieval, BM25, RRF, reranking, evaluation, generation) tested via fakes/doubles — no live DB, Qdrant, or downloaded model required to run the suite

---

## Implemented Features

**Ingestion:** secure upload validation → PDF classification (`text_based` / `partially_extractable` / `scanned_or_image_only` / `empty` / `encrypted` / `malformed`) → page-level extraction with conservative Unicode cleaning → deterministic, page-aware, overlap-preserving chunking → transactional PostgreSQL persistence, idempotent on reprocessing.

**Retrieval:** sentence-transformer embeddings, lazily loaded and cached → Qdrant vector indexing with deterministic point IDs → dense retrieval + BM25 lexical retrieval over the same candidate pool → `RRFService` fuses both rankings by position (`1 / (k + rank)`), fully generic over any hashable ID, not just chunks → `CrossEncoderReranker` re-scores the fused shortlist by feeding `(query, chunk)` pairs jointly into a cross-encoder for higher-precision ordering.

**Generation:** provider-abstracted LLM layer (Groq) with both a complete-answer path and a real token-streaming path (Server-Sent Events) → `RAGPipelineService` composes hybrid retrieval → reranking → prompt construction → generation into one live pipeline, exposed via `POST /{document_id}/query` and `POST /{document_id}/query/stream`, with HTTP status mapping for LLM failures (`429` rate limit, `502` connection, `504` timeout).

**Evaluation:** `RetrievalEvaluator` computes Recall@K, Precision@K, Hit Rate@K, MRR, and NDCG@K per query; `RetrievalMetricsAggregator` averages across a query set; `compare_metrics` does before/after configuration comparisons; `LatencyTimer`/`compute_latency_stats` track wall-clock timing (mean/p50/p95) as a separate, reusable concern.

**Observability:** structured JSON logging (`JSONLogFormatter`, queryable fields, not free text) and request tracing via a `ContextVar`-backed correlation ID — every log line produced anywhere during a request's handling carries the same ID automatically.

---

## System Architecture

```mermaid
flowchart TD
    A[Client] --> B[FastAPI]
    B --> C[Upload Service] --> D[(PostgreSQL: documents)]
    D --> E[Processing: Classify -> Extract -> Chunk]
    E --> F[(PostgreSQL: document_chunks)]
    F --> G[Embedding Service] --> H[(Qdrant Vector Index)]

    I[User Query] --> J[Dense Retrieval]
    I --> K[BM25 Lexical Retrieval]
    J --> L[RRF Fusion]
    K --> L
    L --> M[Cross-Encoder Reranker]
    M --> N[Prompt Builder]
    N --> O[Groq LLM: complete or streamed]
    O --> P[Answer + Sources]

    B --> Q[Health Service] --> D
    Q --> H
```

**Retrieval flow:** dense (Qdrant cosine similarity) + BM25 (same candidate pool) → RRF fusion by rank → cross-encoder rerank of the fused shortlist → prompt construction → Groq generation (complete or SSE-streamed).

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language / API | Python 3.10, FastAPI, Pydantic / pydantic-settings |
| Database | PostgreSQL 16, SQLAlchemy 2.0 async, Psycopg 3, Alembic |
| Vector DB | Qdrant |
| PDF processing | PyMuPDF |
| Embeddings / Reranking | sentence-transformers (bi-encoder + CrossEncoder) |
| Lexical retrieval | BM25 (`rank_bm25`) |
| Fusion | Reciprocal Rank Fusion (custom) |
| LLM | Groq (streaming + complete) |
| Evaluation | Custom Recall/Precision/Hit Rate/MRR/NDCG + latency stats |
| Testing | Pytest, HTTPX |
| Infra | Docker Compose (Postgres + Qdrant; backend containerization in progress) |
| Planned | Frontend (Streamlit/Next.js), AWS S3, CI/CD, cloud deployment |

---

## Repository Structure

```text
evidencevault-ai/
├── backend/
│   ├── alembic/versions/       # Migration history
│   ├── app/
│   │   ├── api/routes/         # FastAPI routes + middleware
│   │   ├── clients/            # External service clients (Qdrant)
│   │   ├── core/               # Settings, exceptions, logging, DI wiring
│   │   ├── db/models/          # SQLAlchemy models
│   │   ├── repositories/       # Database access
│   │   ├── schemas/            # Pydantic schemas
│   │   ├── services/           # Ingestion, retrieval, reranking, generation, evaluation
│   │   └── main.py
│   ├── scripts/                # Dev utilities
│   ├── tests/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── requirements.lock.txt
├── frontend/                   # Not yet built
├── compose.yaml
└── README.md
```

---

## Local Development Setup

Requires: Git, Python 3.10, Docker Desktop/Engine with Compose.

```bash
git clone https://github.com/dhirenlulla/evidencevault-ai.git
cd evidencevault-ai

cp .env.example .env
cp backend/.env.example backend/.env

python3.10 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r backend/requirements.txt

docker compose up -d                 # starts Postgres + Qdrant
cd backend
python -m alembic upgrade head
python -m uvicorn app.main:app --reload
```

Open:
- Swagger UI: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/api/v1/health`
- Qdrant dashboard: `http://localhost:6333/dashboard`

> Windows PowerShell users: activate with `.venv\Scripts\Activate.ps1`; all other commands above are identical.

---

## Environment Configuration

Key variables (see `backend/.env.example` for the full list):

```env
DATABASE_URL=postgresql+psycopg://evidencevault_user:evidencevault_local_password@localhost:5432/evidencevault_db
QDRANT_URL=http://localhost:6333

EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5
RETRIEVAL_TOP_K=5
RRF_K=60

RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_TOP_K=5

GROQ_API_KEY=
GROQ_MODEL_NAME=llama-3.3-70b-versatile

LOG_LEVEL=INFO
```

> Never commit real secrets. These are local-development examples only.

---

## Available API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | PostgreSQL and Qdrant health |
| `POST` | `/api/v1/documents/upload` | Validate, store, and register a PDF |
| `GET` | `/api/v1/documents` | List documents |
| `GET` | `/api/v1/documents/{id}` | Document metadata |
| `POST` | `/api/v1/documents/{id}/process` | Classify → extract → chunk → persist |
| `GET` | `/api/v1/documents/{id}/chunks` | Paginated persisted chunks |
| `GET` | `/api/v1/documents/{id}/retrieve` | Dense-only retrieval (fast, no LLM call) |
| `POST` | `/api/v1/documents/{id}/answer` | Dense retrieval + generation (fast path) |
| `POST` | `/api/v1/documents/{id}/query` | **Full pipeline**: hybrid retrieval + RRF + rerank + generation |
| `POST` | `/api/v1/documents/{id}/query/stream` | Same as `/query`, streamed via Server-Sent Events |

`/query` and `/query/stream` are slower than `/answer` but generally higher quality — the reranking pass costs latency in exchange for better-ordered context.

### Example: upload + query

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/documents/upload" \
  -F "file=@/path/to/document.pdf"

curl -X POST "http://127.0.0.1:8000/api/v1/documents/<DOCUMENT_UUID>/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the termination notice period?"}'
```

---

## Testing

```bash
cd backend
pytest -v
```

Current verified result: **277 passed**, 0 failed. Individual test files map 1:1 to services under `app/services/` and routes under `app/api/routes/` — e.g. `tests/test_rag_pipeline_service.py` covers `app/services/rag_pipeline.py`.

---

## Design Decisions

**Why fuse rankings instead of averaging scores?** Dense cosine similarity and BM25 scores live on incompatible scales — a `0.83` and a `3.1` mean nothing comparable side by side. RRF uses only rank position (`1 / (k + rank)`), which is robust to retrievers with completely different score distributions — the standard approach in production hybrid search.

**Why rerank only a shortlist, not the whole corpus?** A cross-encoder reads the query and chunk together in one forward pass — more accurate than comparing separately-computed embeddings, but too slow to run against every chunk. Cheap retrieval finds a shortlist; the cross-encoder reorders just that shortlist.

**Why a standalone evaluation framework instead of "it feels better"?** Recall, Precision, Hit Rate, MRR, and NDCG each catch different failure modes. Measuring them explicitly turns "I added reranking" into "I added reranking and it improved NDCG@5 by a measurable amount" — a defensible engineering claim, not a vibe.

**Why deterministic chunk IDs?** `document UUID + page number + page-local index + SHA-256 text hash → UUID5`. Stable across reprocessing runs when content is unchanged, changes when content changes — enables idempotent upserts and safe retries.

**Why separate `/answer` and `/query` instead of one endpoint?** A real cost/quality trade-off, made visible rather than hidden: `/answer` is dense-only and fast; `/query` runs the full hybrid + rerank pipeline and is slower but generally more accurate. Callers choose.

---

## Roadmap

**Complete:** ingestion pipeline · embeddings & Qdrant indexing · hybrid retrieval (dense + BM25 + RRF) · cross-encoder reranking · retrieval evaluation framework · streaming generation · structured logging & request tracing · full pipeline wired behind live REST + SSE endpoints.

**In progress (Phase 12 — deployment hardening):** Dockerized backend + Compose wiring · health check that verifies embedding/reranker models are actually loaded, not just DB/Qdrant connectivity · fail-fast startup validation (e.g. missing `GROQ_API_KEY` caught at boot in production, not on a user's first request) · test coverage for `/answer` and `/retrieve`.

**Planned:**
- Frontend demo (Streamlit or Next.js) with a live, shareable link
- Page-level citations in API responses + "not enough evidence" fallback
- Faithfulness / RAGAS-style scoring
- Authentication, AWS S3 storage, CI/CD, cloud deployment, rate limiting

---

## Known Limitations

Not yet implemented: OCR for image-only PDFs; non-PDF ingestion (DOCX/HTML/CSV); page-level citations in responses; faithfulness scoring; authentication; cloud storage; production deployment; frontend. Tracked transparently above, not hidden.

---

## Skills Demonstrated

Python backend engineering · FastAPI · async programming · REST + SSE API design · PostgreSQL/SQLAlchemy/Alembic · Docker Compose · Qdrant · PDF processing · hybrid retrieval (dense + lexical) · Reciprocal Rank Fusion · cross-encoder reranking · information retrieval evaluation · LLM integration & streaming · structured logging & tracing · dependency-injected testing · production-aware system design.

---

## Author

**Dhiren Lulla** — [GitHub](https://github.com/dhirenlulla) · [Repository](https://github.com/dhirenlulla/evidencevault-ai)

> Evidence before answers. Grounding before generation. Measurement before claims.