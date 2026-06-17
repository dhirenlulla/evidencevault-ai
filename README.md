# EvidenceVault AI

EvidenceVault AI is a production-aware document intelligence platform that
answers questions from uploaded enterprise documents using grounded retrieval,
page-level citations, retrieval evaluation, and hallucination controls.

## Current Status

Step 1 completed:

- FastAPI project foundation
- Environment-based configuration
- Versioned health API
- OpenAPI documentation
- Automated tests

## Technology Stack

- Python
- FastAPI
- Pydantic
- Pytest

PostgreSQL, Qdrant, Docker, embeddings, hybrid retrieval, reranking,
Streamlit, and AWS will be added incrementally.

## Local Setup

```bash
cd backend
python -m pip install -r requirements.txt
uvicorn app.main:app --reload


## Local Infrastructure

EvidenceVault AI uses Docker Compose to run:

- PostgreSQL for relational application data
- Qdrant for vector storage and semantic retrieval

### Start Infrastructure

From the project root:

```bash
docker compose up -d

Check Service Status
```bash
docker compose ps

View Logs
```bash
docker compose logs -f

Test PostgreSQL
```bash
docker compose exec postgres pg_isready \
-U evidencevault_user \
-d evidencevault_db


Qdrant: 
REST API: http://localhost:6333
Dashboard: http://localhost:6333/dashboard
Health: http://localhost:6333/healthz

Stop Infrastructure
```bash
docker compose down