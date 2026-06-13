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