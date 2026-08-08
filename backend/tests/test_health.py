from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root_endpoint() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Welcome to EvidenceVault AI"


def test_health_endpoint_when_dependencies_are_available() -> None:
    with patch(
        "app.api.routes.health.check_database_connection",
        new=AsyncMock(
            return_value=(
                True,
                "PostgreSQL connection is available",
            )
        ),
    ), patch(
        "app.api.routes.health.check_qdrant_connection",
        new=AsyncMock(
            return_value=(
                True,
                "Qdrant connection is available",
            )
        ),
    ):
        response = client.get("/api/v1/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert body["service"] == "EvidenceVault AI API"
    assert body["version"] == "0.3.0"
    assert body["environment"] == "development"
    assert body["postgres"]["status"] == "ok"
    assert body["qdrant"]["status"] == "ok"


def test_health_endpoint_when_qdrant_is_unavailable() -> None:
    with patch(
        "app.api.routes.health.check_database_connection",
        new=AsyncMock(
            return_value=(
                True,
                "PostgreSQL connection is available",
            )
        ),
    ), patch(
        "app.api.routes.health.check_qdrant_connection",
        new=AsyncMock(
            return_value=(
                False,
                "Qdrant connection is unavailable",
            )
        ),
    ):
        response = client.get("/api/v1/health")

    assert response.status_code == 503

    body = response.json()

    assert body["status"] == "degraded"
    assert body["postgres"]["status"] == "ok"
    assert body["qdrant"]["status"] == "error"
    
def test_health_response_includes_model_readiness_fields() -> None:
    """
    The response should always include embedding_model and
    reranker_model, whether or not the models happen to be
    loaded yet.
    """

    with patch(
        "app.api.routes.health.check_database_connection",
        new=AsyncMock(return_value=(True, "ok")),
    ), patch(
        "app.api.routes.health.check_qdrant_connection",
        new=AsyncMock(return_value=(True, "ok")),
    ):
        response = client.get("/api/v1/health")

    body = response.json()

    assert "embedding_model" in body
    assert "reranker_model" in body
    assert body["embedding_model"]["status"] in (
        "loaded",
        "not_loaded",
    )
    assert body["reranker_model"]["status"] in (
        "loaded",
        "not_loaded",
    )
    assert body["embedding_model"]["model_name"]
    assert body["reranker_model"]["model_name"]


def test_model_not_loaded_does_not_affect_overall_status() -> None:
    """
    An unloaded model must never drag the overall status/HTTP
    code down to "degraded"/503 - that's reserved for real
    infrastructure failures (Postgres/Qdrant). TestClient here
    is constructed without triggering the app's lifespan startup
    (no warm-up runs), so the models are guaranteed not_loaded
    for this test, which is exactly the case being verified.
    """

    with patch(
        "app.api.routes.health.check_database_connection",
        new=AsyncMock(return_value=(True, "ok")),
    ), patch(
        "app.api.routes.health.check_qdrant_connection",
        new=AsyncMock(return_value=(True, "ok")),
    ):
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"