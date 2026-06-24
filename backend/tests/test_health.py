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