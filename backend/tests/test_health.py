from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_root_endpoint() -> None:
    response = client.get("/")
    
    assert response.status_code == 200
    assert response.json()["message"] == "Welcome to EvidenceVault AI"
    
def test_health_endpoint() -> None:
    response = client.get("/api/v1/health")
    
    assert response.status_code == 200
    
    body = response.json()
    
    assert body["status"]=="ok"
    assert body["service"] == "EvidenceVault AI API"
    assert body["version"] == "0.1.0"
    assert body["environment"] == "development"
    
    
# Why write tests immediately?

# Every major feature we add will need a success test and failure tests.

# Later, document upload tests will check:

# Valid PDF accepted
# Wrong file type rejected
# Oversized file rejected
# Empty file rejected
# Corrupted PDF handled cleanly

# Testing from the beginning prevents us from building an untestable application.