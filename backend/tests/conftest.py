from collections.abc import Generator
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.main import app

@pytest.fixture
def fake_db_session() -> AsyncMock:
    """
    Return a controlled substitute for SQLAlchemy's AsyncSession.
    
    The document repository functions are mocked in the API tests,
    so this session is passed through the route but does not connect
    to the real PostgreSQL database.
    """
    
    return AsyncMock(
        spec=AsyncSession
    )
    
    
@pytest.fixture
def client(fake_db_session: AsyncMock) -> Generator[TestClient, None, None]:
    """ 
    Create a FastAPI TestClient with the database dependency replaced.
    
    The override exists only during one test and is removed afterward.
    """
    
    async def override_get_db_session():
        yield fake_db_session
        
    app.dependency_overrides[get_db_session] = override_get_db_session
    
    test_client = TestClient(app)
    
    try:
        yield test_client
        
    finally:
        app.dependency_overrides.pop(
            get_db_session,
            None
        )
        
        test_client.close()