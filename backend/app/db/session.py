import logging
from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

engine: AsyncEngine = create_async_engine(
    settings.database_url,
    echo=settings.sql_echo,
    pool_pre_ping=True,
)

AsyncSessionFactory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Provide one database session for a FastAPI request.

    The session is opened before the route executes and closed automatically
    after the request finishes. If an unexpected error occurs, any unfinished
    transaction is rolled back.
    """
    
    async with AsyncSessionFactory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        
async def check_database_connection() -> tuple[bool, str]:
    """
    Verify that PostgreSQL accepts a simple query.

    SELECT 1 does not read application data. It is a lightweight command
    commonly used to prove that the database connection is functional.
    """
    
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
            
        return True, "PostgreSQL connection is available"
    
    except SQLAlchemyError:
        logger.exception("PostgreSQL health check failed")
        return False, "PostgreSQL connection is unavailable"
    
    
async def close_database_engine() -> None:
    """
    Close SQLAlchemy's connection pool during application shutdown. 
    """
    
    await engine.dispose()