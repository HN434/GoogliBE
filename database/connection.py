"""
Database connection and session management
"""

import asyncio
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from config import settings

logger = logging.getLogger(__name__)

# Base class for database models
Base = declarative_base()

# Database engine and session factory
engine = None
async_session_maker = None


def get_database_url() -> str:
    """
    Construct PostgreSQL database URL from settings
    
    Returns:
        Database connection URL
    """
    return (
        f"postgresql+asyncpg://{settings.DB_USER}:{settings.DB_PASSWORD}"
        f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
    )


async def init_db():
    """
    Initialize database connection and create tables
    """
    import time
    start_time = time.time()
    global engine, async_session_maker
    
    try:
        database_url = get_database_url()
        logger.info(f"Connecting to PostgreSQL database: {settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}")
        
        # Create async engine with connection timeout
        # Note: timeout is passed to asyncpg.connect(), command_timeout is for query timeouts
        engine = create_async_engine(
            database_url,
            echo=settings.DB_ECHO,
            future=True,
            pool_pre_ping=True,  # Verify connections before using
            pool_size=10,
            max_overflow=20,
            connect_args={
                "server_settings": {
                    "application_name": "cricket_backend",
                },
                "timeout": 10,  # 10 second connection timeout for asyncpg
            }
        )
        
        # Test connection with timeout
        try:
            conn = await asyncio.wait_for(engine.connect(), timeout=10.0)
            await conn.close()
        except asyncio.TimeoutError:
            logger.error("❌ Database connection timeout after 10 seconds")
            raise ConnectionError("Database connection timeout")
        
        connect_time = time.time() - start_time
        logger.info(f"✅ Database connection established in {connect_time:.2f}s")
        
        # Create session factory
        async_session_maker = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Create all tables (only if they don't exist - this is idempotent)
        table_start = time.time()
        async with engine.begin() as conn:
            from database.models import Commentary, Match  # ensure models are imported
            # Use create_all with checkfirst=True to avoid unnecessary work
            await conn.run_sync(lambda sync_conn: Base.metadata.create_all(sync_conn, checkfirst=True))
        
        table_time = time.time() - table_start
        total_time = time.time() - start_time
        logger.info(f"✅ Database tables ready in {table_time:.2f}s (total init: {total_time:.2f}s)")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}", exc_info=True)
        raise


async def close_db():
    """
    Close database connections
    """
    global engine
    
    if engine:
        await engine.dispose()
        logger.info("✅ Database connections closed")


async def get_db() -> AsyncSession:
    """
    Get database session
    
    Yields:
        AsyncSession instance
    """
    if not async_session_maker:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

