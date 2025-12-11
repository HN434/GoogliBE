"""
Commentary Router - FastAPI endpoints for live cricket commentary
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query
from contextlib import asynccontextmanager

from services.redis_service import redis_service
from services.commentary_service import CommentaryService
from services.translation_service import (
    BedrockTranslationService,
    CommentaryTranslationService,
)
from workers.worker_supervisor import WorkerSupervisor
from ws_manager.connection_manager import WebSocketConnectionManager
from database.connection import init_db, close_db
from database.db_service import db_service
from config import settings
from models.commentary_schemas import CommentaryLine

# Setup logging - import and configure
from utils.logging_config import setup_logging

# Setup logging based on config
setup_logging(
    level=getattr(settings, 'COMMENTARY_LOG_LEVEL', 'DEBUG'),
    log_to_file=getattr(settings, 'COMMENTARY_LOG_TO_FILE', True)
)

logger = logging.getLogger(__name__)

# Global instances (initialized in lifespan)
commentary_service: CommentaryService = None
worker_supervisor: WorkerSupervisor = None
ws_manager: WebSocketConnectionManager = None
translation_service: CommentaryTranslationService = None

# Create router
router = APIRouter(prefix="/api/commentary", tags=["Commentary"])

HISTORICAL_COMMENTARY_WINDOW = 20

@asynccontextmanager
async def commentary_lifespan(app=None):
    """
    Lifespan context manager for commentary system
    Initializes and cleans up Redis, services, and supervisor
    """
    import time
    global commentary_service, worker_supervisor, ws_manager, translation_service
    
    total_start = time.time()
    logger.info("🚀 Starting commentary system...")
    
    try:
        # Initialize Database
        db_start = time.time()
        await init_db()
        logger.info(f"⏱️  Database init took {time.time() - db_start:.2f}s")
        
        # Initialize Redis
        redis_start = time.time()
        await redis_service.connect()
        logger.info(f"⏱️  Redis init took {time.time() - redis_start:.2f}s")
        
        # Initialize commentary service with key rotation
        service_start = time.time()
        api_keys = settings.get_rapidapi_keys()
        if not api_keys:
            logger.warning("⚠️  No RapidAPI keys configured. Commentary fetching will fail.")
        
        commentary_service = CommentaryService(
            api_keys=api_keys or [""],
            api_host=settings.RAPIDAPI_HOST,
            base_url=settings.RAPIDAPI_BASE_URL
        )
        logger.info(f"⏱️  Commentary service init took {time.time() - service_start:.2f}s")
        
        # Initialize WebSocket manager first (needed by supervisor)
        ws_start = time.time()
        ws_manager = WebSocketConnectionManager(
            redis_service=redis_service,
            worker_supervisor=None,  # Will be set after supervisor is created
            database_service=db_service
        )
        logger.info(f"⏱️  WebSocket manager init took {time.time() - ws_start:.2f}s")
        
        # Initialize translation services if enabled
        translation_service = None
        if settings.is_translation_enabled():
            try:
                bedrock_translator = BedrockTranslationService(
                    model_id=settings.BEDROCK_MODEL_ID,
                    region=settings.BEDROCK_REGION,
                    temperature=settings.BEDROCK_TEMPERATURE,
                    top_p=settings.BEDROCK_TOP_P,
                    max_tokens=settings.BEDROCK_MAX_TOKENS,
                )
                translation_service = CommentaryTranslationService(
                    redis_service=redis_service,
                    translator=bedrock_translator,
                    source_language=settings.COMMENTARY_DEFAULT_LANGUAGE,
                    cache_ttl_seconds=settings.TRANSLATION_CACHE_TTL_SECONDS,
                )
                logger.info("✅ Translation service initialized (Bedrock)")
            except Exception as exc:
                logger.error("Failed to initialize translation service: %s", exc, exc_info=True)
                translation_service = None
        else:
            logger.info("Commentary translation disabled via configuration")

        # Initialize worker supervisor with WebSocket manager reference
        supervisor_start = time.time()
        worker_supervisor = WorkerSupervisor(
            commentary_service=commentary_service,
            redis_service=redis_service,
            cleanup_interval=settings.WORKER_CLEANUP_INTERVAL,
            ws_manager=ws_manager,
            enable_historical_commentary=settings.ENABLE_HISTORICAL_COMMENTARY,
            translation_service=translation_service,
        )
        
        # Set supervisor reference in WebSocket manager
        ws_manager.worker_supervisor = worker_supervisor
        
        await worker_supervisor.start()
        logger.info(f"⏱️  Worker supervisor init took {time.time() - supervisor_start:.2f}s")
        
        total_time = time.time() - total_start
        logger.info(f"✅ Commentary system started successfully in {total_time:.2f}s")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to start commentary system: {e}")
        raise
    finally:
        # Cleanup
        logger.info("🛑 Shutting down commentary system...")
        
        if worker_supervisor:
            await worker_supervisor.stop()
        
        if commentary_service:
            await commentary_service.close()
        
        if redis_service:
            await redis_service.disconnect()
        
        # Close database connections
        await close_db()
        
        logger.info("✅ Commentary system shut down")


@router.get("/health")
async def commentary_health():
    """Health check for commentary system"""
    return {
        "status": "online",
        "redis_connected": redis_service.redis_client is not None,
        "active_workers": worker_supervisor.get_worker_count() if worker_supervisor else 0,
        "active_connections": ws_manager.get_connection_count() if ws_manager else 0
    }


@router.get("/workers")
async def get_workers_status():
    """Get status of all workers"""
    if not worker_supervisor:
        raise HTTPException(status_code=503, detail="Worker supervisor not initialized")
    
    return {
        "worker_count": worker_supervisor.get_worker_count(),
        "workers": worker_supervisor.get_all_workers_status()
    }


@router.get("/workers/{match_id}")
async def get_worker_status(match_id: str):
    """Get status of a specific worker"""
    if not worker_supervisor:
        raise HTTPException(status_code=503, detail="Worker supervisor not initialized")
    
    status = worker_supervisor.get_worker_status(match_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Worker not found for match {match_id}")
    
    return status


@router.get("/matches/{match_id}/subscribers")
async def get_subscriber_count(match_id: str):
    """Get number of subscribers for a match"""
    count = await redis_service.get_subscriber_count(match_id)
    return {
        "match_id": match_id,
        "subscriber_count": count
    }


def _timestamp_ms_to_datetime(timestamp_ms: int) -> datetime:
    """Convert milliseconds/seconds timestamp to datetime."""
    if timestamp_ms <= 0:
        raise ValueError("timestamp must be positive")
    if timestamp_ms > 1e12:  # microseconds
        timestamp_ms = timestamp_ms / 1000
    if timestamp_ms > 1e10:  # milliseconds
        timestamp_seconds = timestamp_ms / 1000
    else:
        timestamp_seconds = timestamp_ms
    return datetime.fromtimestamp(timestamp_seconds)


def _serialize_db_commentaries(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize DB commentary rows so frontend receives consistent payloads."""
    normalized = []
    for record in records:
        normalized.append(
            {
                "id": record.get("id"),
                "match_id": record.get("match_id"),
                "text": record.get("text"),
                "event_type": record.get("event_type"),
                "ball_number": record.get("ball_number"),
                "over_number": record.get("over_number"),
                "runs": record.get("runs"),
                "wickets": record.get("wickets"),
                "timestamp": record.get("timestamp"),
                "metadata": record.get("metadata"),
                "innings_id": record.get("innings_id"),
            }
        )
    return normalized


def _serialize_api_commentaries(match_id: str, lines: List[CommentaryLine]) -> List[Dict[str, Any]]:
    """Normalize CommentaryLine models returned by RapidAPI."""
    serialized: List[Dict[str, Any]] = []
    for line in lines:
        serialized.append(
            {
                "id": line.id,
                "match_id": match_id,
                "text": line.text,
                "event_type": line.event_type.value if line.event_type else None,
                "ball_number": line.ball_number,
                "over_number": line.over_number,
                "runs": line.runs,
                "wickets": line.wickets,
                "timestamp": line.timestamp.isoformat(),
                "metadata": line.metadata,
                "innings_id": (line.metadata or {}).get("inningsid") or (line.metadata or {}).get("inningsId"),
            }
        )
    return serialized


async def _maybe_translate_historical_commentaries(
    match_id: str,
    language: str,
    lines: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Translate commentary lines for historical endpoint when needed.
    """
    default_lang = (settings.COMMENTARY_DEFAULT_LANGUAGE or "en").lower()
    if (
        not lines
        or language == default_lang
        or not translation_service
    ):
        return lines

    try:
        translated = await translation_service.translate_lines(
            match_id,
            lines,
            language,
        )
        return translated or lines
    except Exception as exc:
        logger.error(
            "Failed to translate historical commentary for match %s (lang=%s): %s",
            match_id,
            language,
            exc,
            exc_info=True,
        )
        return lines


@router.get("/matches/{match_id}/comm-previous")
async def get_historical_commentary(
    match_id: str,
    tms: int = Query(..., description="Target timestamp (ms) to look back from"),
    iid: Optional[int] = Query(None, description="Innings identifier"),
    language: Optional[str] = Query(None, description="Language code for translated commentary"),
):
    """
    Return up to 20 commentary entries before the supplied timestamp.
    Prefers cached DB rows, falls back to RapidAPI and persists missing rows.
    """
    if not commentary_service:
        raise HTTPException(status_code=503, detail="Commentary service is not initialized")

    try:
        timestamp_dt = _timestamp_ms_to_datetime(tms)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    language_param = (language or settings.COMMENTARY_DEFAULT_LANGUAGE).lower()
    supported_languages = settings.get_supported_commentary_languages()
    if language_param not in supported_languages:
        logger.warning(
            "Unsupported language '%s' requested for historical commentary on match %s. Falling back to %s.",
            language_param,
            match_id,
            settings.COMMENTARY_DEFAULT_LANGUAGE,
        )
        language_param = settings.COMMENTARY_DEFAULT_LANGUAGE.lower()

    db_chain = await db_service.get_linked_commentaries(
        match_id=match_id,
        timestamp=timestamp_dt,
        limit=HISTORICAL_COMMENTARY_WINDOW,
        innings_id=iid,
    )

    if len(db_chain) >= HISTORICAL_COMMENTARY_WINDOW:
        lines = _serialize_db_commentaries(list(reversed(db_chain)))
        lines = [line for line in lines if line["metadata"]["timestamp_ms"] != tms]
        lines = await _maybe_translate_historical_commentaries(match_id, language_param, lines)
        return {
            "match_id": match_id,
            "innings_id": iid,
            "requested_timestamp": tms,
            "source": "database_chain",
            "language": language_param,
            "count": len(lines),
            "lines": lines,
        }

    lines_from_api: List[CommentaryLine] = []
    if commentary_service:
        lines_from_api, _ = await commentary_service.fetch_previous_commentaries(
            match_id=match_id,
            timestamp_ms=tms,
            innings_id=iid,
        )

    if lines_from_api:
        await db_service.store_commentaries(match_id, lines_from_api)
        lines = _serialize_api_commentaries(match_id, lines_from_api)
        lines = [line for line in lines if line["metadata"]["timestamp_ms"] != tms]
        lines = await _maybe_translate_historical_commentaries(match_id, language_param, lines)
        return {
            "match_id": match_id,
            "innings_id": iid,
            "requested_timestamp": tms,
            "source": "rapidapi",
            "language": language_param,
            "count": len(lines),
            "lines": lines,
        }

    raise HTTPException(
        status_code=502,
        detail="Unable to load historical commentary from cache or RapidAPI",
    )


@router.get("/test/message-format")
async def test_message_format():
    """Test endpoint to verify message format"""
    from models.commentary_schemas import WebSocketMessage, CommentaryUpdate, CommentaryLine, CommentaryEventType
    from datetime import datetime
    
    # Create a sample message
    sample_line = CommentaryLine(
        id="test_123",
        timestamp=datetime.now(),
        text="Test commentary line",
        event_type=CommentaryEventType.BALL,
        ball_number=1,
        over_number=1.1,
        runs=0
    )
    
    sample_update = CommentaryUpdate(
        match_id="test_match",
        timestamp=datetime.now(),
        lines=[sample_line],
        match_status="live",
        language=settings.COMMENTARY_DEFAULT_LANGUAGE
    )
    
    sample_ws_message = WebSocketMessage(
        type="commentary",
        match_id="test_match",
        language=settings.COMMENTARY_DEFAULT_LANGUAGE,
        data=sample_update.model_dump(mode="json", exclude_none=True)
    )
    
    return {
        "message_format": sample_ws_message.model_dump(mode="json", exclude_none=True),
        "serialized": sample_ws_message.model_dump_json(exclude_none=True)
    }


@router.websocket("/ws/match/{match_id}")
async def websocket_commentary(websocket: WebSocket, match_id: str):
    """
    WebSocket endpoint for subscribing to live commentary
    
    Args:
        websocket: WebSocket connection
        match_id: Match identifier
    
    Messages:
        - Incoming: Any text (keeps connection alive)
        - Outgoing: JSON messages with type, match_id, data, timestamp
    """
    if not ws_manager:
        logger.error("WebSocket manager not initialized")
        await websocket.close(code=503, reason="WebSocket manager not initialized")
        return
    
    language_param = (
        websocket.query_params.get("language")
        or websocket.query_params.get("lang")
        or settings.COMMENTARY_DEFAULT_LANGUAGE
    )
    language = (language_param or settings.COMMENTARY_DEFAULT_LANGUAGE).lower()
    supported_languages = settings.get_supported_commentary_languages()
    if language not in supported_languages:
        logger.warning(
            "Unsupported language '%s' requested for match %s. Falling back to %s.",
            language,
            match_id,
            settings.COMMENTARY_DEFAULT_LANGUAGE,
        )
        language = settings.COMMENTARY_DEFAULT_LANGUAGE.lower()

    logger.info(f"WebSocket connection request for match {match_id} (lang={language})")
    try:
        await ws_manager.handle_websocket(websocket, match_id, language)
    except Exception as e:
        logger.error(f"Error in WebSocket handler for match {match_id}: {e}", exc_info=True)
        try:
            await websocket.close(code=1011, reason=f"Internal error: {str(e)}")
        except:
            pass