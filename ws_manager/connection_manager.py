"""
WebSocket Connection Manager
Handles WebSocket connections, subscriber management, and message forwarding
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Optional
from fastapi import WebSocket, WebSocketDisconnect
from services.redis_service import RedisService
from workers.worker_supervisor import WorkerSupervisor
from models.commentary_schemas import WebSocketMessage
from database.db_service import db_service

logger = logging.getLogger(__name__)


class WebSocketConnectionManager:
    """
    Manages WebSocket connections for live commentary
    - Tracks active connections per match
    - Subscribes to Redis Pub/Sub channels
    - Forwards commentary updates to clients
    - Manages subscriber lifecycle
    """

    def __init__(
        self,
        redis_service: RedisService,
        worker_supervisor: WorkerSupervisor,
        database_service=db_service,
    ):
        """
        Initialize WebSocket connection manager
        
        Args:
            redis_service: Redis service instance
            worker_supervisor: Worker supervisor instance
        """
        self.redis_service = redis_service
        self.worker_supervisor = worker_supervisor
        self.db_service = database_service
        
        # Track active connections: connection_id -> WebSocket
        self.connections: Dict[str, WebSocket] = {}
        
        # Track connection metadata: connection_id -> {match_id, subscriber_task, worker_id}
        self.connection_metadata: Dict[str, dict] = {}
        
        # Track connections currently disconnecting to prevent double cleanup
        self.disconnecting: set = set()
        
        # Track connections that are marked for closure (e.g., when match finishes)
        # These connections are in "closing" state and should not accept new operations
        self.closing_connections: set = set()

    async def connect(self, websocket: WebSocket, match_id: str, language: str) -> str:
        """
        Accept WebSocket connection and set up subscription
        
        Args:
            websocket: WebSocket connection
            match_id: Match identifier
        
        Returns:
            Connection ID
        """
        await websocket.accept()
        
        # Generate unique connection ID
        connection_id = str(uuid.uuid4())
        
        # Store connection (worker_id will be set after worker is ensured)
        self.connections[connection_id] = websocket
        self.connection_metadata[connection_id] = {
            "match_id": match_id,
            "language": language,
            "subscriber_task": None,
            "worker_id": None,  # Will be set after worker is ensured
            "created_at": asyncio.get_event_loop().time(),
            "language_subscriber_removed": False  # Track if language subscriber was already cleaned up
        }
        
        # Add to Redis subscriber set
        subscriber_count = await self.redis_service.add_subscriber(match_id, connection_id)
        await self.redis_service.add_language_subscriber(match_id, language)
        logger.info(
            f"WebSocket connected: {connection_id} for match {match_id} "
            f"(total subscribers: {subscriber_count})"
        )
        
        # Ensure worker is running for this match
        logger.info(f"🔍 Ensuring worker exists for match {match_id}")
        worker = await self.worker_supervisor.ensure_worker(match_id)
        
        # Now set the worker_id in metadata and verify worker
        if worker:
            self.connection_metadata[connection_id]["worker_id"] = id(worker)
            logger.info(f"✅ Active worker confirmed running for match {match_id} (worker_id={id(worker)})")
            # Verify the worker is actually running (not just created)
            if not worker.is_running:
                logger.error(f"Worker for match {match_id} was created but is not running!")
                # Clean up the failed connection - remove BOTH subscriber types
                await self.redis_service.remove_subscriber(match_id, connection_id)
                await self.redis_service.remove_language_subscriber(match_id, language)
                # Mark as removed to prevent double cleanup
                self.connection_metadata[connection_id]["language_subscriber_removed"] = True
                del self.connections[connection_id]
                del self.connection_metadata[connection_id]
                await websocket.close(code=1011, reason="Worker failed to start")
                return None
        else:
            logger.warning(f"Worker not created for match {match_id} (no subscribers?)")
            # Clean up and reject connection - remove BOTH subscriber types
            await self.redis_service.remove_subscriber(match_id, connection_id)
            await self.redis_service.remove_language_subscriber(match_id, language)
            # Mark as removed to prevent double cleanup
            self.connection_metadata[connection_id]["language_subscriber_removed"] = True
            del self.connections[connection_id]
            del self.connection_metadata[connection_id]
            await websocket.close(code=1011, reason="Failed to start worker")
            return None
        
        # Send stored commentary history if available
        if self.db_service:
            await self._send_commentary_history(connection_id, match_id, language)
        else:
            logger.warning("Database service unavailable; unable to send commentary history")

        # Start subscription task to listen to Redis Pub/Sub
        logger.info(f"Starting subscription task for connection {connection_id} on match {match_id}")
        subscriber_task = asyncio.create_task(
            self._subscribe_and_forward(connection_id, match_id, language)
        )
        self.connection_metadata[connection_id]["subscriber_task"] = subscriber_task
        
        # Send welcome message
        logger.info(f"Sending welcome message to connection {connection_id}")
        try:
            await self._send_message(
                connection_id,
                WebSocketMessage(
                    type="status",
                    match_id=match_id,
                    language=language,
                    data={
                        "message": "Connected to live commentary",
                        "match_id": match_id,
                        "subscriber_count": subscriber_count
                    }
                )
            )
            logger.info(f"✅ Welcome message sent to connection {connection_id}")
        except Exception as e:
            logger.error(f"❌ Failed to send welcome message to {connection_id}: {e}", exc_info=True)
        
        return connection_id
    async def _send_commentary_history(self, connection_id: str, match_id: str, language: str):
        """
        Send stored commentary history from the database to the connected client.
        Triggers historical backfill worker if no history exists and match is not complete.
        """
        if not self.db_service:
            logger.warning("Database service not available; skipping history send")
            return
        try:
            history = await self.db_service.get_match_commentaries(
                match_id,
                limit=500,
                offset=0,
                order="asc",
            )
            match_status = await self.db_service.get_match_status(match_id) if self.db_service else None
            miniscore = None
            if match_status:
                miniscore = (
                    (match_status.get("extra_metadata") or {}).get("miniscore")
                    or match_status.get("miniscore")
                )

            if history:
                logger.info(f"Sending {len(history)} stored commentaries to connection {connection_id}")
                await self._send_message(
                    connection_id,
                    WebSocketMessage(
                        type="history",
                        match_id=match_id,
                        language=language,
                        data={
                            "commentaries": history,
                            "miniscore": miniscore,
                        },
                    ),
                )
                return

            logger.info(f"No stored commentaries for match {match_id}, checking match state before historical fetch")
            state = (match_status or {}).get("state", "").upper() if match_status else ""
            if state in {"COMPLETE", "FINISHED"}:
                logger.info(f"Match {match_id} already complete; skipping historical worker")
                return

            if self.worker_supervisor:
                logger.info(f"Starting historical fetch for match {match_id}")
                await self.worker_supervisor.ensure_historical_worker(match_id)
            else:
                logger.warning("Worker supervisor unavailable; cannot start historical worker")

        except Exception as e:
            logger.error(f"Error sending commentary history for match {match_id}: {e}", exc_info=True)

    async def disconnect(self, connection_id: str):
        """
        Disconnect WebSocket and clean up
        
        Args:
            connection_id: Connection identifier
        """
        # Check if already disconnecting or disconnected
        if connection_id not in self.connections:
            logger.debug(f"Connection {connection_id} already disconnected")
            return
        
        # Prevent concurrent disconnect operations for the same connection
        if connection_id in self.disconnecting:
            logger.warning(f"Connection {connection_id} is already being disconnected, skipping duplicate cleanup")
            return
        
        # Mark as disconnecting
        self.disconnecting.add(connection_id)
        
        try:
            match_id = self.connection_metadata[connection_id]["match_id"]
            subscriber_task = self.connection_metadata[connection_id].get("subscriber_task")
            language = self.connection_metadata[connection_id].get("language", "en")
            language_subscriber_removed = self.connection_metadata[connection_id].get("language_subscriber_removed", False)
            
            # Cancel subscription task
            if subscriber_task and not subscriber_task.done():
                subscriber_task.cancel()
                try:
                    await subscriber_task
                except asyncio.CancelledError:
                    pass
            
            # Remove from Redis subscriber set
            subscriber_count = await self.redis_service.remove_subscriber(match_id, connection_id)
            
            # Only remove language subscriber if not already removed
            if not language_subscriber_removed:
                logger.debug(
                    f"Removing language subscriber for connection {connection_id} "
                    f"(match={match_id}, lang={language})"
                )
                await self.redis_service.remove_language_subscriber(match_id, language)
                self.connection_metadata[connection_id]["language_subscriber_removed"] = True
            else:
                logger.warning(
                    f"⚠️ Language subscriber already removed for connection {connection_id}, "
                    f"skipping duplicate removal (this indicates the cleanup was called twice)"
                )
            
            logger.info(
                f"WebSocket disconnected: {connection_id} from match {match_id} "
                f"(remaining subscribers: {subscriber_count})"
            )
            
            # If this was the last subscriber, immediately notify the worker to enter grace period
            if subscriber_count == 0 and self.worker_supervisor:
                worker = self.worker_supervisor.active_workers.get(match_id)
                if worker and not worker.in_grace_period:
                    logger.info(
                        f"⏳ Last subscriber disconnected from match {match_id}, "
                        f"immediately marking worker for grace period"
                    )
                    worker.in_grace_period = True
                    worker.zero_subscribers_since = datetime.now()
            
            # Remove connection
            del self.connections[connection_id]
            del self.connection_metadata[connection_id]
            
        finally:
            # Always remove from disconnecting set
            self.disconnecting.discard(connection_id)
        
        # Note: Worker cleanup is handled by supervisor's cleanup loop
        # when subscriber count reaches zero

    async def _subscribe_and_forward(self, connection_id: str, match_id: str, language: str):
        """
        Subscribe to Redis Pub/Sub channel and forward messages to WebSocket
        
        Args:
            connection_id: Connection identifier
            match_id: Match identifier
        """
        logger.info(f"Starting subscription forwarding for connection {connection_id} on match {match_id}")
        
        try:
            # Subscribe to match channel
            logger.debug(f"Subscribing to Redis channel for match {match_id}")
            message_count = 0
            async for message in self.redis_service.subscribe_to_match(match_id, language=language):
                message_count += 1
                logger.info(f"📨 Received message #{message_count} from Redis for match {match_id}, connection {connection_id}")
                logger.debug(f"Message content: {str(message)[:200]}...")  # Log first 200 chars
                
                # Check if connection still exists
                if connection_id not in self.connections:
                    logger.warning(f"Connection {connection_id} no longer exists, stopping subscription")
                    break
                
                # Forward message to WebSocket
                logger.info(f"📤 Forwarding message #{message_count} to WebSocket {connection_id}")
                try:
                    await self._send_message(
                        connection_id,
                        WebSocketMessage(
                            type="commentary",
                            match_id=match_id,
                            language=language,
                            data=message
                        )
                    )
                    logger.info(f"✅ Successfully forwarded message #{message_count} to WebSocket {connection_id}")
                except Exception as e:
                    logger.error(f"❌ Failed to forward message #{message_count} to {connection_id}: {e}", exc_info=True)
                
        except asyncio.CancelledError:
            logger.info(f"Subscription forwarding cancelled for connection {connection_id}")
        except Exception as e:
            logger.error(f"Error in subscription forwarding for {connection_id}: {e}")
            # Send error message to client
            try:
                await self._send_message(
                    connection_id,
                        WebSocketMessage(
                            type="error",
                            match_id=match_id,
                            language=language,
                            data={"error": str(e)}
                    )
                )
            except:
                pass

    async def _send_message(self, connection_id: str, message: WebSocketMessage):
        """
        Send message to WebSocket connection
        
        Args:
            connection_id: Connection identifier
            message: Message to send
        """
        if connection_id not in self.connections:
            logger.warning(f"Connection {connection_id} not found in connections dict")
            return
        
        websocket = self.connections[connection_id]
        
        try:
            # Serialize message with proper datetime handling
            message_dict = message.model_dump(mode="json", exclude_none=True)
            logger.debug(f"Sending message to {connection_id}: type={message.type}, match_id={message.match_id}, data_keys={list(message_dict.get('data', {}).keys())}")
            
            # Ensure non-ASCII languages (e.g., Hindi) are preserved on the wire
            await websocket.send_text(json.dumps(message_dict, ensure_ascii=False))
            logger.debug(f"✅ Successfully sent message to WebSocket {connection_id}")
        except Exception as e:
            logger.error(f"❌ Error sending message to {connection_id}: {e}", exc_info=True)
            # Connection might be closed, clean up
            await self.disconnect(connection_id)

    async def handle_websocket(self, websocket: WebSocket, match_id: str, language: str):
        """
        Handle WebSocket connection lifecycle
        
        Args:
            websocket: WebSocket connection
            match_id: Match identifier
        """
        connection_id = None
        
        try:
            # Connect
            connection_id = await self.connect(websocket, match_id, language)
            
            # Keep connection alive and handle any incoming messages
            while True:
                try:
                    # Wait for any message from client (ping/pong, etc.)
                    # Use receive() to handle both text and ping/pong
                    message = await websocket.receive()
                    
                    if message["type"] == "websocket.disconnect":
                        logger.info(f"WebSocket {connection_id} disconnected by client")
                        break
                    elif message["type"] == "websocket.receive":
                        if "text" in message:
                            logger.debug(f"Received text message from {connection_id}: {message['text']}")
                        elif "bytes" in message:
                            logger.debug(f"Received binary message from {connection_id}")
                    # Handle ping/pong automatically by FastAPI
                    
                except WebSocketDisconnect:
                    logger.info(f"WebSocket {connection_id} disconnected")
                    break
                except Exception as e:
                    logger.error(f"Error receiving message from {connection_id}: {e}", exc_info=True)
                    break
                    
        except Exception as e:
            logger.error(f"Error handling WebSocket for match {match_id}: {e}")
        finally:
            # Clean up on disconnect
            if connection_id:
                await self.disconnect(connection_id)

    def get_connection_count(self, match_id: Optional[str] = None) -> int:
        """
        Get number of active connections
        
        Args:
            match_id: Optional match ID to filter by
        
        Returns:
            Number of connections
        """
        if match_id:
            return sum(
                1 for meta in self.connection_metadata.values()
                if meta["match_id"] == match_id
            )
        return len(self.connections)

    async def disconnect_all_for_match(self, match_id: str, delay_seconds: float = 2.0, worker_id: int = None):
        """
        Disconnect WebSocket connections for a match
        Only disconnects connections that belong to a specific worker (if worker_id provided)
        This prevents disconnecting new connections that joined after the worker started closing
        
        Args:
            match_id: Match identifier
            delay_seconds: Delay before disconnecting (to ensure final message is sent)
            worker_id: ID of the worker that's finishing (only disconnect its connections)
        """
        # Find connections for this match that belong to the specific worker
        # If worker_id is None, disconnect all connections for the match (backward compatibility)
        connections_to_disconnect = []
        
        for conn_id, meta in self.connection_metadata.items():
            if meta["match_id"] == match_id:
                # If worker_id specified, only include connections from that worker
                if worker_id is None or meta.get("worker_id") == worker_id:
                    connections_to_disconnect.append(conn_id)
                else:
                    logger.info(
                        f"Skipping connection {conn_id} for match {match_id} "
                        f"(belongs to different worker: {meta.get('worker_id')} vs {worker_id})"
                    )
        
        if not connections_to_disconnect:
            logger.debug(f"No connections to disconnect for match {match_id} (worker_id={worker_id})")
            return
        
        # Mark these specific connections as closing
        for conn_id in connections_to_disconnect:
            self.closing_connections.add(conn_id)
        
        logger.info(
            f"🔌 Marked {len(connections_to_disconnect)} connections as closing for match {match_id} "
            f"(worker_id={worker_id}, will disconnect after {delay_seconds}s delay)"
        )
        
        # Wait a bit to ensure final message is sent and received
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        
        # Disconnect the marked connections
        disconnected = 0
        skipped = 0
        for connection_id in connections_to_disconnect:
            # Double-check connection still exists and is marked for closing
            if connection_id not in self.connections:
                logger.debug(f"Connection {connection_id} already disconnected")
                skipped += 1
                continue
                
            try:
                await self.disconnect(connection_id)
                disconnected += 1
                logger.info(f"✅ Disconnected connection {connection_id} for finished match {match_id}")
            except Exception as e:
                logger.error(f"❌ Error disconnecting {connection_id} for match {match_id}: {e}")
            finally:
                # Remove from closing set
                self.closing_connections.discard(connection_id)
        
        logger.info(
            f"🎯 Disconnect completed for match {match_id}: "
            f"{disconnected} disconnected, {skipped} already gone"
        )

