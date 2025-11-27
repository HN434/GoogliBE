"""
WebSocket Connection Manager
Handles WebSocket connections, subscriber management, and message forwarding
"""

import asyncio
import logging
import uuid
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
        
        # Track connection metadata: connection_id -> {match_id, subscriber_task}
        self.connection_metadata: Dict[str, dict] = {}

    async def connect(self, websocket: WebSocket, match_id: str) -> str:
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
        
        # Store connection
        self.connections[connection_id] = websocket
        self.connection_metadata[connection_id] = {
            "match_id": match_id,
            "subscriber_task": None
        }
        
        # Add to Redis subscriber set
        subscriber_count = await self.redis_service.add_subscriber(match_id, connection_id)
        logger.info(
            f"WebSocket connected: {connection_id} for match {match_id} "
            f"(total subscribers: {subscriber_count})"
        )
        
        # Ensure worker is running for this match
        logger.info(f"Ensuring worker exists for match {match_id}")
        worker = await self.worker_supervisor.ensure_worker(match_id)
        if worker:
            logger.info(f"Worker confirmed running for match {match_id}")
        else:
            logger.warning(f"Worker not created for match {match_id} (no subscribers?)")
        
        # Send stored commentary history if available
        if self.db_service:
            await self._send_commentary_history(connection_id, match_id)
        else:
            logger.warning("Database service unavailable; unable to send commentary history")

        # Start subscription task to listen to Redis Pub/Sub
        logger.info(f"Starting subscription task for connection {connection_id} on match {match_id}")
        subscriber_task = asyncio.create_task(
            self._subscribe_and_forward(connection_id, match_id)
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
    async def _send_commentary_history(self, connection_id: str, match_id: str):
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
        if connection_id not in self.connections:
            return
        
        match_id = self.connection_metadata[connection_id]["match_id"]
        subscriber_task = self.connection_metadata[connection_id].get("subscriber_task")
        
        # Cancel subscription task
        if subscriber_task and not subscriber_task.done():
            subscriber_task.cancel()
            try:
                await subscriber_task
            except asyncio.CancelledError:
                pass
        
        # Remove from Redis subscriber set
        subscriber_count = await self.redis_service.remove_subscriber(match_id, connection_id)
        logger.info(
            f"WebSocket disconnected: {connection_id} from match {match_id} "
            f"(remaining subscribers: {subscriber_count})"
        )
        
        # Remove connection
        del self.connections[connection_id]
        del self.connection_metadata[connection_id]
        
        # Note: Worker cleanup is handled by supervisor's cleanup loop
        # when subscriber count reaches zero

    async def _subscribe_and_forward(self, connection_id: str, match_id: str):
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
            async for message in self.redis_service.subscribe_to_match(match_id):
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
            
            await websocket.send_json(message_dict)
            logger.debug(f"✅ Successfully sent message to WebSocket {connection_id}")
        except Exception as e:
            logger.error(f"❌ Error sending message to {connection_id}: {e}", exc_info=True)
            # Connection might be closed, clean up
            await self.disconnect(connection_id)

    async def handle_websocket(self, websocket: WebSocket, match_id: str):
        """
        Handle WebSocket connection lifecycle
        
        Args:
            websocket: WebSocket connection
            match_id: Match identifier
        """
        connection_id = None
        
        try:
            # Connect
            connection_id = await self.connect(websocket, match_id)
            
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

    async def disconnect_all_for_match(self, match_id: str, delay_seconds: float = 2.0):
        """
        Disconnect all WebSocket connections for a match
        Useful when match is finished
        
        Args:
            match_id: Match identifier
            delay_seconds: Delay before disconnecting (to ensure final message is sent)
        """
        # Find all connections for this match
        connections_to_disconnect = [
            conn_id for conn_id, meta in self.connection_metadata.items()
            if meta["match_id"] == match_id
        ]
        
        if not connections_to_disconnect:
            logger.debug(f"No connections to disconnect for match {match_id}")
            return
        
        logger.info(f"Disconnecting {len(connections_to_disconnect)} WebSocket connections for match {match_id} (after {delay_seconds}s delay)")
        
        # Wait a bit to ensure final message is sent and received
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        
        # Disconnect all connections
        for connection_id in connections_to_disconnect:
            try:
                await self.disconnect(connection_id)
                logger.info(f"Disconnected connection {connection_id} for finished match {match_id}")
            except Exception as e:
                logger.error(f"Error disconnecting {connection_id} for match {match_id}: {e}")
        
        logger.info(f"✅ Disconnected all {len(connections_to_disconnect)} connections for match {match_id}")

