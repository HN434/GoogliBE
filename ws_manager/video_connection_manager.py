"""
Video Analysis WebSocket Connection Manager
Handles WebSocket connections for video analysis results
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, Optional
from fastapi import WebSocket, WebSocketDisconnect
from config import settings

logger = logging.getLogger(__name__)


class VideoWebSocketManager:
    """
    Manages WebSocket connections for video analysis
    - Tracks active connections per video_id
    - Sends analysis results (keypoints and bedrock analytics) to clients
    """

    def __init__(self):
        """Initialize video WebSocket connection manager"""
        # Track active connections: video_id -> WebSocket
        self.connections: Dict[str, WebSocket] = {}
        
        # Track connection metadata: video_id -> connection_id
        self.connection_metadata: Dict[str, str] = {}

    async def connect(self, websocket: WebSocket, video_id: str) -> str:
        """
        Accept WebSocket connection for a video
        
        Args:
            websocket: WebSocket connection
            video_id: Video identifier
            
        Returns:
            Connection ID
        """
        await websocket.accept()
        
        # Generate unique connection ID
        connection_id = str(uuid.uuid4())
        
        # Store connection (one connection per video_id)
        if video_id in self.connections:
            # Close existing connection if any
            try:
                await self.connections[video_id].close()
            except:
                pass
        
        self.connections[video_id] = websocket
        self.connection_metadata[video_id] = connection_id
        
        logger.info(f"Video WebSocket connected: {connection_id} for video {video_id}")
        
        # Send welcome message
        try:
            await self._send_message(
                video_id,
                {
                    "type": "status",
                    "video_id": video_id,
                    "message": "Connected to video analysis stream",
                    "connection_id": connection_id
                }
            )
        except Exception as e:
            logger.error(f"Failed to send welcome message: {e}", exc_info=True)
        
        return connection_id

    async def disconnect(self, video_id: str):
        """
        Disconnect WebSocket for a video
        
        Args:
            video_id: Video identifier
        """
        if video_id not in self.connections:
            return
        
        try:
            websocket = self.connections[video_id]
            await websocket.close()
        except Exception as e:
            logger.warning(f"Error closing WebSocket for video {video_id}: {e}")
        
        # Remove connection
        if video_id in self.connections:
            del self.connections[video_id]
        if video_id in self.connection_metadata:
            del self.connection_metadata[video_id]
        
        logger.info(f"Video WebSocket disconnected for video {video_id}")

    async def send_keypoints(self, video_id: str, keypoints_data: dict):
        """
        Send keypoints (video overlay JSON) to client
        
        Args:
            video_id: Video identifier
            keypoints_data: Keypoints JSON data
        """
        await self._send_message(
            video_id,
            {
                "type": "keypoints",
                "video_id": video_id,
                "data": keypoints_data,
                "done": False
            }
        )

    async def send_bedrock_analysis(self, video_id: str, bedrock_analysis: dict):
        """
        Send Bedrock analysis to client
        
        Args:
            video_id: Video identifier
            bedrock_analysis: Bedrock analysis data
        """
        await self._send_message(
            video_id,
            {
                "type": "bedrock_analysis",
                "video_id": video_id,
                "data": bedrock_analysis,
                # 'done' is managed at the Redis publisher level; here we just forward
                "done": bedrock_analysis.get("done", False) if isinstance(bedrock_analysis, dict) else False,
            }
        )

    async def _send_message(self, video_id: str, message: dict):
        """
        Send message to WebSocket connection
        
        Args:
            video_id: Video identifier
            message: Message to send
        """
        if video_id not in self.connections:
            logger.warning(f"No WebSocket connection found for video {video_id}")
            return
        
        websocket = self.connections[video_id]
        
        try:
            await websocket.send_text(json.dumps(message, ensure_ascii=False))
            logger.debug(f"Sent message to video {video_id}: type={message.get('type')}")
        except Exception as e:
            logger.error(f"Error sending message to video {video_id}: {e}", exc_info=True)
            # Connection might be closed, clean up
            await self.disconnect(video_id)

    async def handle_websocket(self, websocket: WebSocket, video_id: str):
        """
        Handle WebSocket connection lifecycle
        
        Args:
            websocket: WebSocket connection
            video_id: Video identifier
        """
        connection_id = None
        ping_task = None
        
        try:
            # Connect
            connection_id = await self.connect(websocket, video_id)
            
            # Start keepalive ping task (sends ping periodically to prevent timeout)
            ping_interval = getattr(settings, 'WEBSOCKET_PING_INTERVAL_SECONDS', 30)
            
            async def send_keepalive_ping():
                """Send periodic ping messages to keep connection alive"""
                try:
                    while video_id in self.connections:
                        await asyncio.sleep(ping_interval)
                        if video_id in self.connections:
                            try:
                                # Send a ping frame to keep connection alive
                                await websocket.send_text(json.dumps({
                                    "type": "ping",
                                    "video_id": video_id,
                                    "timestamp": asyncio.get_event_loop().time()
                                }))
                                logger.debug(f"Sent keepalive ping to video {video_id}")
                            except Exception as e:
                                logger.debug(f"Failed to send keepalive ping: {e}")
                                break
                except asyncio.CancelledError:
                    logger.debug(f"Keepalive ping task cancelled for video {video_id}")
                except Exception as e:
                    logger.error(f"Error in keepalive ping task: {e}", exc_info=True)
            
            ping_task = asyncio.create_task(send_keepalive_ping())
            
            # Keep connection alive and handle any incoming messages
            while True:
                try:
                    # Wait for any message from client (ping/pong, etc.)
                    # Use configurable timeout to prevent connection from timing out
                    timeout_seconds = getattr(settings, 'WEBSOCKET_TIMEOUT_SECONDS', 3600)
                    try:
                        message = await asyncio.wait_for(websocket.receive(), timeout=float(timeout_seconds))
                    except asyncio.TimeoutError:
                        # Timeout reached, but connection is still alive
                        # Send a ping to verify connection is still active
                        if video_id in self.connections:
                            try:
                                await websocket.send_text(json.dumps({
                                    "type": "ping",
                                    "video_id": video_id
                                }))
                                logger.debug(f"Sent timeout ping to video {video_id}")
                                continue
                            except Exception as e:
                                logger.info(f"Connection appears closed during timeout ping: {e}")
                                break
                        else:
                            break
                    
                    if message["type"] == "websocket.disconnect":
                        logger.info(f"WebSocket {connection_id} disconnected by client")
                        break
                    elif message["type"] == "websocket.receive":
                        if "text" in message:
                            logger.debug(f"Received text message from {connection_id}: {message['text']}")
                        elif "bytes" in message:
                            logger.debug(f"Received binary message from {connection_id}")
                    
                except WebSocketDisconnect:
                    logger.info(f"WebSocket {connection_id} disconnected")
                    break
                except Exception as e:
                    logger.error(f"Error receiving message from {connection_id}: {e}", exc_info=True)
                    break
                    
        except Exception as e:
            logger.error(f"Error handling WebSocket for video {video_id}: {e}")
        finally:
            # Cancel keepalive ping task
            if ping_task and not ping_task.done():
                ping_task.cancel()
                try:
                    await ping_task
                except asyncio.CancelledError:
                    pass
            
            # Clean up on disconnect
            if video_id:
                await self.disconnect(video_id)

    def has_connection(self, video_id: str) -> bool:
        """
        Check if there's an active connection for a video
        
        Args:
            video_id: Video identifier
            
        Returns:
            True if connection exists, False otherwise
        """
        return video_id in self.connections

