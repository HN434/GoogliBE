"""
Redis Service for Cricket Commentary System
Manages Redis connections, pub/sub channels, and shared state
"""

import redis.asyncio as aioredis
from typing import Optional, Set, AsyncIterator
import json
import logging
from datetime import datetime
from config import settings

logger = logging.getLogger(__name__)


class RedisService:
    """
    Redis service for managing:
    - Subscriber tracking (Redis Sets)
    - Worker status (Redis Keys)
    - Pub/Sub channels for broadcasting commentary
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize Redis service
        
        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self.redis_client: Optional[aioredis.Redis] = None
        self.pubsub_client: Optional[aioredis.Redis] = None
        self.default_language = (settings.COMMENTARY_DEFAULT_LANGUAGE or "en").lower()

    async def connect(self):
        """Establish Redis connections with timeout"""
        import time
        import asyncio
        start_time = time.time()
        
        try:
            # Main Redis client for regular operations with connection timeout
            self.redis_client = await asyncio.wait_for(
                aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=5,  # 5 second connection timeout
                    socket_timeout=5,  # 5 second socket timeout
                ),
                timeout=10.0  # Overall timeout for connection
            )
            
            # Separate client for pub/sub (required by Redis)
            self.pubsub_client = await asyncio.wait_for(
                aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                ),
                timeout=10.0
            )
            
            # Test connection with timeout
            await asyncio.wait_for(self.redis_client.ping(), timeout=5.0)
            
            connect_time = time.time() - start_time
            logger.info(f"✅ Redis connection established in {connect_time:.2f}s")
            
        except asyncio.TimeoutError:
            logger.error("❌ Redis connection timeout after 10 seconds")
            raise ConnectionError("Redis connection timeout")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise

    async def disconnect(self):
        """Close Redis connections"""
        try:
            if self.pubsub_client:
                await self.pubsub_client.close()
            
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("✅ Redis connections closed")
        except Exception as e:
            logger.error(f"Error closing Redis connections: {e}")

    # ===== Subscriber Management =====

    def _normalize_language(self, language: Optional[str]) -> str:
        lang = (language or self.default_language).strip().lower()
        return lang or self.default_language

    def _channel_name(self, match_id: str, language: Optional[str] = None) -> str:
        lang = self._normalize_language(language)
        if lang == self.default_language:
            return f"channel:match:{match_id}"
        return f"channel:match:{match_id}:lang:{lang}"

    def _language_subscriber_key(self, match_id: str) -> str:
        return f"match:{match_id}:language_subscribers"

    def _translation_cache_key(self, match_id: str, language: str) -> str:
        lang = self._normalize_language(language)
        return f"match:{match_id}:translations:{lang}"

    async def add_subscriber(self, match_id: str, subscriber_id: str) -> int:
        """
        Add a subscriber to a match
        
        Args:
            match_id: Match identifier
            subscriber_id: Unique subscriber identifier (e.g., WebSocket connection ID)
        
        Returns:
            New subscriber count for the match
        """
        key = f"match:{match_id}:subscribers"
        await self.redis_client.sadd(key, subscriber_id)
        count = await self.redis_client.scard(key)
        logger.debug(f"Added subscriber {subscriber_id} to match {match_id}. Total: {count}")
        return count

    async def remove_subscriber(self, match_id: str, subscriber_id: str) -> int:
        """
        Remove a subscriber from a match
        
        Args:
            match_id: Match identifier
            subscriber_id: Subscriber identifier to remove
        
        Returns:
            Remaining subscriber count for the match
        """
        key = f"match:{match_id}:subscribers"
        await self.redis_client.srem(key, subscriber_id)
        count = await self.redis_client.scard(key)
        logger.debug(f"Removed subscriber {subscriber_id} from match {match_id}. Remaining: {count}")
        return count

    async def add_language_subscriber(self, match_id: str, language: Optional[str]) -> int:
        """
        Increment subscriber count for a language on a match.

        Returns current count for that language.
        """
        lang = self._normalize_language(language)
        key = self._language_subscriber_key(match_id)
        new_count = await self.redis_client.hincrby(key, lang, 1)
        logger.debug(f"Language subscriber +1 for match {match_id}, lang={lang}, total={new_count}")
        return new_count

    async def remove_language_subscriber(self, match_id: str, language: Optional[str]) -> int:
        """
        Decrement subscriber count for a language.

        Returns remaining count for that language (>=0).
        """
        lang = self._normalize_language(language)
        key = self._language_subscriber_key(match_id)
        remaining = await self.redis_client.hincrby(key, lang, -1)
        if remaining <= 0:
            await self.redis_client.hdel(key, lang)
            remaining = 0
        logger.debug(f"Language subscriber -1 for match {match_id}, lang={lang}, remaining={remaining}")
        return remaining

    async def get_active_languages(self, match_id: str) -> Set[str]:
        """
        Return set of languages that currently have at least one subscriber.
        """
        key = self._language_subscriber_key(match_id)
        lang_counts = await self.redis_client.hgetall(key)
        if not lang_counts:
            return {self.default_language}
        active = {
            lang
            for lang, count in lang_counts.items()
            if count and int(count) > 0
        }
        if not active:
            active.add(self.default_language)
        return active

    async def cache_translation(
        self,
        match_id: str,
        language: str,
        line_id: str,
        translated_text: str,
        ttl_seconds: int = 900,
    ):
        """
        Cache translated commentary text for a specific line and language.
        """
        key = self._translation_cache_key(match_id, language)
        pipe = self.redis_client.pipeline()
        pipe.hset(key, line_id, translated_text)
        if ttl_seconds > 0:
            pipe.expire(key, ttl_seconds)
        await pipe.execute()

    async def get_cached_translation(self, match_id: str, language: str, line_id: str) -> Optional[str]:
        """
        Fetch cached translation for a commentary line if available.
        """
        key = self._translation_cache_key(match_id, language)
        return await self.redis_client.hget(key, line_id)

    async def get_subscriber_count(self, match_id: str) -> int:
        """
        Get the number of subscribers for a match
        
        Args:
            match_id: Match identifier
        
        Returns:
            Number of active subscribers
        """
        key = f"match:{match_id}:subscribers"
        return await self.redis_client.scard(key)

    async def get_all_subscribers(self, match_id: str) -> Set[str]:
        """
        Get all subscriber IDs for a match
        
        Args:
            match_id: Match identifier
        
        Returns:
            Set of subscriber IDs
        """
        key = f"match:{match_id}:subscribers"
        members = await self.redis_client.smembers(key)
        return members

    # ===== Worker Status Management =====

    async def set_worker_running(self, match_id: str, is_running: bool):
        """
        Set worker running status
        
        Args:
            match_id: Match identifier
            is_running: Whether the worker is running
        """
        key = f"match:{match_id}:worker_running"
        if is_running:
            await self.redis_client.set(key, "1")
        else:
            await self.redis_client.delete(key)
        logger.debug(f"Set worker status for match {match_id}: {is_running}")

    async def is_worker_running(self, match_id: str) -> bool:
        """
        Check if a worker is running for a match
        
        Args:
            match_id: Match identifier
        
        Returns:
            True if worker is running
        """
        key = f"match:{match_id}:worker_running"
        value = await self.redis_client.get(key)
        return value == "1"

    # ===== Pub/Sub Operations =====

    async def publish_commentary(self, match_id: str, message: dict, language: Optional[str] = None):
        """
        Publish commentary update to Redis Pub/Sub channel
        
        Args:
            match_id: Match identifier
            message: Commentary update message (dict)
        """
        channel = self._channel_name(match_id, language)
        try:
            # Serialize with proper datetime handling
            message_json = json.dumps(message, default=str)  # default=str handles datetime serialization
            subscribers = await self.get_subscriber_count(match_id)
            logger.info(f"📢 Publishing to channel {channel} (subscribers: {subscribers})")
            logger.debug(f"Message preview: {str(message)[:200]}...")
            
            result = await self.pubsub_client.publish(channel, message_json)
            logger.info(f"✅ Published to {channel}, {result} subscribers notified")
        except Exception as e:
            logger.error(f"❌ Error publishing to channel {channel}: {e}", exc_info=True)
            raise

    async def subscribe_to_match(self, match_id: str, language: Optional[str] = None) -> AsyncIterator[dict]:
        """
        Subscribe to commentary updates for a match
        
        Args:
            match_id: Match identifier
        
        Yields:
            Commentary update messages as dictionaries
        """
        channel = self._channel_name(match_id, language)
        
        # Create a new pubsub instance for this subscription
        # Each subscription needs its own pubsub instance
        pubsub = self.pubsub_client.pubsub()
        
        try:
            # Subscribe to the channel
            await pubsub.subscribe(channel)
            logger.info(f"Subscribed to channel: {channel}")
            
            async for message in pubsub.listen():
                # Skip subscription confirmation messages
                if message["type"] == "subscribe":
                    continue
                
                # Skip non-message types
                if message["type"] != "message":
                    continue
                
                # Parse JSON message
                try:
                    data = json.loads(message["data"])
                    yield data
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message from {channel}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in subscription to {channel}: {e}")
            raise
        finally:
            # Unsubscribe and close pubsub when done
            try:
                await pubsub.unsubscribe(channel)
                await pubsub.close()
            except:
                pass
            logger.info(f"Unsubscribed from channel: {channel}")

    async def unsubscribe_from_match(self, match_id: str):
        """
        Unsubscribe from a match channel
        Note: This method is kept for API compatibility but actual unsubscription
        is handled in subscribe_to_match's finally block
        
        Args:
            match_id: Match identifier
        """
        # Unsubscription is handled per-pubsub instance in subscribe_to_match
        pass

    # ===== Utility Methods =====

    async def cleanup_match_data(self, match_id: str):
        """
        Clean up all Redis data for a match
        
        Args:
            match_id: Match identifier
        """
        keys_to_delete = [
            f"match:{match_id}:subscribers",
            f"match:{match_id}:worker_running",
        ]
        
        for key in keys_to_delete:
            await self.redis_client.delete(key)
        
        logger.info(f"Cleaned up Redis data for match {match_id}")

    async def get_all_active_matches(self) -> Set[str]:
        """
        Get all match IDs that have active subscribers
        
        Returns:
            Set of match IDs with at least one subscriber
        """
        pattern = "match:*:subscribers"
        keys = await self.redis_client.keys(pattern)
        match_ids = set()
        
        for key in keys:
            # Extract match_id from key pattern "match:{match_id}:subscribers"
            parts = key.split(":")
            if len(parts) >= 2:
                match_ids.add(parts[1])
        
        return match_ids


# Global Redis service instance
redis_service = RedisService()

