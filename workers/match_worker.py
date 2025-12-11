"""
Match Worker - Fetches commentary for a single match every 20 seconds
Publishes new commentary events to Redis Pub/Sub
"""

import asyncio
import logging
from typing import Set, Optional, Dict, Any, List
from datetime import datetime
from services.commentary_service import CommentaryService
from services.redis_service import RedisService
from models.commentary_schemas import CommentaryLine, CommentaryUpdate
from database.db_service import db_service

logger = logging.getLogger(__name__)


class MatchWorker:
    """
    Worker that fetches commentary for a single match
    Runs every 20 seconds, computes deltas, and publishes updates
    """

    RECENT_SYNC_LIMIT = 20

    def __init__(
        self,
        match_id: str,
        commentary_service: CommentaryService,
        redis_service: RedisService,
        fetch_interval: int = 20,
        ws_manager=None,
        translation_service=None,
    ):
        """
        Initialize match worker
        
        Args:
            match_id: Match identifier
            commentary_service: Commentary service instance
            redis_service: Redis service instance
            fetch_interval: Seconds between fetches (default: 20)
            ws_manager: Optional WebSocket manager for disconnecting clients when match finishes
        """
        self.match_id = match_id
        self.commentary_service = commentary_service
        self.redis_service = redis_service
        self.fetch_interval = fetch_interval
        self.ws_manager = ws_manager
        self.translation_service = translation_service
        
        # Track seen commentary keys (match_id + timestamp) to compute deltas
        self.seen_commentary_keys: Set[str] = set()
        self.last_miniscore: Optional[Dict[str, Any]] = None
        
        # Worker state
        self.is_running = False
        
        self.task: Optional[asyncio.Task] = None
        self.error_count = 0
        self.last_fetch: Optional[datetime] = None
        
        # Track when subscriber count becomes zero to add grace period
        self.zero_subscribers_since: Optional[datetime] = None
        self.zero_subscribers_grace_period: int = 5  # seconds to wait before stopping
        self.in_grace_period: bool = False  # Flag to indicate worker is in shutdown grace period

    async def start(self):
        """Start the worker"""
        if self.is_running:
            logger.warning(f"Worker for match {self.match_id} is already running")
            return
        
        self.is_running = True
        await self.redis_service.set_worker_running(self.match_id, True)
        
        # Start worker task
        self.task = asyncio.create_task(self._run())
        logger.info(f"✅ Started worker for match {self.match_id}")

    async def stop(self):
        """Stop the worker"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.in_grace_period = False  # Clear grace period when stopping
        await self.redis_service.set_worker_running(self.match_id, False)
        
        # Cancel task if running
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"🛑 Stopped worker for match {self.match_id}")

    async def _run(self):
        """
        Main worker loop
        Fetches commentary every 20 seconds until stopped
        """
        logger.info(f"Worker loop started for match {self.match_id}")
        
        # Do immediate fetch on start (don't wait 20 seconds)
        try:
            logger.info(f"Performing initial fetch for match {self.match_id}")
            await self._fetch_and_publish()
        except Exception as e:
            logger.error(f"Error in initial fetch for match {self.match_id}: {e}", exc_info=True)
        
        while self.is_running:
            try:
                # Wait before next fetch
                logger.debug(f"Worker for match {self.match_id} sleeping for {self.fetch_interval} seconds")
                await asyncio.sleep(self.fetch_interval)
                
                if not self.is_running:
                    break
                
                # Check if match is finished
                logger.debug(f"Checking if match {self.match_id} is finished")
                is_finished = await self.commentary_service.is_match_finished(self.match_id)
                if is_finished:
                    logger.info(f"Match {self.match_id} is finished. Sending final status and stopping worker.")
                    
                    # Get match status and result
                    match_status_info = await self.commentary_service.get_match_status(self.match_id)
                    
                    # Send final status message to subscribers
                    try:
                        from models.commentary_schemas import CommentaryUpdate
                        
                        # Prepare match status data
                        status_data = {
                            "state": "finished",
                            "status": match_status_info.get("status", "Match finished") if match_status_info else "Match finished",
                            "complete": True
                        }
                        
                        # Add additional match information if available
                        if match_status_info:
                            status_data.update({
                                "state": match_status_info.get("state", "finished"),
                                "status": match_status_info.get("status", "Match finished"),
                                "winning_team": match_status_info.get("winning_team"),
                                "winning_team_id": match_status_info.get("winning_team_id"),
                                "team1": match_status_info.get("team1"),
                                "team2": match_status_info.get("team2"),
                                "match_format": match_status_info.get("match_format"),
                                "series_name": match_status_info.get("series_name"),
                                "match_desc": match_status_info.get("match_desc")
                            })
                        
                        final_update = CommentaryUpdate(
                            match_id=self.match_id,
                            timestamp=datetime.now(),
                            lines=[],
                            match_status=status_data.get("status", "finished"),
                            score=status_data,  # Include full status in score field for backward compatibility
                            language=self.redis_service.default_language,
                        )
                        
                        # Add status_data to the update
                        update_dict = final_update.model_dump(mode="json", exclude_none=True)
                        update_dict["language"] = self.redis_service.default_language
                        update_dict["match_status_info"] = status_data
                        if self.last_miniscore:
                            update_dict["miniscore"] = self.last_miniscore
                            status_data.setdefault("miniscore", self.last_miniscore)
                        
                        await self.redis_service.publish_commentary(
                            self.match_id,
                            update_dict
                        )
                        if self.translation_service:
                            await self._fan_out_translations(update_dict)
                        logger.info(f"✅ Sent final status message for match {self.match_id}: {status_data.get('status', 'finished')}")
                        
                        # Update match status in database
                        await db_service.update_match_status(self.match_id, status_data)

                        # Disconnect all WebSocket clients for this match after sending final message
                        # Pass worker_id so only THIS worker's connections are disconnected
                        if self.ws_manager:
                            worker_id = id(self)  # Unique ID for this worker instance
                            logger.info(
                                f"Disconnecting WebSocket clients for finished match {self.match_id} "
                                f"(worker_id={worker_id})"
                            )
                            # Use asyncio.create_task to not block worker shutdown
                            asyncio.create_task(
                                self.ws_manager.disconnect_all_for_match(
                                    self.match_id, 
                                    delay_seconds=2.0,
                                    worker_id=worker_id
                                )
                            )
                        else:
                            logger.warning(f"WebSocket manager not available, cannot disconnect clients for match {self.match_id}")
                            
                    except Exception as e:
                        logger.error(f"❌ Error sending final status for match {self.match_id}: {e}", exc_info=True)
                    
                    await self.stop()
                    break
                
                # Check subscriber count with grace period to avoid race conditions
                subscriber_count = await self.redis_service.get_subscriber_count(self.match_id)
                logger.debug(f"Match {self.match_id} has {subscriber_count} subscribers")
                
                if subscriber_count == 0:
                    # Track when we first saw zero subscribers
                    if self.zero_subscribers_since is None:
                        self.zero_subscribers_since = datetime.now()
                        self.in_grace_period = True  # Mark as in grace period
                        logger.info(
                            f"⏳ No subscribers for match {self.match_id}. "
                            f"Starting grace period of {self.zero_subscribers_grace_period}s... "
                            f"(Worker marked as closing - new connections should create new worker)"
                        )
                    else:
                        # Check if grace period has elapsed
                        elapsed = (datetime.now() - self.zero_subscribers_since).total_seconds()
                        if elapsed >= self.zero_subscribers_grace_period:
                            logger.info(
                                f"⏱️ No subscribers for match {self.match_id} after {elapsed:.1f}s grace period. "
                                f"Stopping worker."
                            )
                            await self.stop()
                            break
                        else:
                            logger.debug(
                                f"Grace period active for match {self.match_id}: {elapsed:.1f}s / "
                                f"{self.zero_subscribers_grace_period}s"
                            )
                else:
                    # Reset grace period if we have subscribers again
                    if self.zero_subscribers_since is not None:
                        logger.info(
                            f"🔄 Match {self.match_id} regained subscribers ({subscriber_count}). "
                            f"Resetting grace period."
                        )
                        self.zero_subscribers_since = None
                        self.in_grace_period = False  # Clear grace period flag
                
                # Fetch commentary
                logger.debug(f"Fetching commentary for match {self.match_id}")
                await self._fetch_and_publish()
                
            except asyncio.CancelledError:
                logger.info(f"Worker for match {self.match_id} was cancelled")
                break
            except Exception as e:
                self.error_count += 1
                logger.error(f"Error in worker loop for match {self.match_id}: {e}")
                # Continue running even on error, but wait before retry
                await asyncio.sleep(self.fetch_interval)

    async def _fetch_and_publish(self):
        """
        Fetch commentary, compute deltas, and publish to Redis
        """
        try:
            logger.debug(f"Starting fetch for match {self.match_id}")
            # Fetch all commentary and miniscore
            all_commentary, miniscore = await self.commentary_service.fetch_commentary(self.match_id)
            logger.debug(f"Fetched {len(all_commentary)} total commentary lines for match {self.match_id}")
            if miniscore:
                self.last_miniscore = miniscore

            # Ensure last RECENT_SYNC_LIMIT balls are in sync with DB text
            recent_lines = self._select_recent_lines(all_commentary, limit=self.RECENT_SYNC_LIMIT)
            recent_updates = 0
            if recent_lines:
                try:
                    recent_updates = await db_service.sync_recent_commentaries(self.match_id, recent_lines)
                    if recent_updates:
                        logger.debug(f"Synced {recent_updates} recent commentary lines for match {self.match_id}")
                except Exception as e:
                    logger.error(f"Error syncing recent commentaries for match {self.match_id}: {e}", exc_info=True)
            
            # Compute delta: only new commentary lines
            new_commentary = [
                line for line in all_commentary
                if self._get_commentary_key(line) not in self.seen_commentary_keys
            ]
            
            logger.debug(
                f"Match {self.match_id}: {len(new_commentary)} new lines (recent updates: {recent_updates}) "
                f"out of {len(all_commentary)} fetched (seen: {len(self.seen_commentary_keys)})"
            )

            if new_commentary:
                # Update seen IDs
                for line in new_commentary:
                    self.seen_commentary_keys.add(self._get_commentary_key(line))

            changes_detected = bool(new_commentary or recent_updates)

            if not changes_detected:
                logger.debug(
                    f"No new or updated commentary for match {self.match_id} "
                    f"(checked {len(all_commentary)} lines)"
                )
                self.last_fetch = datetime.now()
                self.error_count = 0
                return

            # Get actual match status from API (single fetch per change batch)
            match_status_info = await self.commentary_service.get_match_status(self.match_id)

            # Determine match status string
            if match_status_info:
                match_status_str = match_status_info.get("status", "")
                state = match_status_info.get("state", "").upper()
                if state in ["COMPLETE", "FINISHED", "STUMPS"]:
                    match_status_str = match_status_info.get("status", "Match finished")
                elif not match_status_str:
                    match_status_str = "live"
            else:
                match_status_str = "live"

            combined_status_info = dict(match_status_info or {})

            # Store only genuinely new commentaries
            if new_commentary:
                try:
                    stored_count = await db_service.store_commentaries(
                        self.match_id,
                        new_commentary,
                        combined_status_info or None
                    )
                    logger.debug(f"Stored {stored_count} new commentaries in database for match {self.match_id}")
                except Exception as e:
                    logger.error(f"Error storing commentaries in database for match {self.match_id}: {e}", exc_info=True)

            logger.debug(f"Match {self.match_id} status: {match_status_str}")

            # Create update message that includes the full current feed
            update = CommentaryUpdate(
                match_id=self.match_id,
                timestamp=datetime.now(),
                lines=all_commentary,
                match_status=match_status_str,
                language=self.redis_service.default_language,
            )

            update_dict = update.model_dump(mode="json", exclude_none=True)
            if miniscore:
                update_dict["miniscore"] = miniscore
            if combined_status_info:
                update_dict["match_status_info"] = combined_status_info
            update_dict["change_summary"] = {
                "new_count": len(new_commentary),
                "recent_updates": recent_updates
            }

            logger.debug(
                f"Publishing {len(all_commentary)} commentary lines (new: {len(new_commentary)}, "
                f"recent updates: {recent_updates}) to Redis for match {self.match_id}"
            )
            await self.redis_service.publish_commentary(
                self.match_id,
                update_dict
            )

            if self.translation_service:
                await self._fan_out_translations(update_dict)
            
            logger.info(
                f"✅ Published commentary snapshot for match {self.match_id} "
                f"(new: {len(new_commentary)}, updated: {recent_updates})"
            )
            self.last_fetch = datetime.now()
            self.error_count = 0  # Reset error count on success
            logger.debug(f"Fetch completed successfully for match {self.match_id}")
            
        except Exception as e:
            self.error_count += 1
            logger.error(
                f"❌ Error fetching/publishing commentary for match {self.match_id}: {e}",
                exc_info=True
            )
            # Don't raise - let the worker continue and retry
            self.last_fetch = datetime.now()

    def get_status(self) -> dict:
        """
        Get worker status
        
        Returns:
            Dictionary with worker status information
        """
        return {
            "match_id": self.match_id,
            "is_running": self.is_running,
            "in_grace_period": self.in_grace_period,
            "last_fetch": self.last_fetch.isoformat() if self.last_fetch else None,
            "error_count": self.error_count,
            "seen_commentary_count": len(self.seen_commentary_keys)
        }

    def _get_commentary_key(self, line: CommentaryLine) -> str:
        """
        Generate a unique key for a commentary line based on match_id and timestamp
        """
        timestamp = line.timestamp
        if isinstance(timestamp, datetime):
            timestamp_str = timestamp.isoformat()
        else:
            timestamp_str = str(timestamp)
        return f"{self.match_id}:{timestamp_str}"

    def _select_recent_lines(
        self,
        lines: List[CommentaryLine],
        limit: int = RECENT_SYNC_LIMIT,
    ) -> List[CommentaryLine]:
        """
        Pick the most recent commentary lines based on timestamp for DB sync.
        """
        if not lines:
            return []

        sorted_lines = sorted(
            lines,
            key=lambda line: line.timestamp or datetime.min,
            reverse=True
        )
        return sorted_lines[:limit]

    async def _fan_out_translations(self, update_payload: dict):
        """
        Translate commentary snapshot for subscribers requesting other languages.
        """
        if not self.translation_service:
            return
        active_languages = await self.redis_service.get_active_languages(self.match_id)
        default_lang = self.redis_service.default_language
        target_languages = [lang for lang in active_languages if lang != default_lang]
        if not target_languages:
            return
        await self.translation_service.translate_and_publish(
            self.match_id,
            update_payload,
            target_languages,
        )