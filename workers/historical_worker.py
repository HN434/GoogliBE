"""
Historical Commentary Worker
Fetches and stores older commentary pages until exhausted
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Tuple, List

from services.commentary_service import CommentaryService
from database.db_service import db_service
from models.commentary_schemas import CommentaryLine

logger = logging.getLogger(__name__)


class HistoricalCommentaryWorker:
    """
    Worker that backfills historical commentary for a match
    """

    def __init__(
        self,
        match_id: str,
        commentary_service: CommentaryService,
        fetch_delay: float = 1.0,
        max_iterations: Optional[int] = None,
    ):
        self.match_id = match_id
        self.commentary_service = commentary_service
        self.fetch_delay = fetch_delay
        self.max_iterations = max_iterations

        self.is_running = False
        self.task: Optional[asyncio.Task] = None
        self.error_count = 0
        self.pages_fetched = 0
        self.commentaries_stored = 0

    async def start(self):
        if self.is_running:
            logger.debug(f"Historical worker for match {self.match_id} already running")
            return

        self.is_running = True
        self.task = asyncio.create_task(self._run())
        logger.info(f"🕰️ Started historical worker for match {self.match_id}")

    async def stop(self):
        if not self.is_running:
            return
        self.is_running = False
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info(f"🛑 Stopped historical worker for match {self.match_id}")

    async def _run(self):
        try:
            await self._backfill_commentary()
        except asyncio.CancelledError:
            logger.info(f"Historical worker for match {self.match_id} cancelled")
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error in historical worker for match {self.match_id}: {e}", exc_info=True)
        finally:
            self.is_running = False

    async def _backfill_commentary(self):
        """
        Fetch and store historical commentary pages until exhausted
        """
        logger.info(f"Fetching historical commentary for match {self.match_id}")

        # Determine starting point (earliest stored commentary)
        oldest_info = await db_service.get_oldest_commentary_info(self.match_id)

        next_timestamp_ms = None
        next_innings_id = None

        if oldest_info:
            next_timestamp_ms = oldest_info.get("timestamp_ms")
            next_innings_id = oldest_info.get("innings_id")
        else:
            # No records stored yet; fetch latest page first
            latest_batch, _ = await self.commentary_service.fetch_commentary(self.match_id)
            stored = await db_service.store_commentaries(self.match_id, latest_batch)
            self.commentaries_stored += stored
            if latest_batch:
                next_timestamp_ms, next_innings_id = self._get_oldest_pagination_params(latest_batch)
            else:
                logger.info(f"No commentary data available for match {self.match_id}")
                return

        iterations = 0

        while self.is_running:
            if next_timestamp_ms is None:
                logger.info(f"No more pagination parameters for match {self.match_id}")
                break

            # Prevent infinite loops
            if self.max_iterations is not None and iterations >= self.max_iterations:
                logger.info(f"Reached max iterations for match {self.match_id}")
                break

            batch, _ = await self.commentary_service.fetch_previous_commentaries(
                self.match_id,
                timestamp_ms=next_timestamp_ms,
                innings_id=next_innings_id,
            )

            iterations += 1
            self.pages_fetched += 1

            if not batch:
                logger.info(f"No more historical commentary pages for match {self.match_id}")
                break

            stored = await db_service.store_commentaries(self.match_id, batch)
            self.commentaries_stored += stored

            if stored == 0:
                logger.info(f"No new commentaries stored for match {self.match_id}, stopping historical worker")
                break

            next_timestamp_ms, next_innings_id = self._get_oldest_pagination_params(batch)

            # Add delay between API calls
            await asyncio.sleep(self.fetch_delay)

        logger.info(
            f"Historical worker for match {self.match_id} completed: "
            f"{self.pages_fetched} pages fetched, {self.commentaries_stored} commentaries stored"
        )

    def _get_oldest_pagination_params(
        self, lines: List[CommentaryLine]
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Determine pagination parameters for next API call based on oldest commentary line
        """
        if not lines:
            return None, None

        oldest_line = min(lines, key=lambda line: line.timestamp)
        metadata = oldest_line.metadata or {}
        timestamp_ms = metadata.get("timestamp_ms")
        innings_id = metadata.get("inningsid")

        if timestamp_ms is None:
            # Fall back to converting datetime to milliseconds
            ts = oldest_line.timestamp
            if isinstance(ts, datetime):
                timestamp_ms = int(ts.timestamp() * 1000)

        return timestamp_ms, innings_id

