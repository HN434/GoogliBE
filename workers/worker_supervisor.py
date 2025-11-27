"""
Worker Supervisor (Master Process)
Manages lifecycle of match workers - creates, monitors, and cleans up workers
"""

import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime
from workers.match_worker import MatchWorker
from workers.historical_worker import HistoricalCommentaryWorker
from services.commentary_service import CommentaryService
from services.redis_service import RedisService
from database.db_service import db_service

logger = logging.getLogger(__name__)


class WorkerSupervisor:
    """
    Master process that supervises all match workers
    - Tracks running workers
    - Creates workers when needed
    - Cleans up workers when no subscribers
    - Monitors worker health
    """

    def __init__(
        self,
        commentary_service: CommentaryService,
        redis_service: RedisService,
        cleanup_interval: int = 60,
        ws_manager=None,
        enable_historical_commentary: bool = False,
    ):
        """
        Initialize worker supervisor
        
        Args:
            commentary_service: Commentary service instance
            redis_service: Redis service instance
            cleanup_interval: Seconds between cleanup checks (default: 60)
            ws_manager: Optional WebSocket manager for disconnecting clients when matches finish
        """
        self.commentary_service = commentary_service
        self.redis_service = redis_service
        self.cleanup_interval = cleanup_interval
        self.ws_manager = ws_manager
        self.enable_historical_commentary = enable_historical_commentary
        
        # Registry of running workers: match_id -> worker
        self.workers: Dict[str, MatchWorker] = {}
        self.historical_workers: Dict[str, HistoricalCommentaryWorker] = {}
        
        # Supervisor state
        self.is_running = False
        self.supervisor_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the supervisor"""
        if self.is_running:
            logger.warning("Supervisor is already running")
            return
        
        self.is_running = True
        
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("✅ Worker supervisor started")

    async def stop(self):
        """Stop the supervisor and all workers"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop cleanup task
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Stop all workers
        stop_tasks = [worker.stop() for worker in self.workers.values()]
        stop_tasks += [worker.stop() for worker in self.historical_workers.values()]
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        self.workers.clear()
        self.historical_workers.clear()
        logger.info("🛑 Worker supervisor stopped")

    async def ensure_worker(self, match_id: str) -> MatchWorker:
        """
        Ensure a worker is running for a match
        Creates worker if it doesn't exist, otherwise returns existing worker
        
        Args:
            match_id: Match identifier
        
        Returns:
            MatchWorker instance
        """
        # Return existing worker if already running
        if match_id in self.workers:
            worker = self.workers[match_id]
            if worker.is_running:
                return worker
            else:
                # Worker exists but not running, remove it
                del self.workers[match_id]
        
        # Check if worker should be running (has subscribers)
        subscriber_count = await self.redis_service.get_subscriber_count(match_id)
        if subscriber_count == 0:
            logger.debug(f"No subscribers for match {match_id}, not starting worker")
            return None
        
        # Create and start new worker
        logger.info(f"Creating new worker for match {match_id} (subscribers: {subscriber_count})")
        worker = MatchWorker(
            match_id=match_id,
            commentary_service=self.commentary_service,
            redis_service=self.redis_service,
            ws_manager=self.ws_manager,
        )
        
        self.workers[match_id] = worker
        await worker.start()
        
        logger.info(f"✅ Created and started worker for match {match_id}")
        return worker

    async def stop_worker(self, match_id: str):
        """
        Stop and remove a worker for a match
        
        Args:
            match_id: Match identifier
        """
        if match_id in self.workers:
            worker = self.workers[match_id]
            await worker.stop()
            del self.workers[match_id]
            logger.info(f"Stopped and removed worker for match {match_id}")

    async def ensure_historical_worker(self, match_id: str) -> Optional[HistoricalCommentaryWorker]:
        """
        Ensure a historical worker is running for a match (backfills commentaries)
        """
        if not self.enable_historical_commentary:
            logger.debug(
                "Historical commentary disabled via feature flag; skipping worker for match %s",
                match_id,
            )
            return None
        if match_id in self.historical_workers:
            worker = self.historical_workers[match_id]
            if worker.is_running:
                return worker
            else:
                del self.historical_workers[match_id]

        match_status = await db_service.get_match_status(match_id)
        state = (match_status or {}).get("state", "").upper() if match_status else ""
        if state in {"COMPLETE", "FINISHED"}:
            logger.info(f"Skipping historical worker for match {match_id} because state is {state}")
            return None

        logger.info(f"Starting historical worker for match {match_id}")
        worker = HistoricalCommentaryWorker(
            match_id=match_id,
            commentary_service=self.commentary_service,
        )

        self.historical_workers[match_id] = worker
        await worker.start()
        return worker

    async def stop_historical_worker(self, match_id: str):
        """
        Stop historical worker for a match if running
        """
        if match_id in self.historical_workers:
            worker = self.historical_workers[match_id]
            await worker.stop()
            del self.historical_workers[match_id]
            logger.info(f"Stopped historical worker for match {match_id}")

    async def _cleanup_loop(self):
        """
        Background task that periodically cleans up workers with no subscribers
        """
        logger.info("Cleanup loop started")
        
        while self.is_running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                if not self.is_running:
                    break
                
                # Check all running workers
                workers_to_stop = []
                
                for match_id, worker in list(self.workers.items()):
                    # Check subscriber count
                    subscriber_count = await self.redis_service.get_subscriber_count(match_id)
                    
                    if subscriber_count == 0:
                        logger.info(
                            f"Match {match_id} has no subscribers. "
                            f"Marking worker for cleanup."
                        )
                        workers_to_stop.append(match_id)
                
                # Stop workers with no subscribers
                for match_id in workers_to_stop:
                    await self.stop_worker(match_id)
                
                if workers_to_stop:
                    logger.info(f"Cleaned up {len(workers_to_stop)} workers")
                    
            except asyncio.CancelledError:
                logger.info("Cleanup loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    def get_worker_status(self, match_id: str) -> Optional[dict]:
        """
        Get status of a worker
        
        Args:
            match_id: Match identifier
        
        Returns:
            Worker status dictionary or None if worker doesn't exist
        """
        if match_id in self.workers:
            return self.workers[match_id].get_status()
        return None

    def get_all_workers_status(self) -> Dict[str, dict]:
        """
        Get status of all workers
        
        Returns:
            Dictionary mapping match_id to worker status
        """
        return {
            match_id: worker.get_status()
            for match_id, worker in self.workers.items()
        }

    def get_worker_count(self) -> int:
        """Get number of running workers"""
        return len(self.workers)

    def get_historical_worker_status(self, match_id: str) -> Optional[dict]:
        if match_id in self.historical_workers:
            worker = self.historical_workers[match_id]
            return {
                "match_id": match_id,
                "is_running": worker.is_running,
                "pages_fetched": worker.pages_fetched,
                "commentaries_stored": worker.commentaries_stored,
                "error_count": worker.error_count,
            }
        return None


# Global supervisor instance (will be initialized in main.py)
worker_supervisor: Optional[WorkerSupervisor] = None

