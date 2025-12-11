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
        translation_service=None,
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
        self.translation_service = translation_service
        
        # Registry of active workers: match_id -> worker
        # Only one active worker per match at a time
        self.active_workers: Dict[str, MatchWorker] = {}
        
        # Track workers that are closing: match_id -> list of workers
        # Multiple closing workers allowed per match
        self.closing_workers: Dict[str, list[MatchWorker]] = {}
        
        self.historical_workers: Dict[str, HistoricalCommentaryWorker] = {}
        
        # Locks for worker lifecycle operations to prevent concurrent start/stop
        self.worker_locks: Dict[str, asyncio.Lock] = {}
        
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
        
        # Stop all workers (both active and closing)
        stop_tasks = [worker.stop() for worker in self.active_workers.values()]
        for closing_list in self.closing_workers.values():
            stop_tasks.extend([worker.stop() for worker in closing_list])
        stop_tasks += [worker.stop() for worker in self.historical_workers.values()]
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        self.active_workers.clear()
        self.closing_workers.clear()
        self.historical_workers.clear()
        self.worker_locks.clear()
        logger.info("🛑 Worker supervisor stopped")

    async def ensure_worker(self, match_id: str) -> MatchWorker:
        """
        Ensure a worker is running for a match
        Creates worker if it doesn't exist, otherwise returns existing active worker
        
        If current active worker is stopping, it moves it to closing list and creates a new one.
        This prevents race conditions when users reconnect during worker cleanup.
        
        Args:
            match_id: Match identifier
        
        Returns:
            Active MatchWorker instance
        """
        # Get or create lock for this match
        if match_id not in self.worker_locks:
            self.worker_locks[match_id] = asyncio.Lock()
        
        async with self.worker_locks[match_id]:
            # Check if there's an active worker
            if match_id in self.active_workers:
                worker = self.active_workers[match_id]
                
                # Check if worker is in grace period (shutting down soon)
                if worker.in_grace_period:
                    closing_count = len(self.closing_workers.get(match_id, []))
                    logger.info(
                        f"⏳ Active worker for match {match_id} is in grace period (shutting down), "
                        f"moving to closing list ({closing_count} already closing) and creating new worker"
                    )
                    self._move_to_closing(match_id, worker)
                    # Continue to create new worker below
                
                # If worker is running and healthy (not in grace period), return it
                elif worker.is_running:
                    logger.debug(f"✅ Returning existing active worker for match {match_id}")
                    return worker
                else:
                    # Worker is not running anymore - move it to closing list
                    closing_count = len(self.closing_workers.get(match_id, []))
                    logger.info(
                        f"🔄 Active worker for match {match_id} is not running, "
                        f"moving to closing list ({closing_count} already closing) and creating new worker"
                    )
                    self._move_to_closing(match_id, worker)
                    # Continue to create new worker below
            
            # Check if worker should be running (has subscribers)
            subscriber_count = await self.redis_service.get_subscriber_count(match_id)
            if subscriber_count == 0:
                logger.debug(f"No subscribers for match {match_id}, not starting worker")
                return None
            
            # Create and start new active worker
            logger.info(f"Creating new active worker for match {match_id} (subscribers: {subscriber_count})")
            worker = MatchWorker(
                match_id=match_id,
                commentary_service=self.commentary_service,
                redis_service=self.redis_service,
                ws_manager=self.ws_manager,
                translation_service=self.translation_service,
            )
            
            # Set as active worker
            self.active_workers[match_id] = worker
            await worker.start()
            
            logger.info(f"✅ Created and started new active worker for match {match_id}")
            return worker
    
    def _move_to_closing(self, match_id: str, worker: MatchWorker):
        """
        Move a worker to the closing list and remove from active
        
        Args:
            match_id: Match identifier
            worker: Worker to move to closing
        """
        # Remove from active
        if match_id in self.active_workers and self.active_workers[match_id] == worker:
            del self.active_workers[match_id]
        
        # Add to closing list
        if match_id not in self.closing_workers:
            self.closing_workers[match_id] = []
        self.closing_workers[match_id].append(worker)
        
        closing_count = len(self.closing_workers[match_id])
        
        # Schedule cleanup of this worker from closing list after it finishes
        asyncio.create_task(self._cleanup_closed_worker(match_id, worker))
        
        logger.info(
            f"📦 Moved worker for match {match_id} to closing list "
            f"(total closing for this match: {closing_count})"
        )
    
    async def _cleanup_closed_worker(self, match_id: str, worker: MatchWorker):
        """
        Wait for a worker to finish closing, then remove it from closing list
        
        Args:
            match_id: Match identifier
            worker: Worker that is closing
        """
        try:
            # Wait for worker to finish (check every second for up to 30 seconds)
            max_wait = 30
            waited = 0
            for i in range(max_wait):
                if not worker.is_running:
                    waited = i
                    break
                await asyncio.sleep(1)
            else:
                # Worker still running after max wait
                logger.warning(
                    f"⚠️ Worker for match {match_id} still running after {max_wait}s, "
                    f"removing from closing list anyway"
                )
            
            # Remove from closing list
            if match_id in self.closing_workers:
                if worker in self.closing_workers[match_id]:
                    self.closing_workers[match_id].remove(worker)
                    remaining = len(self.closing_workers[match_id])
                    logger.info(
                        f"🧹 Cleaned up closed worker for match {match_id} after {waited}s "
                        f"({remaining} still closing for this match)"
                    )
                
                # Clean up empty list
                if not self.closing_workers[match_id]:
                    del self.closing_workers[match_id]
                    logger.debug(f"Removed empty closing list for match {match_id}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up closed worker for match {match_id}: {e}", exc_info=True)

    async def stop_worker(self, match_id: str):
        """
        Stop the active worker for a match and move it to closing list
        
        The worker will continue running in the background until it finishes cleanup.
        New connections can create a new active worker immediately.
        
        Args:
            match_id: Match identifier
        """
        # Get or create lock for this match
        if match_id not in self.worker_locks:
            self.worker_locks[match_id] = asyncio.Lock()
        
        async with self.worker_locks[match_id]:
            if match_id not in self.active_workers:
                logger.debug(f"No active worker for match {match_id} to stop")
                return
            
            worker = self.active_workers[match_id]
            logger.info(f"Stopping active worker for match {match_id}, moving to closing list...")
            
            # Move to closing list (will be cleaned up automatically)
            self._move_to_closing(match_id, worker)
            
            # Trigger worker stop asynchronously (don't wait for it)
            asyncio.create_task(worker.stop())
            
            logger.info(f"✅ Moved worker for match {match_id} to closing list")

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
        Background task that periodically cleans up active workers with no subscribers
        Closing workers clean themselves up independently
        """
        logger.info("Cleanup loop started")
        
        while self.is_running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                if not self.is_running:
                    break
                
                # Check all active workers
                workers_to_stop = []
                
                for match_id, worker in list(self.active_workers.items()):
                    # Check subscriber count
                    subscriber_count = await self.redis_service.get_subscriber_count(match_id)
                    
                    if subscriber_count == 0:
                        logger.info(
                            f"Match {match_id} has no subscribers. "
                            f"Marking active worker for cleanup."
                        )
                        workers_to_stop.append(match_id)
                
                # Stop workers with no subscribers
                for match_id in workers_to_stop:
                    # Double-check subscriber count before stopping (in case new subscribers joined)
                    subscriber_count = await self.redis_service.get_subscriber_count(match_id)
                    if subscriber_count == 0:
                        await self.stop_worker(match_id)
                    else:
                        logger.info(f"Skipping cleanup for match {match_id} - new subscribers detected ({subscriber_count})")
                
                if workers_to_stop:
                    stopped_count = len([m for m in workers_to_stop if m not in self.active_workers])
                    logger.info(f"Cleanup cycle completed, moved {stopped_count} workers to closing list")
                
                # Log status for debugging
                active_count = len(self.active_workers)
                closing_count = sum(len(workers) for workers in self.closing_workers.values())
                if active_count > 0 or closing_count > 0:
                    logger.debug(f"Worker status: {active_count} active, {closing_count} closing")
                    
            except asyncio.CancelledError:
                logger.info("Cleanup loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    def get_worker_status(self, match_id: str) -> Optional[dict]:
        """
        Get status of the active worker for a match
        
        Args:
            match_id: Match identifier
        
        Returns:
            Worker status dictionary or None if no active worker exists
        """
        if match_id in self.active_workers:
            status = self.active_workers[match_id].get_status()
            status['state'] = 'active'
            return status
        return None

    def get_all_workers_status(self) -> Dict[str, dict]:
        """
        Get status of all workers (both active and closing)
        
        Returns:
            Dictionary mapping match_id to worker status
        """
        status = {}
        
        # Add active workers
        for match_id, worker in self.active_workers.items():
            worker_status = worker.get_status()
            worker_status['state'] = 'active'
            status[match_id] = worker_status
        
        # Add closing workers (with index to distinguish multiple)
        for match_id, workers in self.closing_workers.items():
            for idx, worker in enumerate(workers):
                worker_status = worker.get_status()
                worker_status['state'] = 'closing'
                key = f"{match_id}_closing_{idx}" if idx > 0 else f"{match_id}_closing"
                status[key] = worker_status
        
        return status

    def get_worker_count(self) -> int:
        """Get number of active workers"""
        return len(self.active_workers)
    
    def get_total_worker_count(self) -> int:
        """Get total number of workers (active + closing)"""
        active = len(self.active_workers)
        closing = sum(len(workers) for workers in self.closing_workers.values())
        return active + closing

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

