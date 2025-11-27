"""
Database service for commentary system
Provides async database operations
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from database.connection import get_db
from database.commentary_repository import CommentaryRepository
from models.commentary_schemas import CommentaryLine

logger = logging.getLogger(__name__)


class DatabaseService:
    """
    Service for database operations
    Handles connection management and provides repository access
    """
    
    async def store_commentaries(
        self,
        match_id: str,
        commentary_lines: List[CommentaryLine],
        match_status_info: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Store commentary lines in database
        
        Args:
            match_id: External match ID
            commentary_lines: List of commentary lines to store
            match_status_info: Optional match status information
        
        Returns:
            Number of commentaries stored
        """
        try:
            db_gen = get_db()
            session = await db_gen.__anext__()
            try:
                repo = CommentaryRepository(session)
                count = await repo.store_commentaries(match_id, commentary_lines, match_status_info)
                await session.commit()
                return count
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
        except Exception as e:
            logger.error(f"Error storing commentaries for match {match_id}: {e}", exc_info=True)
            return 0

    async def update_match_status(
        self,
        match_id: str,
        status_info: Dict[str, Any],
    ) -> bool:
        """
        Update match metadata/status row
        """
        try:
            db_gen = get_db()
            session = await db_gen.__anext__()
            try:
                repo = CommentaryRepository(session)
                match = await repo.update_match_status(match_id, status_info)
                await session.commit()
                return match is not None
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
        except Exception as e:
            logger.error(f"Error updating match status for {match_id}: {e}", exc_info=True)
            return False

    async def sync_recent_commentaries(
        self,
        match_id: str,
        commentary_lines: List[CommentaryLine],
    ) -> int:
        """
        Update stored commentary text for recent balls if API provides changes.
        """
        if not commentary_lines:
            return 0

        try:
            db_gen = get_db()
            session = await db_gen.__anext__()
            try:
                repo = CommentaryRepository(session)
                updated = await repo.sync_recent_commentaries(match_id, commentary_lines)
                await session.commit()
                return updated
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
        except Exception as e:
            logger.error(f"Error syncing recent commentaries for match {match_id}: {e}", exc_info=True)
            return 0
    
    async def get_match_commentaries(
        self,
        match_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
        order: str = "desc"
    ) -> List[Dict[str, Any]]:
        """
        Get commentaries for a match
        
        Args:
            match_id: External match ID
            limit: Maximum number of results
            offset: Offset for pagination
        
        Returns:
            List of commentary dictionaries
        """
        try:
            db_gen = get_db()
            session = await db_gen.__anext__()
            try:
                repo = CommentaryRepository(session)
                commentaries = await repo.get_match_commentaries(match_id, limit, offset, order)
                
                # Convert to dictionaries
                return [
                    {
                        "id": str(c.id),
                        "match_id": c.match_id,
                        "text": c.text,
                        "event_type": c.event_type,
                        "ball_number": c.ball_number,
                        "over_number": c.over_number,
                        "runs": c.runs,
                        "wickets": c.wickets,
                        "timestamp": c.timestamp.isoformat() if c.timestamp else None,
                        "metadata": c.extra_metadata
                    }
                    for c in commentaries
                ]
            finally:
                await session.close()
        except Exception as e:
            logger.error(f"Error getting commentaries for match {match_id}: {e}", exc_info=True)
            return []

    async def get_commentaries_before_timestamp(
        self,
        match_id: str,
        before_timestamp: datetime,
        innings_id: Optional[int] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Fetch commentaries that occurred before/at the supplied timestamp.
        """
        try:
            db_gen = get_db()
            session = await db_gen.__anext__()
            try:
                repo = CommentaryRepository(session)
                commentaries = await repo.get_commentaries_before_timestamp(
                    match_id=match_id,
                    before_timestamp=before_timestamp,
                    innings_id=innings_id,
                    limit=limit,
                )
                return [
                    {
                        "id": str(c.id),
                        "match_id": c.match_id,
                        "text": c.text,
                        "event_type": c.event_type,
                        "ball_number": c.ball_number,
                        "over_number": c.over_number,
                        "runs": c.runs,
                        "wickets": c.wickets,
                        "timestamp": c.timestamp.isoformat() if c.timestamp else None,
                        "metadata": c.extra_metadata,
                        "innings_id": c.innings_id,
                    }
                    for c in commentaries
                ]
            finally:
                await session.close()
        except Exception as e:
            logger.error(
                f"Error getting commentaries before timestamp for match {match_id}: {e}",
                exc_info=True,
            )
            return []

    async def get_commentary_count(self, match_id: str) -> int:
        """
        Get number of stored commentaries for a match
        """
        try:
            db_gen = get_db()
            session = await db_gen.__anext__()
            try:
                repo = CommentaryRepository(session)
                count = await repo.get_commentary_count(match_id)
                return count
            finally:
                await session.close()
        except Exception as e:
            logger.error(f"Error getting commentary count for match {match_id}: {e}", exc_info=True)
            return 0

    async def get_linked_commentaries(
        self,
        match_id: str,
        timestamp: datetime,
        limit: int = 20,
        innings_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        try:
            db_gen = get_db()
            session = await db_gen.__anext__()
            try:
                repo = CommentaryRepository(session)
                chain = await repo.get_chained_commentaries(
                    match_id=match_id,
                    timestamp=timestamp,
                    limit=limit,
                    innings_id=innings_id,
                )
                return [
                    {
                        "id": str(c.id),
                        "match_id": c.match_id,
                        "text": c.text,
                        "event_type": c.event_type,
                        "ball_number": c.ball_number,
                        "over_number": c.over_number,
                        "runs": c.runs,
                        "wickets": c.wickets,
                        "timestamp": c.timestamp.isoformat() if c.timestamp else None,
                        "metadata": c.extra_metadata,
                        "innings_id": c.innings_id,
                        "prev_commentary_id": str(c.prev_commentary_id) if c.prev_commentary_id else None,
                    }
                    for c in chain
                ]
            finally:
                await session.close()
        except Exception as e:
            logger.error(
                f"Error getting linked commentaries for match {match_id}: {e}",
                exc_info=True,
            )
            return []

    async def get_match_status(self, match_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve match status/metadata from database
        """
        try:
            db_gen = get_db()
            session = await db_gen.__anext__()
            try:
                repo = CommentaryRepository(session)
                match = await repo.get_match(match_id)
                if not match:
                    return None
                extra = match.extra_metadata or {}
                return {
                    "match_id": match.match_id,
                    "team1": match.team1_name,
                    "team2": match.team2_name,
                    "state": match.state,
                    "status": match.status,
                    "match_format": match.match_format,
                    "series_name": match.series_name,
                    "match_desc": match.match_desc,
                    "winning_team": match.winning_team_name,
                    "winning_team_id": match.winning_team_id,
                    "is_complete": match.is_complete,
                    "match_start_timestamp": match.match_start_timestamp.isoformat() if match.match_start_timestamp else None,
                    "match_end_timestamp": match.match_end_timestamp.isoformat() if match.match_end_timestamp else None,
                    "extra_metadata": extra,
                    "miniscore": extra.get("miniscore"),
                }
            finally:
                await session.close()
        except Exception as e:
            logger.error(f"Error retrieving match status for {match_id}: {e}", exc_info=True)
            return None

    async def get_oldest_commentary_info(self, match_id: str) -> Optional[Dict[str, Any]]:
        """
        Get oldest commentary metadata for a match
        """
        try:
            db_gen = get_db()
            session = await db_gen.__anext__()
            try:
                repo = CommentaryRepository(session)
                record = await repo.get_oldest_commentary_info(match_id)
                if not record:
                    return None
                return {
                    "timestamp": record.timestamp,
                    "timestamp_ms": (record.extra_metadata or {}).get("timestamp_ms"),
                    "innings_id": record.innings_id,
                }
            finally:
                await session.close()
        except Exception as e:
            logger.error(f"Error getting oldest commentary info for match {match_id}: {e}", exc_info=True)
            return None


# Global database service instance
db_service = DatabaseService()

