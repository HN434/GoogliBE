"""
Repository for storing and retrieving commentaries from database
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func

from database.models import Commentary, Match
from models.commentary_schemas import CommentaryLine

logger = logging.getLogger(__name__)


class CommentaryRepository:
    """
    Repository for database operations on commentaries and matches
    """
    
    def __init__(self, session: AsyncSession):
        """
        Initialize repository with database session
        
        Args:
            session: Async database session
        """
        self.session = session
    
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
        if not commentary_lines:
            return 0
        
        # Ensure match record exists / update metadata
        await self._get_or_create_match(match_id, match_status_info)

        # Gather timestamps to check for duplicates
        commentary_timestamps = [
            line.timestamp for line in commentary_lines if line.timestamp is not None
        ]
        
        existing_records = {}
        if commentary_timestamps:
            stmt = select(Commentary).where(
                and_(
                    Commentary.match_id == match_id,
                    Commentary.timestamp.in_(commentary_timestamps)
                )
            )
            result = await self.session.execute(stmt)
            existing = result.scalars().all()
            existing_records = {record.timestamp: record for record in existing}
        existing_timestamps = set(existing_records.keys())
        
        new_commentaries = [
            line for line in commentary_lines
            if line.timestamp not in existing_timestamps
        ]
        
        commentary_records = []
        updated_count = 0
        for line in commentary_lines:
            if line.timestamp in existing_records:
                existing = existing_records[line.timestamp]
                existing.text = line.text
                existing.event_type = line.event_type.value if line.event_type else None
                existing.ball_number = line.ball_number
                existing.over_number = line.over_number
                existing.innings_id = line.metadata.get("inningsid") if line.metadata else None
                existing.runs = line.runs
                existing.wickets = line.wickets
                existing.batting_team_name = line.metadata.get("batTeamName") if line.metadata else None
                existing.extra_metadata = line.metadata
                updated_count += 1
                continue

            commentary = Commentary(
                match_id=match_id,
                text=line.text,
                event_type=line.event_type.value if line.event_type else None,
                ball_number=line.ball_number,
                over_number=line.over_number,
                innings_id=line.metadata.get("inningsid") if line.metadata else None,
                runs=line.runs,
                wickets=line.wickets,
                batting_team_name=line.metadata.get("batTeamName") if line.metadata else None,
                timestamp=line.timestamp,
                extra_metadata=line.metadata,
            )
            commentary_records.append(commentary)
        
        if commentary_records:
            self.session.add_all(commentary_records)
        
        await self.session.flush()
        
        record_map = existing_records.copy()
        for record in commentary_records:
            record_map[record.timestamp] = record
        
        await self._link_commentary_payload(match_id, commentary_lines, record_map)
        
        logger.info(
            f"✅ Stored {len(commentary_records)} new commentaries and updated {updated_count} existing for match {match_id}"
        )
        return len(commentary_records)

    async def _link_commentary_payload(
        self,
        match_id: str,
        commentary_lines: List[CommentaryLine],
        record_map: Dict[datetime, Commentary],
    ):
        if not commentary_lines or not record_map:
            return

        lines_with_timestamps = [
            line for line in commentary_lines if line.timestamp is not None
        ]
        if not lines_with_timestamps:
            return

        sorted_lines = sorted(lines_with_timestamps, key=lambda line: line.timestamp)
        prev_record: Optional[Commentary] = None
        prev_innings: Optional[int] = None

        for line in sorted_lines:
            record = record_map.get(line.timestamp)
            if not record:
                continue

            if (
                prev_record
                and record.match_id == match_id
                and record.innings_id == prev_innings
            ):
                record.prev_commentary_id = prev_record.id
            else:
                record.prev_commentary_id = None

            prev_record = record
            prev_innings = record.innings_id

        await self.session.flush()

    async def sync_recent_commentaries(
        self,
        match_id: str,
        commentary_lines: List[CommentaryLine],
    ) -> int:
        """
        Ensure recent commentary entries match latest API text.
        Only updates existing rows (no inserts) and focuses on last N balls.
        """
        if not commentary_lines:
            return 0

        timestamps = [
            line.timestamp for line in commentary_lines
            if line.timestamp is not None
        ]
        if not timestamps:
            return 0

        stmt = select(Commentary).where(
            and_(
                Commentary.match_id == match_id,
                Commentary.timestamp.in_(timestamps),
            )
        )
        result = await self.session.execute(stmt)
        existing_records = {record.timestamp: record for record in result.scalars().all()}

        updated_count = 0
        for line in commentary_lines:
            if not line.timestamp:
                continue
            record = existing_records.get(line.timestamp)
            if not record:
                continue

            if record.text == line.text:
                continue

            record.text = line.text
            record.event_type = line.event_type.value if line.event_type else record.event_type
            record.ball_number = line.ball_number
            record.over_number = line.over_number
            record.innings_id = line.metadata.get("inningsid") if line.metadata else record.innings_id
            record.runs = line.runs
            record.wickets = line.wickets
            bat_team = line.metadata.get("batTeamName") if line.metadata else None
            if bat_team is not None:
                record.batting_team_name = bat_team
            record.extra_metadata = line.metadata or record.extra_metadata
            updated_count += 1

        if updated_count:
            await self.session.flush()
            logger.info(f"🔁 Synced {updated_count} recent commentary lines for match {match_id}")

        return updated_count
    
    async def get_commentary_by_timestamp(
        self,
        match_id: str,
        timestamp: datetime,
        innings_id: Optional[int] = None,
    ) -> Optional[Commentary]:
        stmt = select(Commentary).where(
            and_(
                Commentary.match_id == match_id,
                Commentary.timestamp == timestamp,
            )
        )
        if innings_id is not None:
            stmt = stmt.where(Commentary.innings_id == innings_id)
        result = await self.session.execute(stmt.limit(1))
        return result.scalar_one_or_none()

    async def get_chained_commentaries(
        self,
        match_id: str,
        timestamp: datetime,
        limit: int = 20,
        innings_id: Optional[int] = None,
    ) -> List[Commentary]:
        start = await self.get_commentary_by_timestamp(match_id, timestamp, innings_id)
        if not start:
            return []
        
        chain = [start]
        current = start
        
        while len(chain) < limit and current.prev_commentary_id:
            prev = await self.session.get(Commentary, current.prev_commentary_id)
            if not prev or prev.match_id != match_id:
                break
            if innings_id is not None and prev.innings_id != innings_id:
                break
            chain.append(prev)
            current = prev
        
        return chain
    
    async def get_match_commentaries(
        self,
        match_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
        order: str = "desc"
    ) -> List[Commentary]:
        """
        Get commentaries for a match ordered by timestamp
        """
        stmt = select(Commentary).where(Commentary.match_id == match_id)
        
        if order == "asc":
            stmt = stmt.order_by(Commentary.timestamp.asc())
        else:
            stmt = stmt.order_by(Commentary.timestamp.desc())
        
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_commentaries_before_timestamp(
        self,
        match_id: str,
        before_timestamp: datetime,
        innings_id: Optional[int] = None,
        limit: int = 20,
    ) -> List[Commentary]:
        """
        Fetch latest commentaries occurring at or before the given timestamp.
        """
        filters = [
            Commentary.match_id == match_id,
            Commentary.timestamp <= before_timestamp,
        ]
        if innings_id is not None:
            filters.append(Commentary.innings_id == innings_id)

        stmt = (
            select(Commentary)
            .where(and_(*filters))
            .order_by(Commentary.timestamp.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_commentary_count(self, match_id: str) -> int:
        """
        Get number of stored commentaries for a match
        """
        stmt = select(func.count()).where(Commentary.match_id == match_id)
        result = await self.session.execute(stmt)
        return result.scalar_one() or 0

    async def get_oldest_commentary_info(self, match_id: str) -> Optional[Commentary]:
        """
        Get oldest commentary for a match
        """
        stmt = (
            select(Commentary)
            .where(Commentary.match_id == match_id)
            .order_by(Commentary.timestamp.asc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def update_match_status(self, match_id: str, status_info: Dict[str, Any]) -> Optional[Match]:
        stmt = select(Match).where(Match.match_id == match_id)
        result = await self.session.execute(stmt)
        match = result.scalar_one_or_none()
        if not match:
            return None
        await self._update_match_from_status(match, status_info)
        await self.session.flush()
        return match

    async def get_match(self, match_id: str) -> Optional[Match]:
        """
        Retrieve match record
        """
        stmt = select(Match).where(Match.match_id == match_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def _get_or_create_match(
        self,
        match_id: str,
        status_info: Optional[Dict[str, Any]] = None,
    ) -> Match:
        match = await self.get_match(match_id)
        if not match:
            match = Match(match_id=match_id, is_complete=False)
            self.session.add(match)
            await self.session.flush()
        if status_info:
            await self._update_match_from_status(match, status_info)
        return match

    async def _update_match_from_status(self, match: Match, status_info: Dict[str, Any]):
        if not status_info:
            return

        match.team1_name = status_info.get("team1") or match.team1_name
        match.team2_name = status_info.get("team2") or match.team2_name
        match.state = status_info.get("state") or match.state
        match.status = status_info.get("status") or match.status
        match.match_format = status_info.get("match_format") or match.match_format
        match.series_name = status_info.get("series_name") or match.series_name
        match.series_id = status_info.get("series_id") or match.series_id
        match.match_desc = status_info.get("match_desc") or match.match_desc
        match.winning_team_id = status_info.get("winning_team_id") or match.winning_team_id
        match.winning_team_name = status_info.get("winning_team") or match.winning_team_name
        start_ts = status_info.get("match_start_timestamp")
        end_ts = status_info.get("match_end_timestamp")
        if start_ts:
            match.match_start_timestamp = self._to_datetime(start_ts) or match.match_start_timestamp
        if end_ts:
            match.match_end_timestamp = self._to_datetime(end_ts) or match.match_end_timestamp

        state = (status_info.get("state") or "").upper()
        if state in {"COMPLETE", "FINISHED"}:
            match.is_complete = True
            if not match.match_end_timestamp:
                match.match_end_timestamp = datetime.utcnow()

        extra = match.extra_metadata or {}
        extra.update(status_info)
        match.extra_metadata = extra

    def _to_datetime(self, value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        if isinstance(value, (int, float)):
            # assume milliseconds if large
            if value > 1e10:
                value = value / 1000
            return datetime.fromtimestamp(value)
        return None

