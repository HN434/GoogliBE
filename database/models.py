"""
Database models for matches and commentaries
"""

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Text,
    JSON,
    Boolean,
    Index,
)
from sqlalchemy.orm import relationship, foreign
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

from database.connection import Base


class Match(Base):
    """
    Match information table
    Stores match metadata and status
    """

    __tablename__ = "matches"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    match_id = Column(String(50), unique=True, nullable=False, index=True, comment="External match ID from API")

    # Teams
    team1_name = Column(String(200), nullable=True)
    team2_name = Column(String(200), nullable=True)
    team1_id = Column(Integer, nullable=True)
    team2_id = Column(Integer, nullable=True)

    # Status
    state = Column(String(50), nullable=True, index=True, comment="Match state (LIVE, COMPLETE, etc.)")
    status = Column(Text, nullable=True, comment="Match status text (includes result)")
    is_complete = Column(Boolean, default=False, index=True)

    # Match metadata
    match_format = Column(String(20), nullable=True, comment="T20, ODI, Test, etc.")
    series_name = Column(String(200), nullable=True)
    series_id = Column(Integer, nullable=True)
    match_desc = Column(String(200), nullable=True)
    match_start_timestamp = Column(DateTime, nullable=True)
    match_end_timestamp = Column(DateTime, nullable=True)

    # Winning info
    winning_team_id = Column(Integer, nullable=True)
    winning_team_name = Column(String(200), nullable=True)

    # Audit
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    extra_metadata = Column(JSON, nullable=True)

    # Relationships
    commentaries = relationship(
        "Commentary",
        back_populates="match",
        primaryjoin="Match.match_id==foreign(Commentary.match_id)",
        viewonly=True,
    )

    __table_args__ = (
        Index("idx_match_state_complete", "state", "is_complete"),
        Index("idx_match_created", "created_at"),
    )


class Commentary(Base):
    """
    Commentary lines table
    Stores individual commentary events
    """

    __tablename__ = "commentaries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    match_id = Column(String(50), nullable=False, index=True, comment="External match ID")

    text = Column(Text, nullable=False)
    event_type = Column(String(50), nullable=True, index=True, comment="WICKET, FOUR, SIX, BALL, etc.")

    ball_number = Column(Integer, nullable=True, index=True)
    over_number = Column(Float, nullable=True, index=True)
    innings_id = Column(Integer, nullable=True, index=True)

    runs = Column(Integer, nullable=True)
    wickets = Column(Integer, nullable=True)

    batting_team_name = Column(String(512), nullable=True)

    timestamp = Column(DateTime, nullable=False, index=True, comment="When the event occurred")
    prev_commentary_id = Column(UUID(as_uuid=True), nullable=True, index=True, comment="Linked list pointer to previous commentary")
    extra_metadata = Column(JSON, nullable=True, comment="Additional commentary metadata")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    match = relationship(
        "Match",
        back_populates="commentaries",
        primaryjoin="foreign(Commentary.match_id)==Match.match_id",
        viewonly=True,
    )

    __table_args__ = (
        Index("idx_commentary_match_timestamp", "match_id", "timestamp", unique=True),
        Index("idx_commentary_event_type", "event_type"),
        Index("idx_commentary_over_ball", "over_number", "ball_number"),
        Index("idx_commentary_created", "created_at"),
    )