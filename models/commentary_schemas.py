"""
Pydantic schemas for Cricket Commentary System
Defines data models for commentary messages, match information, and WebSocket payloads
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class CommentaryEventType(str, Enum):
    """Types of commentary events"""
    BALL = "ball"
    WICKET = "wicket"
    FOUR = "four"
    SIX = "six"
    OVER = "over"
    MATCH_START = "match_start"
    MATCH_END = "match_end"
    INNINGS_BREAK = "innings_break"
    OTHER = "other"


class CommentaryLine(BaseModel):
    """Single commentary line/event"""
    id: str = Field(..., description="Unique identifier for this commentary event")
    timestamp: datetime = Field(..., description="When this event occurred")
    text: str = Field(..., description="Commentary text")
    event_type: CommentaryEventType = Field(default=CommentaryEventType.OTHER)
    ball_number: Optional[int] = Field(None, description="Ball number in the over")
    over_number: Optional[float] = Field(None, description="Over number (e.g., 12.3)")
    runs: Optional[int] = Field(None, description="Runs scored on this ball")
    wickets: Optional[int] = Field(None, description="Wickets fallen")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional event metadata")


class CommentaryUpdate(BaseModel):
    """Update message containing new commentary lines"""
    match_id: str = Field(..., description="Match identifier")
    timestamp: datetime = Field(default_factory=datetime.now)
    lines: List[CommentaryLine] = Field(default_factory=list, description="New commentary lines since last update")
    match_status: Optional[str] = Field(None, description="Current match status")
    score: Optional[Dict[str, Any]] = Field(None, description="Current match score")
    language: str = Field(default="en", description="Language of the commentary payload")


class MatchInfo(BaseModel):
    """Match information"""
    match_id: str = Field(..., description="Match identifier")
    team1: Optional[str] = Field(None, description="Team 1 name")
    team2: Optional[str] = Field(None, description="Team 2 name")
    status: Optional[str] = Field(None, description="Match status (live, finished, etc.)")
    venue: Optional[str] = Field(None, description="Match venue")
    date: Optional[datetime] = Field(None, description="Match date/time")
    score: Optional[Dict[str, Any]] = Field(None, description="Current score")


class WebSocketMessage(BaseModel):
    """WebSocket message format"""
    type: str = Field(..., description="Message type: commentary, error, status, etc.")
    match_id: str = Field(..., description="Match identifier")
    language: str = Field(default="en", description="Language of the payload")
    data: Dict[str, Any] = Field(..., description="Message payload")
    timestamp: datetime = Field(default_factory=datetime.now)


class WorkerStatus(BaseModel):
    """Worker status information"""
    match_id: str
    is_running: bool
    last_fetch: Optional[datetime] = None
    subscriber_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None

