"""
Chat Session Manager
Manages conversation history per session with automatic cleanup.
"""

import logging
import uuid
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque

from models.chat_schemas import ChatMessage, MessageRole

logger = logging.getLogger(__name__)


class ChatSession:
    """Represents a chat session with conversation history."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        # Store last 5 Q&A pairs (10 messages total: 5 user + 5 assistant)
        self.history: deque = deque(maxlen=10)  # Max 10 messages (5 pairs)
    
    def add_message(self, message: ChatMessage):
        """Add a message to session history."""
        self.last_accessed = datetime.now()
        self.history.append(message)
    
    def get_history(self) -> List[ChatMessage]:
        """Get conversation history as list."""
        return list(self.history)
    
    def clear_history(self):
        """Clear conversation history."""
        self.history.clear()
        logger.info(f"Cleared history for session {self.session_id}")


class ChatSessionManager:
    """Manages chat sessions with automatic cleanup."""
    
    def __init__(self, session_timeout_hours: int = 24):
        """
        Initialize session manager.
        
        Args:
            session_timeout_hours: Hours before inactive session is cleaned up
        """
        self.sessions: Dict[str, ChatSession] = {}
        self.session_timeout = timedelta(hours=session_timeout_hours)
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> ChatSession:
        """
        Get existing session or create a new one.
        
        Args:
            session_id: Optional session ID. If None, creates new session.
        
        Returns:
            ChatSession instance
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
            logger.info(f"Created new chat session: {session_id}")
        
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatSession(session_id)
        
        session = self.sessions[session_id]
        session.last_accessed = datetime.now()
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get session by ID, or None if not found."""
        return self.sessions.get(session_id)
    
    def get_history(self, session_id: Optional[str] = None) -> List[ChatMessage]:
        """
        Get conversation history for a session (last 5 Q&A pairs).
        
        Args:
            session_id: Session ID. If None, returns empty history.
        
        Returns:
            List of ChatMessage objects (last 10 messages max)
        """
        if session_id is None:
            return []
        
        session = self.get_session(session_id)
        if session is None:
            return []
        
        return session.get_history()
    
    def add_to_history(
        self,
        session_id: Optional[str],
        user_message: str,
        assistant_message: str
    ):
        """
        Add a Q&A pair to session history.
        
        Args:
            session_id: Session ID
            user_message: User's message
            assistant_message: Assistant's response
        """
        if session_id is None:
            return
        
        session = self.get_or_create_session(session_id)
        
        # Add user message
        session.add_message(ChatMessage(
            role=MessageRole.USER,
            content=user_message
        ))
        
        # Add assistant message
        session.add_message(ChatMessage(
            role=MessageRole.ASSISTANT,
            content=assistant_message
        ))
        
        logger.debug(
            f"Added Q&A to session {session_id}. "
            f"History now has {len(session.history)} messages"
        )
    
    def clear_session(self, session_id: str):
        """Clear history for a specific session."""
        session = self.get_session(session_id)
        if session:
            session.clear_history()
    
    def cleanup_expired_sessions(self):
        """Remove sessions that haven't been accessed recently."""
        now = datetime.now()
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if now - session.last_accessed > self.session_timeout
        ]
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
            logger.info(f"Cleaned up expired session: {session_id}")
        
        return len(expired_sessions)


# Global session manager instance
_session_manager: Optional[ChatSessionManager] = None


def get_session_manager() -> ChatSessionManager:
    """Get or create ChatSessionManager singleton."""
    global _session_manager
    if _session_manager is None:
        _session_manager = ChatSessionManager()
    return _session_manager

