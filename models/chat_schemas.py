"""
Pydantic schemas for Googli AI Chat Module
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
import base64


class MessageRole(str, Enum):
    """Message role types"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ImageFormat(str, Enum):
    """Supported image formats"""
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"


class ImageData(BaseModel):
    """Image data for multimodal messages"""
    format: ImageFormat = Field(..., description="Image format")
    source: str = Field(..., description="Base64 encoded image data or URL")
    
    @validator('source')
    def validate_source(cls, v):
        """Validate that source is either base64 or URL"""
        if v.startswith('http://') or v.startswith('https://'):
            return v
        # Try to validate base64
        try:
            base64.b64decode(v, validate=True)
            return v
        except Exception:
            raise ValueError("Image source must be a valid URL or base64 encoded string")
    
    class Config:
        json_schema_extra = {
            "example": {
                "format": "jpeg",
                "source": "base64_encoded_image_data_here"
            }
        }


class ChatMessage(BaseModel):
    """Individual chat message (can include text and/or images)"""
    role: MessageRole
    content: str
    images: Optional[List[ImageData]] = Field(default=None, description="Optional images in the message")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    """Request for chat completion (supports text and images)"""
    message: str = Field(..., description="User's message to Googli AI", min_length=1, max_length=5000)
    images: Optional[List[ImageData]] = Field(
        default=None,
        description="Optional images to analyze (e.g., cricket screenshots, scorecards)"
    )
    conversation_history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Previous messages in the conversation for context (auto-managed if session_id provided)"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for maintaining conversation history. If not provided, a new session is created."
    )
    stream: bool = Field(default=False, description="Whether to stream the response")
    use_search: Optional[bool] = Field(
        default=None,
        description="Force enable/disable real-time search. If None, AI decides automatically"
    )
    tone: Optional[str] = Field(
        default="professional",
        description="Response tone: professional, casual, enthusiastic, analytical, friendly, or formal. Defaults to professional."
    )
    language: Optional[str] = Field(
        default="UK - English",
        description="Response language: any language works. Defaults to UK - English."
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Who won the 2023 Cricket World Cup?",
                "images": None,
                "conversation_history": [],
                "stream": False,
                "use_search": True,
                "tone": "professional",
                "language": "UK - English"
            }
        }


class ToolUse(BaseModel):
    """Tool usage information"""
    tool_name: str
    input: Dict[str, Any]
    output: Optional[str] = None


class ChatResponse(BaseModel):
    """Response from chat completion"""
    message: str = Field(..., description="Assistant's response")
    role: MessageRole = Field(default=MessageRole.ASSISTANT)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tools_used: List[ToolUse] = Field(default=[], description="Tools called during response generation")
    model_id: str = Field(..., description="Model used for generation")
    finish_reason: Optional[str] = Field(None, description="Reason for completion")
    session_id: Optional[str] = Field(None, description="Session ID for this conversation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "India won the 2023 Cricket World Cup, defeating Australia in the final...",
                "role": "assistant",
                "timestamp": "2024-12-05T10:30:00Z",
                "tools_used": [
                    {
                        "tool_name": "search_cricket_info",
                        "input": {"query": "2023 Cricket World Cup winner"},
                        "output": "India won..."
                    }
                ],
                "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
                "finish_reason": "end_turn"
            }
        }


class StreamChunk(BaseModel):
    """Streaming response chunk"""
    content: str = Field(..., description="Chunk of response text")
    is_final: bool = Field(default=False, description="Whether this is the final chunk")
    tool_use: Optional[ToolUse] = Field(None, description="Tool usage if applicable")


class ChatHealthResponse(BaseModel):
    """Health check response for chat service"""
    status: str
    serper_configured: bool
    bedrock_configured: bool
    model_id: str
    message: str


class SerperSearchResult(BaseModel):
    """Individual search result from Serper API"""
    title: str
    link: str
    snippet: str
    position: Optional[int] = None


class SerperResponse(BaseModel):
    """Full response from Serper API"""
    searchParameters: Dict[str, Any]
    organic: List[SerperSearchResult]
    answerBox: Optional[Dict[str, Any]] = None
    knowledgeGraph: Optional[Dict[str, Any]] = None

