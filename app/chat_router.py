"""
Googli AI Chat Router - FastAPI endpoints for cricket chatbot
"""
import logging
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, List, Optional
import json

from models.chat_schemas import (
    ChatRequest,
    ChatResponse,
    ChatHealthResponse,
    StreamChunk,
    ImageData,
    ImageFormat
)
from services.chat_service import get_chat_service, BedrockChatError
from services.serper_service import get_serper_service
from services.chat_session_manager import get_session_manager
from utils.image_utils import process_uploaded_image, validate_image_file, ImageProcessingError
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/chat",
    tags=["Googli AI Chat"]
)


@router.get("/health", response_model=ChatHealthResponse)
async def check_chat_health():
    """
    Check health status of Googli AI chat service
    
    Returns:
        ChatHealthResponse with service configuration status
    """
    chat_service = get_chat_service()
    serper_service = get_serper_service()
    
    bedrock_configured = chat_service.is_configured()
    serper_configured = serper_service.is_configured()
    
    if bedrock_configured and serper_configured:
        status_msg = "healthy"
        message = "Googli AI is ready to answer your cricket questions!"
    elif bedrock_configured:
        status_msg = "partial"
        message = "Googli AI is ready, but real-time search is unavailable"
    else:
        status_msg = "unavailable"
        message = "Googli AI chat service is not configured"
    
    return ChatHealthResponse(
        status=status_msg,
        serper_configured=serper_configured,
        bedrock_configured=bedrock_configured,
        model_id=settings.CHAT_MODEL_ID,
        message=message
    )


@router.post("/message", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    """
    Send a message to Googli AI and get a response
    
    Args:
        request: ChatRequest with message and optional conversation history
        
    Returns:
        ChatResponse with assistant's reply
        
    Raises:
        HTTPException: If chat service fails
    """
    if request.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="For streaming responses, use the /api/chat/stream endpoint"
        )
    
    chat_service = get_chat_service()
    
    if not chat_service.is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service is not configured. Please check Bedrock settings."
        )
    
    try:
        logger.info(f"Processing chat message: {request.message[:100]}...")
        
        # Get or create session if session_id provided
        session_manager = get_session_manager()
        session_id = request.session_id
        if session_id:
            session = session_manager.get_or_create_session(session_id)
            logger.debug(f"Using session {session_id} with {len(session.get_history())} previous messages")
        else:
            # Create new session
            session = session_manager.get_or_create_session()
            session_id = session.session_id
            logger.debug(f"Created new session {session_id}")
        
        # Use session history if conversation_history not explicitly provided
        conversation_history = request.conversation_history
        if conversation_history is None and session_id:
            conversation_history = session_manager.get_history(session_id)
        
        response = await chat_service.chat(
            message=request.message,
            images=request.images,
            conversation_history=conversation_history,
            use_search=request.use_search,
            session_id=session_id,
            tone=request.tone,
            language=request.language
        )
        
        logger.info(
            f"Chat response generated successfully. "
            f"Session: {session_id}, Tools used: {len(response.tools_used)}"
        )
        return response
        
    except BedrockChatError as e:
        logger.error(f"Bedrock chat error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat generation failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )


async def _stream_response(request: ChatRequest) -> AsyncGenerator[str, None]:
    """
    Internal generator for streaming chat responses
    
    Args:
        request: ChatRequest
        
    Yields:
        Server-Sent Events formatted strings
    """
    chat_service = get_chat_service()
    
    try:
        # Get or create session if session_id provided
        session_manager = get_session_manager()
        session_id = request.session_id
        if session_id:
            session = session_manager.get_or_create_session(session_id)
        else:
            session = session_manager.get_or_create_session()
            session_id = session.session_id
        
        # Use session history if conversation_history not explicitly provided
        conversation_history = request.conversation_history
        if conversation_history is None and session_id:
            conversation_history = session_manager.get_history(session_id)
        
        async for chunk in chat_service.chat_stream(
            message=request.message,
            images=request.images,
            conversation_history=conversation_history,
            use_search=request.use_search,
            session_id=session_id,
            tone=request.tone,
            language=request.language
        ):
            # Format as Server-Sent Event
            chunk_json = chunk.model_dump_json()
            yield f"data: {chunk_json}\n\n"
            
            if chunk.is_final:
                break
        
        # Send final event
        yield "data: [DONE]\n\n"
        
    except BedrockChatError as e:
        logger.error(f"Streaming error: {e}")
        error_chunk = StreamChunk(
            content=f"Error: {str(e)}",
            is_final=True
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
    except Exception as e:
        logger.error(f"Unexpected streaming error: {e}")
        error_chunk = StreamChunk(
            content=f"Unexpected error: {str(e)}",
            is_final=True
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"


@router.post("/stream")
async def stream_message(request: ChatRequest):
    """
    Send a message to Googli AI and get a streaming response
    
    Args:
        request: ChatRequest with message and optional conversation history
        
    Returns:
        StreamingResponse with Server-Sent Events
        
    Raises:
        HTTPException: If chat service is not configured
    """
    chat_service = get_chat_service()
    
    if not chat_service.is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service is not configured. Please check Bedrock settings."
        )
    
    logger.info(f"Starting streaming response for message: {request.message[:100]}...")
    
    return StreamingResponse(
        _stream_response(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@router.get("/info")
async def get_chat_info():
    """
    Get information about Googli AI chatbot
    
    Returns:
        Dictionary with chatbot information
    """
    return {
        "name": "Googli AI",
        "description": "Your friendly cricket expert chatbot",
        "specialization": "Cricket - matches, players, teams, rules, history, and statistics",
        "capabilities": [
            "Answer questions about cricket",
            "Provide real-time match updates and scores",
            "Share player and team statistics",
            "Explain cricket rules and terminology",
            "Discuss cricket history and memorable moments",
            "Search for current cricket news and information"
        ],
        "features": [
            "Powered by AWS Bedrock (Claude 3.5 Sonnet)",
            "Real-time information via Serper API",
            "Streaming responses for better user experience",
            "Conversation history support"
        ],
        "model": settings.CHAT_MODEL_ID,
        "version": "1.0.0"
    }


@router.post("/message-with-image", response_model=ChatResponse)
async def send_message_with_image(
    message: str = Form(..., description="User's message"),
    images: List[UploadFile] = File(..., description="Cricket-related images to analyze"),
    session_id: Optional[str] = Form(None, description="Session ID for maintaining conversation history"),
    use_search: Optional[bool] = Form(None, description="Force search on/off"),
    stream: bool = Form(False, description="Whether to stream response"),
    tone: Optional[str] = Form("professional", description="Response tone: professional, casual, enthusiastic, analytical, friendly, or formal")
):
    """
    Send a message with images to Googli AI
    
    Args:
        message: Text message
        images: One or more cricket-related images (JPEG, PNG, GIF, WEBP)
        session_id: Optional session ID for maintaining conversation history
        use_search: Optional flag to force search
        stream: Whether to stream response
        
    Returns:
        ChatResponse with assistant's analysis
        
    Raises:
        HTTPException: If processing fails
    """
    if stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Streaming with file upload is not yet supported. Use /api/chat/stream with base64 images."
        )
    
    chat_service = get_chat_service()
    
    if not chat_service.is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service is not configured. Please check Bedrock settings."
        )
    
    # Process uploaded images
    processed_images = []
    
    try:
        for image_file in images:
            # Read file
            image_data = await image_file.read()
            
            # Validate
            validate_image_file(len(image_data), image_file.content_type)
            
            # Process
            logger.info(f"Processing uploaded image: {image_file.filename} ({len(image_data)} bytes)")
            base64_data, image_format = process_uploaded_image(image_data, image_file.content_type)
            
            processed_images.append(ImageData(
                format=ImageFormat(image_format),
                source=base64_data
            ))
        
        logger.info(f"Processing message with {len(processed_images)} images")
        
        # Get or create session if session_id provided
        session_manager = get_session_manager()
        if session_id:
            session = session_manager.get_or_create_session(session_id)
            logger.debug(f"Using session {session_id} with {len(session.get_history())} previous messages")
        else:
            # Create new session
            session = session_manager.get_or_create_session()
            session_id = session.session_id
            logger.debug(f"Created new session {session_id}")
        
        # Use session history if available
        conversation_history = session_manager.get_history(session_id)
        
        # Generate response
        response = await chat_service.chat(
            message=message,
            images=processed_images,
            conversation_history=conversation_history,
            use_search=use_search,
            session_id=session_id,
            tone=tone
        )
        
        logger.info(f"Chat response generated with images. Tools used: {len(response.tools_used)}")
        return response
        
    except ImageProcessingError as e:
        logger.error(f"Image processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image processing failed: {str(e)}"
        )
    except BedrockChatError as e:
        logger.error(f"Bedrock chat error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat generation failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in image chat endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@router.post("/reset")
async def reset_conversation(session_id: Optional[str] = None):
    """
    Reset conversation for a session
    
    Args:
        session_id: Session ID to reset. If not provided, returns instructions.
    
    Returns:
        Success message
    """
    if session_id:
        session_manager = get_session_manager()
        session_manager.clear_session(session_id)
        return {
            "message": f"Conversation history cleared for session {session_id}",
            "session_id": session_id
        }
    else:
        return {
            "message": "To reset a conversation, provide a session_id",
            "tip": "Send a POST request with session_id parameter to clear that session's history"
        }


@router.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """
    Get information about a chat session
    
    Args:
        session_id: Session ID
    
    Returns:
        Session information including message count
    """
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)
    
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    
    history = session.get_history()
    return {
        "session_id": session_id,
        "created_at": session.created_at.isoformat(),
        "last_accessed": session.last_accessed.isoformat(),
        "message_count": len(history),
        "q_a_pairs": len(history) // 2,  # Each Q&A is 2 messages
    }

