"""
Googli AI Chat Service using AWS Bedrock with Tool Calling and Streaming Support
"""
import json
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
from config import settings
from models.chat_schemas import (
    ChatMessage, ChatResponse, MessageRole, ToolUse, StreamChunk, ImageData
)
from services.serper_service import get_serper_service, SerperAPIError
from services.chat_session_manager import get_session_manager
import re

logger = logging.getLogger(__name__)


class BedrockChatError(Exception):
    """Custom exception for Bedrock chat errors"""
    pass


class GoogliAIChatService:
    """Googli AI - Cricket Expert Chatbot Service using AWS Bedrock"""
    
    # Tool definitions for Claude models
    TOOLS = [
        {
            "toolSpec": {
                "name": "search_cricket_info",
                "description": (
                    "Search for real-time cricket information including live scores, "
                    "recent match results, player statistics, team rankings, news, "
                    "and upcoming fixtures. Use this when you need current information "
                    "that may have changed recently or for live match updates."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "The cricket-related search query. Be specific and include "
                                    "relevant details like team names, player names, tournament names, "
                                    "or date ranges."
                                )
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        }
    ]
    
    def __init__(self):
        """Initialize Googli AI Chat Service"""
        self.region = settings.BEDROCK_REGION or settings.AWS_REGION
        # Use inference profile ARN if provided, otherwise use model ID
        # Inference profiles are required for some models with on-demand throughput
        if settings.CHAT_INFERENCE_PROFILE_ARN:
            self.model_id = settings.CHAT_INFERENCE_PROFILE_ARN
            logger.info(f"Using inference profile ARN: {self.model_id}")
        else:
            self.model_id = settings.CHAT_MODEL_ID
            logger.info(f"Using model ID: {self.model_id}")
            logger.warning(
                "If you encounter 'on-demand throughput isn't supported' errors, "
                "set CHAT_INFERENCE_PROFILE_ARN environment variable. "
                "Format: arn:aws:bedrock:{region}::inference-profile/{model-id}"
            )
        self.temperature = settings.CHAT_TEMPERATURE
        self.top_p = settings.CHAT_TOP_P
        self.max_tokens = settings.CHAT_MAX_TOKENS
        self.system_prompt = settings.CHAT_SYSTEM_PROMPT
        
        # Initialize Bedrock client
        try:
            self.bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.region
            )
            logger.info(f"Bedrock client initialized for region: {self.region}")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            self.bedrock_client = None
        
        # Initialize Serper service
        self.serper_service = get_serper_service()
    
    def is_configured(self) -> bool:
        """Check if Bedrock is properly configured"""
        return self.bedrock_client is not None and bool(self.region)
    
    def _get_system_prompt_with_datetime(self, tone: str = "professional", language: str = "UK - English") -> str:
        """
        Get system prompt with current date/time and tone injected
        
        Args:
            tone: Response tone (professional, casual, enthusiastic, analytical, friendly, formal)
        
        Returns:
            System prompt string with current date/time and tone formatted
        """
        try:
            current_datetime = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            # Normalize tone to lowercase and validate
            if not tone or not isinstance(tone, str):
                tone = "professional"
            tone = tone.lower().strip()
            
            formatted_prompt = self.system_prompt.format(current_datetime=current_datetime, tone=tone, language=language)
            return formatted_prompt
        except KeyError as e:
            logger.error(f"Error formatting system prompt: missing placeholder {e}")
            # Fallback: try with just current_datetime if tone fails
            try:
                return self.system_prompt.format(current_datetime=current_datetime, tone="professional")
            except Exception as fallback_error:
                logger.error(f"Fallback formatting also failed: {fallback_error}")
                # Last resort: return prompt with placeholders replaced manually
                return self.system_prompt.replace("{current_datetime}", current_datetime).replace("{tone}", "professional")
        except Exception as e:
            logger.error(f"Unexpected error formatting system prompt: {e}")
            raise BedrockChatError(f"Failed to format system prompt: {str(e)}")
    
    def _format_messages_for_bedrock(
        self,
        current_message: str,
        current_images: Optional[List[ImageData]] = None,
        conversation_history: Optional[List[ChatMessage]] = None
    ) -> List[Dict[str, Any]]:
        """
        Format messages for Bedrock API (supports multimodal content)
        
        Args:
            current_message: Current user message
            current_images: Optional images in current message
            conversation_history: Previous conversation messages
            
        Returns:
            List of formatted messages with text and/or images
        """
        messages = []
        
        # Add conversation history (exclude system messages)
        if conversation_history:
            for msg in conversation_history:
                if msg.role != MessageRole.SYSTEM:
                    content = [{"text": msg.content}]
                    
                    # Add images if present in history
                    if msg.images:
                        for img in msg.images:
                            content.append({
                                "image": {
                                    "format": img.format.value,
                                    "source": {
                                        "bytes": self._prepare_image_data(img.source)
                                    }
                                }
                            })
                    
                    messages.append({
                        "role": msg.role.value,
                        "content": content
                    })
        
        # Add current message with images
        content = [{"text": current_message}]
        
        if current_images:
            for img in current_images:
                content.append({
                    "image": {
                        "format": img.format.value,
                        "source": {
                            "bytes": self._prepare_image_data(img.source)
                        }
                    }
                })
        
        messages.append({
            "role": "user",
            "content": content
        })
        
        return messages
    
    def _prepare_image_data(self, source: str) -> bytes:
        """
        Prepare image data for Bedrock API
        
        Args:
            source: Base64 string or URL
            
        Returns:
            Image bytes
        """
        import base64
        import httpx
        
        # If it's a URL, download it
        if source.startswith('http://') or source.startswith('https://'):
            try:
                response = httpx.get(source, timeout=10.0)
                response.raise_for_status()
                return response.content
            except Exception as e:
                logger.error(f"Failed to download image from URL: {e}")
                raise BedrockChatError(f"Failed to download image: {str(e)}")
        
        # Otherwise, decode base64
        try:
            return base64.b64decode(source)
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            raise BedrockChatError(f"Invalid image data: {str(e)}")
    
    async def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """
        Execute a tool and return the result
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool
            
        Returns:
            Tool execution result as string
        """
        logger.info(f"Executing tool: {tool_name} with input: {tool_input}")
        
        if tool_name == "search_cricket_info":
            try:
                query = tool_input.get("query", "")
                result = await self.serper_service.search_cricket_info(query)
                logger.info(f"Tool execution successful for {tool_name}")
                return result
            except SerperAPIError as e:
                error_msg = f"Search failed: {str(e)}"
                logger.error(error_msg)
                return error_msg
        else:
            return f"Unknown tool: {tool_name}"
    
    async def chat(
        self,
        message: str,
        images: Optional[List[ImageData]] = None,
        conversation_history: Optional[List[ChatMessage]] = None,
        use_search: Optional[bool] = None,
        session_id: Optional[str] = None,
        tone: Optional[str] = "professional",
        language: Optional[str] = "UK - English"
    ) -> ChatResponse:
        """
        Generate a chat response using Bedrock with tool calling
        
        Args:
            message: User's message
            images: Optional images to analyze
            conversation_history: Previous messages in the conversation (overrides session history if provided)
            use_search: Force enable/disable search tool
            session_id: Session ID for maintaining conversation history
            
        Returns:
            ChatResponse with assistant's reply and tool usage info
            
        Raises:
            BedrockChatError: If chat generation fails
        """
        if not self.is_configured():
            raise BedrockChatError("Bedrock client not configured")
        
        # Get conversation history from session if session_id provided and history not explicitly provided
        session_manager = get_session_manager()
        if session_id and conversation_history is None:
            conversation_history = session_manager.get_history(session_id)
            logger.debug(f"Loaded {len(conversation_history)} messages from session {session_id}")
        
        if conversation_history is None:
            conversation_history = []
        
        messages = self._format_messages_for_bedrock(message, images, conversation_history)
        tools_used = []
        
        # Configure tool use based on use_search parameter
        tool_config = None
        if use_search is False:
            # Disable tools
            inference_config = {
                "temperature": self.temperature,
                "topP": self.top_p,
                "maxTokens": self.max_tokens
            }
        else:
            # Enable tools (default behavior or explicitly requested)
            tool_config = {"tools": self.TOOLS}
            inference_config = {
                "temperature": self.temperature,
                "topP": self.top_p,
                "maxTokens": self.max_tokens
            }
        
        try:
            # Get formatted system prompt
            system_prompt_text = self._get_system_prompt_with_datetime(tone=tone, language=language)
            logger.debug(f"System prompt length: {len(system_prompt_text)} characters")
            
            # Initial API call
            request_params = {
                "modelId": self.model_id,
                "messages": messages,
                "system": [{"text": system_prompt_text}],
                "inferenceConfig": inference_config
            }
            
            if tool_config:
                request_params["toolConfig"] = tool_config
            
            logger.debug(f"Making Bedrock API call with model: {self.model_id}, tone: {tone}")
            response = self.bedrock_client.converse(**request_params)
            
            # Handle tool use loop
            max_iterations = 5
            iteration = 0
            
            while response.get("stopReason") == "tool_use" and iteration < max_iterations:
                iteration += 1
                logger.info(f"Tool use iteration {iteration}")
                
                # Extract tool use from response
                assistant_message = response["output"]["message"]
                messages.append(assistant_message)
                
                # Execute tools
                tool_results = []
                for content_block in assistant_message.get("content", []):
                    if "toolUse" in content_block:
                        tool_use = content_block["toolUse"]
                        tool_name = tool_use["name"]
                        tool_input = tool_use["input"]
                        tool_use_id = tool_use["toolUseId"]
                        
                        # Execute tool
                        tool_result = await self._execute_tool(tool_name, tool_input)
                        
                        # Record tool usage
                        tools_used.append(ToolUse(
                            tool_name=tool_name,
                            input=tool_input,
                            output=tool_result
                        ))
                        
                        # Add tool result to conversation
                        tool_results.append({
                            "toolUseId": tool_use_id,
                            "content": [{"text": tool_result}]
                        })
                
                # Add tool results message
                messages.append({
                    "role": "user",
                    "content": [{"toolResult": tr} for tr in tool_results]
                })
                
                # Continue conversation with tool results
                request_params["messages"] = messages
                response = self.bedrock_client.converse(**request_params)
            
            # Extract final response
            output_message = response["output"]["message"]
            response_text = ""
            
            for content_block in output_message.get("content", []):
                if "text" in content_block:
                    response_text += content_block["text"]
            
            finish_reason = response.get("stopReason", "unknown")
            
            logger.info(f"Chat completed. Tools used: {len(tools_used)}, Finish reason: {finish_reason}")
            
            # Save to session history if session_id provided
            if session_id:
                session_manager.add_to_history(session_id, message, response_text)
                logger.debug(f"Saved Q&A to session {session_id}")
            
            return ChatResponse(
                message=response_text,
                role=MessageRole.ASSISTANT,
                tools_used=tools_used,
                model_id=self.model_id,
                finish_reason=finish_reason,
                session_id=session_id
            )
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            error_msg = f"Bedrock API error ({error_code}): {error_message}"
            logger.error(error_msg)
            logger.debug(f"Full error response: {e.response}")
            raise BedrockChatError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during chat: {str(e)}"
            logger.error(error_msg)
            raise BedrockChatError(error_msg)
    
    async def chat_stream(
        self,
        message: str,
        images: Optional[List[ImageData]] = None,
        conversation_history: Optional[List[ChatMessage]] = None,
        use_search: Optional[bool] = None,
        session_id: Optional[str] = None,
        tone: Optional[str] = "professional",
        language: Optional[str] = "UK - English"
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Generate a streaming chat response using Bedrock with tool calling
        
        Args:
            message: User's message
            images: Optional images to analyze
            conversation_history: Previous messages in the conversation (overrides session history if provided)
            use_search: Force enable/disable search tool
            session_id: Session ID for maintaining conversation history
            
        Yields:
            StreamChunk objects with response content
            
        Raises:
            BedrockChatError: If chat generation fails
        """
        if not self.is_configured():
            raise BedrockChatError("Bedrock client not configured")
        
        # Get conversation history from session if session_id provided
        session_manager = get_session_manager()
        if session_id and conversation_history is None:
            conversation_history = session_manager.get_history(session_id)
            logger.debug(f"Loaded {len(conversation_history)} messages from session {session_id}")
        
        if conversation_history is None:
            conversation_history = []
        
        # Accumulate full response for session storage and word limiting
        full_response = ""
        
        messages = self._format_messages_for_bedrock(message, images, conversation_history)
        
        # Configure tool use
        tool_config = None
        if use_search is False:
            inference_config = {
                "temperature": self.temperature,
                "topP": self.top_p,
                "maxTokens": self.max_tokens
            }
        else:
            tool_config = {"tools": self.TOOLS}
            inference_config = {
                "temperature": self.temperature,
                "topP": self.top_p,
                "maxTokens": self.max_tokens
            }
        
        try:
            # Get formatted system prompt
            system_prompt_text = self._get_system_prompt_with_datetime(tone=tone, language=language)
            logger.debug(f"System prompt length: {len(system_prompt_text)} characters")
            
            request_params = {
                "modelId": self.model_id,
                "messages": messages,
                "system": [{"text": system_prompt_text}],
                "inferenceConfig": inference_config
            }
            
            if tool_config:
                request_params["toolConfig"] = tool_config
            
            logger.debug(f"Making Bedrock streaming API call with model: {self.model_id}, tone: {tone}")
            # Start streaming
            response = self.bedrock_client.converse_stream(**request_params)
            stream = response.get("stream")
            
            if not stream:
                raise BedrockChatError("No stream returned from Bedrock")
            
            tool_use_blocks = {}
            stop_reason = None
            
            # Process stream events
            for event in stream:
                if "contentBlockStart" in event:
                    block_start = event["contentBlockStart"]
                    if "toolUse" in block_start.get("start", {}):
                        tool_use = block_start["start"]["toolUse"]
                        tool_use_id = tool_use["toolUseId"]
                        tool_use_blocks[tool_use_id] = {
                            "name": tool_use["name"],
                            "input": ""
                        }
                
                elif "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"]["delta"]
                    
                    if "text" in delta:
                        # Accumulate for word limiting and session storage
                        full_response += delta["text"]
                        
                        # Stream text content
                        yield StreamChunk(
                            content=delta["text"],
                            is_final=False
                        )
                    
                    elif "toolUse" in delta:
                        # Accumulate tool input
                        tool_use_id = event["contentBlockDelta"]["contentBlockIndex"]
                        if tool_use_id in tool_use_blocks:
                            tool_use_blocks[tool_use_id]["input"] += delta["toolUse"]["input"]
                
                elif "messageStop" in event:
                    stop_reason = event["messageStop"].get("stopReason")
                    logger.info(f"Stream stopped: {stop_reason}")
            
            # Handle tool use if needed
            if stop_reason == "tool_use" and tool_use_blocks:
                logger.info("Processing tool calls in stream")
                
                for tool_use_id, tool_data in tool_use_blocks.items():
                    tool_name = tool_data["name"]
                    tool_input = json.loads(tool_data["input"])
                    
                    # Execute tool
                    tool_result = await self._execute_tool(tool_name, tool_input)
                    
                    # Yield tool use info
                    yield StreamChunk(
                        content="",
                        is_final=False,
                        tool_use=ToolUse(
                            tool_name=tool_name,
                            input=tool_input,
                            output=tool_result
                        )
                    )
                    
                    # Continue conversation with tool result
                    # For streaming, we need to make a new request
                    # This is a simplified version - in production, you might want to
                    # recursively handle multiple tool calls
                    yield StreamChunk(
                        content=f"\n\n[Searching for: {tool_input.get('query', 'information')}...]\n\n",
                        is_final=False
                    )
            
            # Save full response to session history if needed
            if full_response:
                # Save to session history if session_id provided
                if session_id:
                    session_manager.add_to_history(session_id, message, full_response)
                    logger.debug(f"Saved Q&A to session {session_id}")
            
            # Final chunk
            yield StreamChunk(content="", is_final=True)
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            error_msg = f"Bedrock streaming error ({error_code}): {error_message}"
            logger.error(error_msg)
            logger.debug(f"Full error response: {e.response}")
            raise BedrockChatError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during streaming: {str(e)}"
            logger.error(error_msg)
            raise BedrockChatError(error_msg)


# Singleton instance
_chat_service: Optional[GoogliAIChatService] = None


def get_chat_service() -> GoogliAIChatService:
    """Get or create GoogliAIChatService singleton instance"""
    global _chat_service
    if _chat_service is None:
        _chat_service = GoogliAIChatService()
    return _chat_service

