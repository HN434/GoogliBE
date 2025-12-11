"""
Serper API Service for Real-Time Cricket Information Search
"""
import httpx
import logging
from typing import Dict, Any, Optional, List
from config import settings
from models.chat_schemas import SerperResponse, SerperSearchResult

logger = logging.getLogger(__name__)


class SerperAPIError(Exception):
    """Custom exception for Serper API errors"""
    pass


class SerperService:
    """Service for interacting with Serper API for real-time cricket searches"""
    
    def __init__(self):
        """Initialize Serper service"""
        self.api_key = settings.SERPER_API_KEY
        self.api_url = settings.SERPER_API_URL
        self.timeout = 30.0
        
        if not self.api_key:
            logger.warning("SERPER_API_KEY not configured. Real-time search will be unavailable.")
    
    def is_configured(self) -> bool:
        """Check if Serper API is properly configured"""
        return bool(self.api_key)
    
    async def search(
        self,
        query: str,
        num_results: int = 5,
        search_type: str = "search",
        location: Optional[str] = None
    ) -> SerperResponse:
        """
        Perform a search using Serper API
        
        Args:
            query: Search query string
            num_results: Number of results to return (default: 5)
            search_type: Type of search - 'search', 'news', 'images' (default: 'search')
            location: Geographic location for search results (optional)
            
        Returns:
            SerperResponse object with search results
            
        Raises:
            SerperAPIError: If API request fails
        """
        if not self.is_configured():
            raise SerperAPIError("Serper API key not configured")
        
        # Add cricket context to query if not already present
        cricket_keywords = ['cricket', 'ipl', 'test match', 'odi', 't20', 'icc', 'bcci']
        has_cricket_context = any(keyword in query.lower() for keyword in cricket_keywords)
        
        if not has_cricket_context:
            query = f"{query} cricket"
        
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload: Dict[str, Any] = {
            "q": query,
            "num": num_results,
            "type": search_type
        }
        
        if location:
            payload["location"] = location
        
        logger.info(f"Performing Serper search: query='{query}', num_results={num_results}")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.api_url,
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                
                data = response.json()
                logger.info(f"Serper search successful. Found {len(data.get('organic', []))} results")
                
                return SerperResponse(**data)
                
        except httpx.HTTPStatusError as e:
            error_msg = f"Serper API HTTP error: {e.response.status_code} - {e.response.text}"
            logger.error(error_msg)
            raise SerperAPIError(error_msg)
        except httpx.TimeoutException:
            error_msg = "Serper API request timed out"
            logger.error(error_msg)
            raise SerperAPIError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during Serper search: {str(e)}"
            logger.error(error_msg)
            raise SerperAPIError(error_msg)
    
    def format_search_results(self, serper_response: SerperResponse, max_results: int = 5) -> str:
        """
        Format search results into a readable string for the AI model
        
        Args:
            serper_response: Response from Serper API
            max_results: Maximum number of results to include
            
        Returns:
            Formatted string with search results
        """
        formatted = "**Search Results:**\n\n"
        
        # Add answer box if available
        if serper_response.answerBox:
            answer = serper_response.answerBox.get('answer') or serper_response.answerBox.get('snippet')
            if answer:
                formatted += f"**Quick Answer:**\n{answer}\n\n"
        
        # Add knowledge graph if available
        if serper_response.knowledgeGraph:
            kg = serper_response.knowledgeGraph
            if kg.get('title'):
                formatted += f"**Knowledge Graph:**\n"
                formatted += f"Title: {kg['title']}\n"
                if kg.get('description'):
                    formatted += f"Description: {kg['description']}\n"
                formatted += "\n"
        
        # Add organic results
        if serper_response.organic:
            formatted += "**Detailed Results:**\n"
            for i, result in enumerate(serper_response.organic[:max_results], 1):
                formatted += f"{i}. **{result.title}**\n"
                formatted += f"   {result.snippet}\n"
                formatted += f"   Source: {result.link}\n\n"
        
        return formatted
    
    async def search_cricket_info(self, query: str) -> str:
        """
        Convenience method to search for cricket information and return formatted results
        
        Args:
            query: Cricket-related search query
            
        Returns:
            Formatted search results as string
        """
        try:
            response = await self.search(query, num_results=5)
            return self.format_search_results(response)
        except SerperAPIError as e:
            logger.error(f"Failed to search cricket info: {e}")
            return f"Unable to fetch real-time information: {str(e)}"


# Singleton instance
_serper_service: Optional[SerperService] = None


def get_serper_service() -> SerperService:
    """Get or create SerperService singleton instance"""
    global _serper_service
    if _serper_service is None:
        _serper_service = SerperService()
    return _serper_service

