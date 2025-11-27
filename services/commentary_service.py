"""
Commentary Service for fetching live cricket commentary from RapidAPI
Handles API calls, error handling, and data normalization
"""

import httpx
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from models.commentary_schemas import CommentaryLine, CommentaryEventType, MatchInfo
from utils.api_key_rotator import APIKeyRotator

logger = logging.getLogger(__name__)

class CommentaryService:
    """
    Service for fetching cricket commentary from RapidAPI
    Handles API communication, error handling, and response normalization
    """

    def __init__(
        self,
        api_keys: List[str],
        api_host: str = "cricbuzz-cricket.p.rapidapi.com",
        base_url: str = "https://cricbuzz-cricket.p.rapidapi.com",
        timeout: int = 10,
        max_retries: int = 3
    ):
        """
        Initialize commentary service
        
        Args:
            api_keys: List of RapidAPI keys for rotation (or single key in list)
            api_host: RapidAPI host header
            base_url: Base URL for API requests
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        # Initialize key rotator
        self.key_rotator = APIKeyRotator(api_keys)
        self.api_host = api_host
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        
        # HTTP client without persistent headers (we'll set key per request)
        self.client = httpx.AsyncClient(timeout=timeout)

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

    async def fetch_match_info(self, match_id: str) -> Optional[MatchInfo]:
        """
        Fetch match information
        
        Args:
            match_id: Match identifier
        
        Returns:
            MatchInfo object or None if not found
        """
        # Use commentary endpoint as it includes matchHeader
        url = f"{self.base_url}/mcenter/v1/{match_id}/comm"
        
        try:
            response = await self._make_request(url)
            if not response:
                return None
            
            # Extract match header from commentary response
            # Handle both "matchheaders" (new) and "matchHeader" (old)
            match_header = response.get("matchheaders", response.get("matchHeader", {}))
            miniscore = response.get("miniscore", {})
            
            # Get team info
            team1 = match_header.get("team1", {})
            team2 = match_header.get("team2", {})
            
            # Get status
            status = match_header.get("state", match_header.get("status", ""))
            
            # Get match date
            match_start = match_header.get("matchstarttimestamp", match_header.get("matchStartTimestamp"))
            date = None
            if match_start:
                if isinstance(match_start, (int, float)):
                    if match_start > 1e10:  # milliseconds
                        date = datetime.fromtimestamp(match_start / 1000)
                    else:
                        date = datetime.fromtimestamp(match_start)
            
            # Get score from miniscore
            score = None
            if miniscore:
                bat_team = miniscore.get("batTeam", {})
                score = {
                    "runs": bat_team.get("teamScore"),
                    "wickets": bat_team.get("teamWkts"),
                    "overs": miniscore.get("overs")
                }
            
            return MatchInfo(
                match_id=match_id,
                team1=team1.get("name"),
                team2=team2.get("name"),
                status=status,
                venue=None,  # Not in commentary response
                date=date,
                score=score
            )
        except Exception as e:
            logger.error(f"Error fetching match info for {match_id}: {e}")
            return None

    async def fetch_commentary(
        self,
        match_id: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[CommentaryLine], Optional[Dict[str, Any]]]:
        """
        Fetch commentary for a match
        
        Args:
            match_id: Match identifier
            params: Optional query params for pagination (e.g., tms, iid)
        
        Returns:
            Tuple of (commentary lines, miniscore dict)
        """
        url = f"{self.base_url}/mcenter/v1/{match_id}/comm"
        logger.debug(f"Fetching commentary from {url}")
        
        try:
            response = await self._make_request(url, params=params)
            if not response:
                logger.warning(f"No response from API for match {match_id}")
                return [], None
            
            logger.debug(f"Received API response for match {match_id}: {type(response)}")
            
            # Normalize API response to CommentaryLine objects
            commentary_lines = []
            
            # Handle different possible response structures
            # Try comwrapper first (new format)
            comwrapper = response.get("comwrapper", [])
            if comwrapper and isinstance(comwrapper, list):
                logger.debug(f"Found comwrapper with {len(comwrapper)} items")
                for idx, wrapper_item in enumerate(comwrapper):
                    # Extract commentary object from wrapper
                    commentary_obj = wrapper_item.get("commentary", {})
                    if commentary_obj:
                        line = self._normalize_commentary_line(commentary_obj, match_id)
                        if line:
                            commentary_lines.append(line)
                        else:
                            logger.debug(f"Skipped invalid commentary line {idx} for match {match_id}")
            
            # Fallback to commentaryList (old format)
            elif not commentary_lines:
                commentary_data = response.get("commentaryList", [])
                if not commentary_data:
                    # Try alternative response structures
                    commentary_data = response.get("commentary", [])
                    if not commentary_data:
                        if isinstance(response, list):
                            commentary_data = response
                        elif "data" in response:
                            commentary_data = response.get("data", [])
                        elif "items" in response:
                            commentary_data = response.get("items", [])
                
                logger.debug(f"Found {len(commentary_data) if isinstance(commentary_data, list) else 0} commentary items in response")
                
                if isinstance(commentary_data, list):
                    for idx, item in enumerate(commentary_data):
                        line = self._normalize_commentary_line(item, match_id)
                        if line:
                            commentary_lines.append(line)
                        else:
                            logger.debug(f"Skipped invalid commentary line {idx} for match {match_id}")
                else:
                    logger.warning(f"Commentary data is not a list for match {match_id}: {type(commentary_data)}")
            
            miniscore = response.get("miniscore")
            
            logger.info(f"✅ Fetched {len(commentary_lines)} commentary lines for match {match_id}")
            return commentary_lines, miniscore
            
        except Exception as e:
            logger.error(f"❌ Error fetching commentary for {match_id}: {e}", exc_info=True)
            return []

    async def fetch_previous_commentaries(
        self,
        match_id: str,
        timestamp_ms: Optional[int] = None,
        innings_id: Optional[int] = None,
    ) -> Tuple[List[CommentaryLine], Optional[Dict[str, Any]]]:
        """
        Fetch previous commentary page for a match using pagination params
        """
        params: Dict[str, Any] = {}
        if timestamp_ms is not None:
            params["tms"] = timestamp_ms
        if innings_id is not None:
            params["iid"] = innings_id
        return await self.fetch_commentary(match_id, params=params or None)

    def _normalize_commentary_line(self, data: Dict[str, Any], match_id: str) -> Optional[CommentaryLine]:
        """
        Normalize API response to CommentaryLine
        Handles RapidAPI cricket commentary format
        
        Args:
            data: Raw commentary data from API
            match_id: Match identifier
        
        Returns:
            CommentaryLine object or None if invalid
        """
        try:
            # Generate unique ID from timestamp and ball number
            timestamp_val = data.get("timestamp", 0)
            ball_nbr = data.get("ballnbr", data.get("ballNbr", 0))
            line_id = f"{match_id}_{timestamp_val}_{ball_nbr}"
            
            # Parse timestamp (RapidAPI uses milliseconds)
            timestamp_val = data.get("timestamp")
            if isinstance(timestamp_val, (int, float)):
                # Convert milliseconds to seconds for fromtimestamp
                if timestamp_val > 1e10:  # Likely milliseconds
                    timestamp = datetime.fromtimestamp(timestamp_val / 1000)
                else:  # Likely seconds
                    timestamp = datetime.fromtimestamp(timestamp_val)
            elif isinstance(timestamp_val, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp_val.replace("Z", "+00:00"))
                except:
                    timestamp = datetime.now()
            else:
                timestamp = datetime.now()
            
            # Extract text - handle both "commtxt" (new) and "commText" (old)
            text = data.get("commtxt", data.get("commText", ""))
            if not text or text.strip() == "":
                # Skip empty commentary lines
                return None
            
            # Clean up text - remove format markers like "B0$"
            # New format: commentaryformats is an array with objects containing type and value arrays
            commentary_formats = data.get("commentaryformats", data.get("commentaryFormats", []))
            
            if isinstance(commentary_formats, list):
                # New format: array of format objects
                for fmt_obj in commentary_formats:
                    fmt_type = fmt_obj.get("type", "")
                    fmt_values = fmt_obj.get("value", [])
                    
                    if fmt_type == "bold" and isinstance(fmt_values, list):
                        # Replace format markers with actual values
                        for fmt_item in fmt_values:
                            if isinstance(fmt_item, dict):
                                fmt_id = fmt_item.get("id", "")
                                fmt_value = fmt_item.get("value", "")
                                if fmt_id and fmt_value:
                                    text = text.replace(fmt_id, fmt_value)
            elif isinstance(commentary_formats, dict):
                # Old format: dictionary with bold key
                if "bold" in commentary_formats:
                    bold_data = commentary_formats["bold"]
                    format_ids = bold_data.get("formatId", [])
                    format_values = bold_data.get("formatValue", [])
                    
                    # Replace format markers with actual values
                    for fmt_id, fmt_value in zip(format_ids, format_values):
                        text = text.replace(fmt_id, fmt_value)
            
            # Extract event type - handle both "eventtype" (new) and "event" (old)
            event_str = data.get("eventtype", data.get("event", "NONE")).upper()
            # Handle comma-separated events like "over-break,SIX"
            if "," in event_str:
                event_str = event_str.split(",")[-1].strip()  # Take the last event
            event_type = self._map_event_type(event_str)
            
            # Extract additional metadata - handle both formats
            ball_number = data.get("ballnbr", data.get("ballNbr"))
            over_number = data.get("overnum", data.get("overNumber"))
            
            # Extract runs from text (e.g., "2 runs", "FOUR", "SIX")
            runs = self._extract_runs_from_text(text, event_type)
            
            # Extract wickets (1 if event is WICKET)
            wickets = 1 if event_type == CommentaryEventType.WICKET else None
            
            return CommentaryLine(
                id=line_id,
                timestamp=timestamp,
                text=text.strip(),
                event_type=event_type,
                ball_number=ball_number if ball_number else None,
                over_number=over_number if over_number else None,
                runs=runs,
                wickets=wickets,
            metadata={
                "raw_data": data,
                "match_id": match_id,
                "inningsid": data.get("inningsid", data.get("inningsId")),
                "eventtype": data.get("eventtype", data.get("event")),
                "timestamp_ms": data.get("timestamp"),
            }
            )
            
        except Exception as e:
            logger.error(f"Error normalizing commentary line: {e}")
            return None

    def _map_event_type(self, event_str: str) -> CommentaryEventType:
        """
        Map RapidAPI event string to CommentaryEventType
        
        Args:
            event_str: Event string from API (e.g., "WICKET", "FOUR", "SIX")
        
        Returns:
            CommentaryEventType
        """
        event_map = {
            "WICKET": CommentaryEventType.WICKET,
            "FOUR": CommentaryEventType.FOUR,
            "SIX": CommentaryEventType.SIX,
            "OVER-BREAK": CommentaryEventType.OVER,
            "OVER": CommentaryEventType.OVER,
            "NONE": CommentaryEventType.BALL,
        }
        
        return event_map.get(event_str, CommentaryEventType.OTHER)
    
    def _extract_runs_from_text(self, text: str, event_type: CommentaryEventType) -> Optional[int]:
        """
        Extract runs scored from commentary text
        
        Args:
            text: Commentary text
            event_type: Event type
        
        Returns:
            Number of runs or None
        """
        # Handle explicit event types
        if event_type == CommentaryEventType.FOUR:
            return 4
        elif event_type == CommentaryEventType.SIX:
            return 6
        
        # Try to extract from text patterns
        text_lower = text.lower()
        
        # Look for patterns like "2 runs", "1 run", "3 runs"
        import re
        run_patterns = [
            r'(\d+)\s+run',  # "2 runs", "1 run"
            r'(\d+)\s+run',  # "2 run" (without s)
        ]
        
        for pattern in run_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return int(match.group(1))
        
        # Check for "no run" or "dot"
        if "no run" in text_lower or "dot" in text_lower:
            return 0
        
        return None
    
    def _determine_event_type(self, text: str, data: Dict[str, Any]) -> CommentaryEventType:
        """
        Determine event type from commentary text (fallback method)
        
        Args:
            text: Commentary text
            data: Raw data dictionary
        
        Returns:
            CommentaryEventType
        """
        text_lower = text.lower()
        
        # Check explicit event type in data
        event_type_str = data.get("event_type", "").lower()
        if event_type_str:
            try:
                return CommentaryEventType(event_type_str)
            except ValueError:
                pass
        
        # Infer from text content
        if any(word in text_lower for word in ["wicket", "out", "dismissed", "caught", "bowled"]):
            return CommentaryEventType.WICKET
        elif "six" in text_lower or "6" in text_lower:
            return CommentaryEventType.SIX
        elif "four" in text_lower or "4" in text_lower:
            return CommentaryEventType.FOUR
        elif "over" in text_lower:
            return CommentaryEventType.OVER
        elif any(word in text_lower for word in ["ball", "delivery"]):
            return CommentaryEventType.BALL
        else:
            return CommentaryEventType.OTHER

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        """Parse datetime from various formats"""
        if not value:
            return None
        
        if isinstance(value, datetime):
            return value
        
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except:
                pass
        
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value)
        
        return None

    async def get_match_status(self, match_id: str) -> Optional[Dict[str, Any]]:
        """
        Get match status and result information
        
        Args:
            match_id: Match identifier
        
        Returns:
            Dictionary with match status, state, and result, or None if not found
        """
        url = f"{self.base_url}/mcenter/v1/{match_id}/comm"
        
        try:
            response = await self._make_request(url)
            if not response:
                return None
            
            # Handle both "matchheaders" (new) and "matchHeader" (old)
            match_header = response.get("matchheaders", response.get("matchHeader", {}))
            
            # Extract status information
            status_info = {
                "state": match_header.get("state", ""),
                "status": match_header.get("status", ""),  # Contains result like "Team won by X runs"
                "complete": match_header.get("complete", False),
                "winning_team_id": match_header.get("winningteamid", match_header.get("winningTeamId")),
                "match_format": match_header.get("matchformat", match_header.get("matchFormat", "")),
                "series_name": match_header.get("seriesname", match_header.get("seriesName", "")),
                "match_desc": match_header.get("matchdesc", match_header.get("matchDesc", ""))
            }
            
            # Get team names
            team1 = match_header.get("team1", {})
            team2 = match_header.get("team2", {})
            status_info["team1"] = team1.get("name", team1.get("teamname", ""))
            status_info["team2"] = team2.get("name", team2.get("teamname", ""))
            
            # Get winning team name if available
            if status_info["winning_team_id"]:
                if team1.get("teamid", team1.get("teamId")) == status_info["winning_team_id"]:
                    status_info["winning_team"] = status_info["team1"]
                elif team2.get("teamid", team2.get("teamId")) == status_info["winning_team_id"]:
                    status_info["winning_team"] = status_info["team2"]
            
            return status_info
            
        except Exception as e:
            logger.error(f"Error getting match status for {match_id}: {e}")
            return None

    async def is_match_finished(self, match_id: str) -> bool:
        """
        Check if a match has finished
        
        Args:
            match_id: Match identifier
        
        Returns:
            True if match is finished
        """
        # Fetch commentary to get match header
        url = f"{self.base_url}/mcenter/v1/{match_id}/comm"
        
        try:
            response = await self._make_request(url)
            if not response:
                return False
            
            # Handle both "matchheaders" (new) and "matchHeader" (old)
            match_header = response.get("matchheaders", response.get("matchHeader", {}))
            
            # Check state/status fields
            state = (match_header.get("state", "") or "").upper()
            status_text = (match_header.get("status", "") or "").upper()
            logger.debug(f"Match {match_id} state: {state}, status: {status_text}")
            
            state_flags = ["COMPLETE", "FINISHED", "ABANDONED", "CANCELLED", "STUMPS"]
            if any(flag in state for flag in state_flags) or "STUMP" in status_text:
                logger.info(f"Match {match_id} is finished (state/status indicates completion)")
                return True
            
            # Check complete field
            if match_header.get("complete") is True:
                logger.info(f"Match {match_id} is finished (complete: true)")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking if match {match_id} is finished: {e}")
            return False

    async def _make_request(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Make HTTP request with retry logic and key rotation
        
        Args:
            url: Request URL
            params: Optional query parameters
        
        Returns:
            Response JSON as dictionary or None on failure
        """
        last_error = None
        keys_tried = set()
        max_key_attempts = len(self.key_rotator.api_keys)
        
        for attempt in range(self.max_retries):
            # Get next available key
            current_key = await self.key_rotator.get_next_key()
            
            # If we've tried all keys, reset and try again
            if len(keys_tried) >= max_key_attempts:
                keys_tried.clear()
                self.key_rotator.reset_all_keys()
                current_key = await self.key_rotator.get_next_key()
            
            keys_tried.add(current_key)
            
            try:
                logger.debug(f"Making API request to {url} (attempt {attempt + 1}/{self.max_retries}) with key {self.key_rotator._mask_key(current_key)}")
                
                # Set headers with current key
                headers = {
                    "X-RapidAPI-Key": current_key,
                    "X-RapidAPI-Host": self.api_host
                }
                
                response = await self.client.get(url, params=params, headers=headers)
                logger.debug(f"Response status: {response.status_code}")
                
                # Check for rate limit or auth errors
                if self.key_rotator.should_retry_with_different_key(response.status_code):
                    self.key_rotator.mark_key_rate_limited(current_key)
                    logger.warning(f"Key {self.key_rotator._mask_key(current_key)} returned status {response.status_code}, will try different key")
                    
                    # If not last attempt, continue to next iteration with different key
                    if attempt < self.max_retries - 1:
                        continue
                
                response.raise_for_status()
                data = response.json()
                
                # Mark key as successful
                self.key_rotator.mark_key_success(current_key)
                
                logger.debug(f"Successfully parsed JSON response (keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'})")
                return data
                
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                
                if status_code == 404:
                    logger.warning(f"Resource not found (404): {url}")
                    # Don't mark 404 as key failure
                    return None
                
                # Mark key as problematic if it's a rate limit or auth error
                if self.key_rotator.should_retry_with_different_key(status_code):
                    self.key_rotator.mark_key_rate_limited(current_key)
                    logger.warning(f"Key {self.key_rotator._mask_key(current_key)} failed with status {status_code}")
                
                last_error = e
                logger.warning(f"HTTP error on attempt {attempt + 1}/{self.max_retries}: {status_code} - {e}")
                
            except httpx.RequestError as e:
                last_error = e
                logger.warning(f"Request error on attempt {attempt + 1}/{self.max_retries}: {e}")
                
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error on attempt {attempt + 1}/{self.max_retries}: {e}", exc_info=True)
            
            # Wait before retry (exponential backoff)
            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt
                logger.debug(f"Waiting {wait_time} seconds before retry...")
                import asyncio
                await asyncio.sleep(wait_time)
        
        logger.error(f"❌ Failed to fetch {url} after {self.max_retries} attempts: {last_error}")
        return None

