"""
API Key Rotator - Manages rotation of multiple RapidAPI keys
Implements round-robin rotation to distribute load across keys
"""

import asyncio
import logging
from typing import List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class APIKeyRotator:
    """
    Manages rotation of multiple API keys to distribute load
    Implements round-robin with automatic fallback on rate limit errors
    """
    
    def __init__(self, api_keys: List[str]):
        """
        Initialize key rotator
        
        Args:
            api_keys: List of API keys to rotate through
        """
        if not api_keys:
            raise ValueError("At least one API key is required")
        
        # Filter out None/empty keys
        self.api_keys = [key for key in api_keys if key and key.strip()]
        
        if not self.api_keys:
            raise ValueError("No valid API keys provided")
        
        self.current_index = 0
        self.lock = asyncio.Lock()
        
        # Track rate limit errors per key
        self.rate_limited_keys = set()
        self.key_error_counts = defaultdict(int)
        self.max_errors_per_key = 3  # Temporarily skip key after 3 consecutive errors
        
        logger.info(f"Initialized API key rotator with {len(self.api_keys)} keys")
    
    def get_current_key(self) -> str:
        """
        Get the current API key (thread-safe)
        
        Returns:
            Current API key
        """
        return self.api_keys[self.current_index]
    
    async def get_next_key(self) -> str:
        """
        Get the next available API key (round-robin with error handling)
        Skips keys that are rate-limited or have too many errors
        
        Returns:
            Next available API key
        """
        async with self.lock:
            # Find next available key
            attempts = 0
            while attempts < len(self.api_keys):
                key = self.api_keys[self.current_index]
                
                # Check if key is temporarily disabled
                if key not in self.rate_limited_keys:
                    # Move to next key for next call
                    self.current_index = (self.current_index + 1) % len(self.api_keys)
                    return key
                
                # Key is rate-limited, try next one
                self.current_index = (self.current_index + 1) % len(self.api_keys)
                attempts += 1
            
            # All keys are rate-limited, reset and use current
            logger.warning("All keys appear to be rate-limited, resetting and using current key")
            self.rate_limited_keys.clear()
            return self.api_keys[self.current_index]
    
    def mark_key_rate_limited(self, key: str):
        """
        Mark a key as rate-limited (temporarily disabled)
        
        Args:
            key: The API key that was rate-limited
        """
        if key in self.api_keys:
            self.rate_limited_keys.add(key)
            self.key_error_counts[key] += 1
            logger.warning(f"Key {self._mask_key(key)} marked as rate-limited (error count: {self.key_error_counts[key]})")
    
    def mark_key_success(self, key: str):
        """
        Mark a key as successful (reset error count and remove from rate-limited set)
        
        Args:
            key: The API key that succeeded
        """
        if key in self.api_keys:
            self.rate_limited_keys.discard(key)
            self.key_error_counts[key] = 0
            logger.debug(f"Key {self._mask_key(key)} marked as successful")
    
    def is_rate_limited(self, key: str) -> bool:
        """
        Check if a key is currently rate-limited
        
        Args:
            key: The API key to check
            
        Returns:
            True if key is rate-limited
        """
        return key in self.rate_limited_keys
    
    def reset_key(self, key: str):
        """
        Reset a key's error state (remove from rate-limited set)
        
        Args:
            key: The API key to reset
        """
        if key in self.api_keys:
            self.rate_limited_keys.discard(key)
            self.key_error_counts[key] = 0
            logger.info(f"Key {self._mask_key(key)} reset")
    
    def reset_all_keys(self):
        """Reset all keys (clear rate-limited set)"""
        self.rate_limited_keys.clear()
        self.key_error_counts.clear()
        logger.info("All keys reset")
    
    def get_key_stats(self) -> dict:
        """
        Get statistics about key usage
        
        Returns:
            Dictionary with key statistics
        """
        return {
            "total_keys": len(self.api_keys),
            "active_keys": len(self.api_keys) - len(self.rate_limited_keys),
            "rate_limited_keys": len(self.rate_limited_keys),
            "current_index": self.current_index,
            "key_error_counts": {
                self._mask_key(k): v for k, v in self.key_error_counts.items()
            }
        }
    
    @staticmethod
    def _mask_key(key: str) -> str:
        """
        Mask API key for logging (show first 8 and last 4 characters)
        
        Args:
            key: API key to mask
            
        Returns:
            Masked key string
        """
        if not key or len(key) < 12:
            return "***"
        return f"{key[:8]}...{key[-4:]}"
    
    def should_retry_with_different_key(self, status_code: int) -> bool:
        """
        Determine if we should retry with a different key based on HTTP status code
        
        Args:
            status_code: HTTP status code from API response
            
        Returns:
            True if we should try a different key
        """
        # Rate limit errors
        if status_code == 429:
            return True
        
        # Authentication errors (might indicate key issue)
        if status_code == 401:
            return True
        
        # Forbidden (might indicate key quota exceeded)
        if status_code == 403:
            return True
        
        return False

