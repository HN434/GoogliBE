"""
Translation services for live commentary.
Provides Bedrock-powered translation plus orchestration helpers that
cache translations and publish localized payloads.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import re
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from services.redis_service import RedisService

logger = logging.getLogger(__name__)


class BedrockTranslationService:
    """Thin wrapper around AWS Bedrock runtime for batch translations.

    This version is defensive: it normalizes many shapes of Bedrock/Claude-style
    responses into a canonical List[{"id": str, "text": str}].
    """

    def __init__(
        self,
        model_id: str,
        region: str,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 10240,
        max_workers: int = 20,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.client = boto3.client("bedrock-runtime", region_name=region)
        
        # Create dedicated ThreadPoolExecutor for concurrent Bedrock API calls
        # Default of 20 workers allows multiple batches to run truly in parallel
        # AWS Bedrock supports 10-50 concurrent requests depending on model/region
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="bedrock-translate"
        )
        logger.info(
            "BedrockTranslationService initialized with %d worker threads for parallel execution",
            max_workers
        )

    def _build_prompt(self, items: List[Dict[str, str]], source_lang: str, target_lang: str) -> str:
        payload = json.dumps(items, ensure_ascii=False)
        return (
            "You are a professional cricket commentator and translation engine. "
            f"Translate each entry from {source_lang} to {target_lang}. "
            "and avoid transliteration for numbers. "
            "Transliterate the text to the target language using the correct script and characters."
            "Return ONLY valid JSON in this format: "
            '[{"id": "<original id>", "text": "<translated text>"}]. '
            f"Input entries: {payload}"
        )

    def _is_amazon_nova_model(self) -> bool:
        """Check if the model is an Amazon Nova model."""
        return self.model_id and (
            self.model_id.startswith("us.amazon.nova") or 
            self.model_id.startswith("amazon.nova")
        )
    
    def _is_anthropic_model(self) -> bool:
        """Check if the model is an Anthropic Claude model."""
        return self.model_id and (
            self.model_id.startswith("anthropic.") or
            "claude" in self.model_id.lower()
        )
    
    def _invoke_bedrock(self, prompt: str) -> str:
        """
        Synchronously call Bedrock and return the raw text extracted from the response payload.
        This returns the text segment (string) the model produced — which may itself be JSON.
        """
        # Format request body based on model type
        if self._is_amazon_nova_model():
            # Amazon Nova models use messages format with inferenceConfig wrapper
            body = json.dumps({
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ],
                "inferenceConfig": {
                    "max_new_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p
                }
            })
        elif self._is_anthropic_model():
            # Anthropic Claude models use messages format with anthropic_version
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}],
                    }
                ],
            })
        else:
            # Default to Anthropic format for unknown models
            logger.warning(
                "Unknown model type '%s', defaulting to Anthropic format. "
                "If this fails, the model may require a different format.",
                self.model_id
            )
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}],
                    }
                ],
            })
        
        start_time = time.time()
        thread_id = threading.current_thread().ident
        logger.info("Bedrock API call STARTED at %.3f on thread %s", start_time, thread_id)
        
        try:
            response = self.client.invoke_model(modelId=self.model_id, body=body)
            api_duration = time.time() - start_time
            logger.info(
                "Bedrock API call COMPLETED in %.2f seconds on thread %s",
                api_duration,
                thread_id
            )
        except (BotoCoreError, ClientError) as exc:
            api_duration = time.time() - start_time
            logger.exception(
                "invoke_model failed after %.2f seconds on thread %s",
                api_duration,
                thread_id
            )
            raise

        streaming_body = response.get("body")
        if hasattr(streaming_body, "iter_chunks"):
            chunks = []
            for chunk in streaming_body.iter_chunks():
                if chunk:
                    chunks.append(chunk)
            raw_body = b"".join(chunks)
        else:
            # file-like object or bytes
            raw_body = streaming_body.read() if streaming_body else b""

        if isinstance(raw_body, (bytes, bytearray)):
            raw_body = raw_body.decode("utf-8", errors="ignore")

        # helpful debug but avoid huge dumps
        logger.debug(
            "Bedrock raw response: %s",
            raw_body[:500] + ("..." if isinstance(raw_body, str) and len(raw_body) > 500 else ""),
        )

        # Attempt to parse the outer JSON to extract nested text if present.
        # Many Bedrock responses are JSON wrappers that contain the actual model text
        # in places like `content[0].text`, `output`, or `message`.
        try:
            outer = json.loads(raw_body)
        except json.JSONDecodeError:
            # Not valid JSON at top level — assume raw_body itself is the model text
            return raw_body.strip()

        # If top-level parsed to a dict, try to pull the text
        text = self._extract_text(outer)
        if text:
            return text.strip()

        # If we couldn't find text in known keys, fallback to raw string representation
        return raw_body.strip()

    def _extract_text(self, payload: Any) -> str:
        """
        Walk common wrapper shapes to extract the model's textual output.
        Looks at keys like 'content', 'output', 'message' and nested 'text' fields.
        Handles both Anthropic Claude and Amazon Nova response formats.
        """
        def from_block(block: Any) -> Optional[str]:
            if isinstance(block, list):
                for item in block:
                    if isinstance(item, dict):
                        # Common shape: {"type":"text","text":"..."} or {"text":"..."}
                        if "text" in item and isinstance(item["text"], str):
                            return item["text"]
                        # Nested content field
                        if "content" in item:
                            nested = from_block(item["content"])
                            if nested:
                                return nested
            elif isinstance(block, dict):
                if "text" in block and isinstance(block["text"], str):
                    return block["text"]
                if "content" in block:
                    return from_block(block["content"])
            return None

        # Amazon Nova models typically return text in output.message.content or output.text
        if isinstance(payload, dict):
            output = payload.get("output")
            if isinstance(output, dict):
                # Check for output.message.content (Nova format)
                message = output.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, list) and content:
                        # Content might be an array with text objects
                        first = content[0]
                        if isinstance(first, dict) and "text" in first:
                            return first["text"]
                # Check for output.text directly
                if "text" in output and isinstance(output["text"], str):
                    return output["text"]
            # Also check for output as a string directly
            if "output" in payload and isinstance(payload["output"], str):
                return payload["output"]
            # Check for content as string directly (Nova may return this)
            if "content" in payload and isinstance(payload["content"], str):
                return payload["content"]

        # Try common top-level keys
        for key in ("output", "content", "message", "body", "response"):
            if key in payload:
                found = from_block(payload[key])
                if found:
                    return found

        # Sometimes the model text is under payload.get('content')[0]['text']
        if isinstance(payload, dict) and "content" in payload:
            c = payload.get("content")
            if isinstance(c, list) and c:
                first = c[0]
                if isinstance(first, dict) and "text" in first and isinstance(first["text"], str):
                    return first["text"]

        # Nothing found
        return ""

    def _try_parse_json_text(self, text: str) -> Optional[List[Dict[str, str]]]:
        """
        Try to parse `text` as JSON. Accept either a list of objects or a single object.
        Return list on success, None on failure.
        """
        if not isinstance(text, str):
            return None

        s = text.strip()
        # Remove common fences/backticks
        if s.startswith("```") and s.endswith("```"):
            s = s.strip("`")
        s = s.strip("`").strip()

        try:
            data = json.loads(s)
        except json.JSONDecodeError:
            return None

        if isinstance(data, list):
            out = []
            for item in data:
                if isinstance(item, dict) and "id" in item and "text" in item:
                    out.append({"id": item["id"], "text": item.get("text", "")})
            return out if out else None

        if isinstance(data, dict):
            # direct object -> wrap if it has id/text
            if "id" in data and "text" in data:
                return [{"id": data["id"], "text": data.get("text", "")}]
            # wrapper shape with content/text inside
            content = data.get("content")
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict) and "text" in first:
                    return self._try_parse_json_text(first["text"])
        return None

    def _extract_json_substring(self, text: str) -> Optional[str]:
        """
        Find the first balanced JSON array or object substring in `text`.
        Returns the substring or None.
        """
        if not isinstance(text, str):
            return None

        # Look for first balanced '[' ... ']' block
        start_idx = text.find("[")
        if start_idx != -1:
            depth = 0
            for i in range(start_idx, len(text)):
                if text[i] == "[":
                    depth += 1
                elif text[i] == "]":
                    depth -= 1
                    if depth == 0:
                        return text[start_idx : i + 1]

        # Fallback to object {...}
        start_idx = text.find("{")
        if start_idx != -1:
            depth = 0
            for i in range(start_idx, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start_idx : i + 1]

        return None

    def _regex_extract_pairs(self, text: str) -> List[Dict[str, str]]:
        """
        Last-resort extraction using regex to find {"id":"...","text":"..."} fragments.
        Handles multiline text values.
        """
        items: List[Dict[str, str]] = []
        pattern = re.compile(
            r'\{\s*"id"\s*:\s*"(?P<id>[^"]+)"\s*,\s*"text"\s*:\s*"(?P<text>.*?)"\s*\}', re.DOTALL
        )
        for m in pattern.finditer(text):
            gid = m.group("id")
            gtext = m.group("text")
            try:
                # decode common escape sequences
                gtext_fixed = bytes(gtext, "utf-8").decode("unicode_escape")
            except Exception:
                gtext_fixed = gtext
            items.append({"id": gid, "text": gtext_fixed})
        return items

    async def translate_batch(
        self,
        items: List[Dict[str, str]],
        source_lang: str,
        target_lang: str,
    ) -> List[Dict[str, str]]:
        """
        Translate commentary lines using a single Bedrock call.
        Returns list of {"id": str, "text": translated_text}.
        """
        if not items:
            return []

        prompt = self._build_prompt(items, source_lang, target_lang)

        loop = asyncio.get_running_loop()
        executor_start = time.time()
        logger.debug("Submitting batch (%d items) to executor at %.3f", len(items), executor_start)
        
        try:
            # Use dedicated executor for true parallel execution (not default None executor)
            raw_text = await loop.run_in_executor(self.executor, lambda: self._invoke_bedrock(prompt))
            executor_duration = time.time() - executor_start
            logger.debug("Executor completed batch (%d items) in %.2f seconds", len(items), executor_duration)
        except (BotoCoreError, ClientError) as exc:
            executor_duration = time.time() - executor_start
            logger.exception("Bedrock invoke failed after %.2f seconds", executor_duration)
            raise

        # If we got bytes somehow, ensure str
        if isinstance(raw_text, (bytes, bytearray)):
            raw_text = raw_text.decode("utf-8", errors="ignore")

        # Primary strategy: try to parse the raw text directly as JSON
        parsed = self._try_parse_json_text(raw_text)
        if parsed:
            return parsed

        # Secondary: if the raw_text contains a JSON substring, try parsing that
        json_sub = self._extract_json_substring(raw_text)
        if json_sub:
            parsed = self._try_parse_json_text(json_sub)
            if parsed:
                return parsed

        # Tertiary: regex-based extraction of id/text pairs
        regex_items = self._regex_extract_pairs(raw_text)
        if regex_items:
            return regex_items

        # Nothing parsed — log and raise a JSON error so upstream can decide what to do.
        logger.error(
            "Bedrock translation produced unparsable output. raw_text (truncated)=%s",
            (raw_text[:500] + "...") if isinstance(raw_text, str) and len(raw_text) > 500 else raw_text,
        )
        # raise JSONDecodeError-like exception for consistency with previous behaviour
        raise json.JSONDecodeError("Unable to parse translator output", raw_text, 0)



class CommentaryTranslationService:
    """
    Coordinates translation for commentary snapshots:
    - reads cached translations from Redis
    - invokes Bedrock only for missing lines
    - publishes localized updates to language-specific channels

    This version is resilient to malformed translator responses (stringified JSON,
    JSON embedded inside 'content[0].text', or partially invalid JSON).
    """

    def __init__(
        self,
        redis_service: "RedisService",
        translator: "BedrockTranslationService",
        source_language: str = "en",
        cache_ttl_seconds: int = 900,
    ):
        self.redis_service = redis_service
        self.translator = translator
        self.source_language = source_language
        self.cache_ttl = cache_ttl_seconds

    async def translate_and_publish(
        self,
        match_id: str,
        update_payload: Dict,
        target_languages: List[str],
    ):
        """
        Fan-out translations concurrently for the provided languages.
        """
        if not target_languages:
            return

        tasks = [
            self._translate_and_publish_single_language(match_id, update_payload, lang)
            for lang in target_languages
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _translate_and_publish_single_language(
        self,
        match_id: str,
        update_payload: Dict,
        target_language: str,
    ):
        lines = update_payload.get("lines") or []
        if not lines:
            return

        try:
            # Use progressive translation to send latest balls first
            await self._translate_lines_progressive(
                match_id, 
                lines, 
                target_language,
                update_payload
            )
        except Exception as exc:
            logger.error(
                "Failed translating commentary for match %s [lang=%s]: %s",
                match_id,
                target_language,
                exc,
            )
            return

    async def translate_lines(
        self,
        match_id: str,
        lines: List[Dict],
        target_language: str,
    ) -> List[Dict]:
        """
        Translate arbitrary commentary lines without publishing.
        """
        if not lines:
            return []
        return await self._translate_lines(match_id, lines, target_language)
    
    async def _translate_lines_progressive(
        self,
        match_id: str,
        lines: List[Dict],
        target_language: str,
        update_payload: Dict,
    ):
        """
        Translate and publish commentary lines progressively.
        Latest balls are translated and published first, without waiting for older balls.
        """
        if not lines:
            return
        
        # Build initial translations dict from cache
        translations: Dict[str, str] = {}
        missing_entries: List[Dict[str, Any]] = []
        lines_by_id: Dict[str, Dict] = {}  # Map line_id to original line
        
        for line in lines:
            line_id_raw = line.get("id")
            if line_id_raw is None:
                continue
            line_id = str(line_id_raw)
            lines_by_id[line_id] = line
            
            cached = await self.redis_service.get_cached_translation(
                match_id, target_language, line_id
            )
            if cached:
                translations[line_id] = cached
            else:
                # Store timestamp with missing entry for sorting
                missing_entries.append({
                    "id": line_id,
                    "text": line.get("text", ""),
                    "timestamp": line.get("timestamp"),
                    "original_line": line
                })
        
        if not missing_entries:
            # All cached, publish everything at once
            translated_lines = self._build_translated_lines(lines, translations, target_language)
            localized_payload = dict(update_payload)
            localized_payload["language"] = target_language
            localized_payload["lines"] = translated_lines
            await self.redis_service.publish_commentary(
                match_id, localized_payload, language=target_language
            )
            return
        
        # PROGRESSIVE TRANSLATION: Send batches as they complete
        batch_size = 4
        overall_start = time.time()
        
        logger.info(
            "Starting PROGRESSIVE translation for %d entries in %d batches for match %s [lang=%s]",
            len(missing_entries),
            (len(missing_entries) + batch_size - 1) // batch_size,
            match_id,
            target_language
        )
        
        try:
            # Sort missing entries by timestamp (latest first)
            # Use a safe sorting key that handles None/missing timestamps
            def get_sort_key(entry):
                ts = entry.get("timestamp")
                if ts is None:
                    return datetime.min  # Put entries without timestamp at the end
                if isinstance(ts, str):
                    try:
                        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    except:
                        return datetime.min
                return ts if isinstance(ts, datetime) else datetime.min
            
            sorted_missing = sorted(
                missing_entries,
                key=get_sort_key,
                reverse=True  # Latest timestamp first
            )
            
            # Log sorting verification
            if sorted_missing:
                first_ts = sorted_missing[0].get("timestamp")
                last_ts = sorted_missing[-1].get("timestamp") if len(sorted_missing) > 1 else None
                logger.info(
                    "Sorted %d entries by timestamp (latest first) - First: %s, Last: %s",
                    len(sorted_missing),
                    first_ts,
                    last_ts
                )
            
            # Split into batches
            batches = [
                sorted_missing[i:i + batch_size]
                for i in range(0, len(sorted_missing), batch_size)
            ]
            
            # Create translation tasks
            async def translate_and_publish_batch(batch, batch_idx):
                batch_start = time.time()
                logger.info(
                    "Batch %d STARTED at %.3f (%d items, priority: %s) - match %s lang %s",
                    batch_idx,
                    batch_start,
                    len(batch),
                    "HIGH (latest)" if batch_idx == 0 else "normal",
                    match_id,
                    target_language
                )
                
                try:
                    # Extract only id and text for translation API
                    translation_input = [
                        {"id": item["id"], "text": item["text"]}
                        for item in batch
                    ]
                    
                    raw_translated = await self.translator.translate_batch(
                        translation_input,
                        source_lang=self.source_language,
                        target_lang=target_language,
                    )
                    
                    # Normalize results
                    batch_translated = self._normalize_translator_output(raw_translated)
                    
                    if not batch_translated:
                        logger.warning("Batch %d returned no translations", batch_idx)
                        return
                    
                    # Build translations dict for this batch
                    batch_translations = {}
                    for translated in batch_translated:
                        tid = str(translated.get("id")) if translated.get("id") is not None else None
                        text = translated.get("text") or ""
                        if tid:
                            batch_translations[tid] = text
                            # Cache it
                            try:
                                await self.redis_service.cache_translation(
                                    match_id, target_language, tid, text, ttl_seconds=self.cache_ttl
                                )
                            except Exception:
                                logger.exception("Failed caching translation for %s", tid)
                    
                    # Build lines for this batch only - maintain sorted order (latest first)
                    batch_lines = [item["original_line"] for item in batch if item["id"] in batch_translations]
                    translated_batch_lines = self._build_translated_lines(
                        batch_lines, batch_translations, target_language
                    )
                    
                    # Log order verification with IDs and timestamps
                    if translated_batch_lines:
                        line_info = [
                            f"ID:{line.get('id')[-8:] if line.get('id') else 'N/A'}, TS:{line.get('timestamp')}"
                            for line in translated_batch_lines[:2]  # Show first 2 for verification
                        ]
                        logger.info(
                            "Batch %d publishing order (latest first): %s",
                            batch_idx,
                            " | ".join(line_info)
                        )
                    
                    # PUBLISH THIS BATCH IMMEDIATELY
                    if translated_batch_lines:
                        batch_payload = dict(update_payload)
                        batch_payload["language"] = target_language
                        batch_payload["lines"] = translated_batch_lines
                        batch_payload["is_partial"] = len(batches) > 1  # Mark as partial if multiple batches
                        batch_payload["batch_index"] = batch_idx
                        
                        await self.redis_service.publish_commentary(
                            match_id, batch_payload, language=target_language
                        )
                        
                        batch_duration = time.time() - batch_start
                        logger.info(
                            "Batch %d COMPLETED and PUBLISHED in %.2f seconds (%d items) - match %s lang %s",
                            batch_idx,
                            batch_duration,
                            len(translated_batch_lines),
                            match_id,
                            target_language
                        )
                
                except Exception as batch_exc:
                    batch_duration = time.time() - batch_start
                    logger.exception(
                        "Batch %d FAILED after %.2f seconds: %s",
                        batch_idx,
                        batch_duration,
                        batch_exc
                    )
            
            # Run all batches concurrently, but they publish as soon as each completes
            gather_start = time.time()
            logger.info(
                "Starting concurrent execution of %d batches (progressive publishing) at %.3f",
                len(batches),
                gather_start
            )
            
            await asyncio.gather(
                *[translate_and_publish_batch(batch, idx) for idx, batch in enumerate(batches)],
                return_exceptions=True
            )
            
            overall_duration = time.time() - overall_start
            logger.info(
                "PROGRESSIVE translation completed: all batches processed in %.2f seconds (match %s, lang %s)",
                overall_duration,
                match_id,
                target_language
            )
            
        except Exception as exc:
            logger.exception(
                "Progressive translation failed for match %s lang %s: %s",
                match_id,
                target_language,
                exc
            )
    
    def _build_translated_lines(
        self,
        lines: List[Dict],
        translations: Dict[str, str],
        target_language: str
    ) -> List[Dict]:
        """Build translated lines from original lines and translations dict."""
        translated_lines = []
        for line in lines:
            line_id_raw = line.get("id")
            line_id = str(line_id_raw) if line_id_raw is not None else None
            translated_text = translations.get(line_id) if line_id else None
            new_line = dict(line)
            if translated_text:
                new_line["text"] = translated_text
            new_line["language"] = target_language
            translated_lines.append(new_line)
        return translated_lines

    async def _translate_lines(
        self,
        match_id: str,
        lines: List[Dict],
        target_language: str,
    ) -> List[Dict]:
        translations: Dict[str, str] = {}
        missing_entries: List[Dict[str, Any]] = []

        for line in lines:
            line_id_raw = line.get("id")
            if line_id_raw is None:
                continue
            line_id = str(line_id_raw)
            cached = await self.redis_service.get_cached_translation(
                match_id, target_language, line_id
            )
            if cached:
                translations[line_id] = cached
            else:
                # Store timestamp with missing entry for sorting
                missing_entries.append({
                    "id": line_id,
                    "text": line.get("text", ""),
                    "timestamp": line.get("timestamp"),
                    "original_index": len(missing_entries)  # Track original position
                })

        if missing_entries:
            # TRANSLATION BATCHING STRATEGY:
            # 1. Sort by timestamp (latest first) to prioritize recent commentary
            # 2. Split into batches of 4 balls each (5 batches for 20 balls)
            # 3. Process ALL batches concurrently (parallel Bedrock API calls)
            # 4. asyncio.gather preserves order, so batch 0 (latest) appears first in results
            # 5. This ensures latest commentary is translated first while maintaining speed
            batch_size = 4
            
            overall_start = time.time()
            logger.info(
                "Starting translation for %d entries in %d batches for match %s [lang=%s]",
                len(missing_entries),
                (len(missing_entries) + batch_size - 1) // batch_size,
                match_id,
                target_language
            )
            
            try:
                # Sort missing entries by timestamp (latest first) to prioritize recent commentary
                # Use same sorting key as progressive translation
                def get_sort_key(entry):
                    ts = entry.get("timestamp")
                    if ts is None:
                        return datetime.min
                    if isinstance(ts, str):
                        try:
                            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        except:
                            return datetime.min
                    return ts if isinstance(ts, datetime) else datetime.min
                
                sorted_missing = sorted(
                    missing_entries,
                    key=get_sort_key,
                    reverse=False  # Latest timestamp first
                )
                
                if sorted_missing:
                    first_ts = sorted_missing[0].get("timestamp")
                    last_ts = sorted_missing[-1].get("timestamp") if len(sorted_missing) > 1 else None
                    logger.info(
                        "Sorted %d entries by timestamp (latest first) - First: %s, Last: %s",
                        len(sorted_missing),
                        first_ts,
                        last_ts
                    )
                
                # Split sorted entries into batches of 4
                # Batch 0 will contain the 4 most recent entries
                batches = [
                    sorted_missing[i:i + batch_size]
                    for i in range(0, len(sorted_missing), batch_size)
                ]
                
                # Create translation tasks for all batches to run concurrently
                async def translate_single_batch(batch, batch_idx):
                    batch_start = time.time()
                    logger.info(
                        "Batch %d STARTED at %.3f (%d items, priority: %s) - match %s lang %s",
                        batch_idx,
                        batch_start,
                        len(batch),
                        "HIGH (latest)" if batch_idx == 0 else "normal",
                        match_id,
                        target_language
                    )
                    try:
                        # Extract only id and text for translation API
                        translation_input = [
                            {"id": item["id"], "text": item["text"]}
                            for item in batch
                        ]
                        
                        raw_translated = await self.translator.translate_batch(
                            translation_input,
                            source_lang=self.source_language,
                            target_lang=target_language,
                        )
                        
                        # Normalize and return results from this batch
                        batch_translated = self._normalize_translator_output(raw_translated)
                        batch_duration = time.time() - batch_start
                        logger.info(
                            "Batch %d COMPLETED in %.2f seconds (%d items) - match %s lang %s",
                            batch_idx,
                            batch_duration,
                            len(batch),
                            match_id,
                            target_language
                        )
                        return batch_translated if batch_translated else []
                    
                    except Exception as batch_exc:
                        batch_duration = time.time() - batch_start
                        logger.exception(
                            "Translator.translate_batch FAILED for match %s lang %s batch %d after %.2f seconds: %s",
                            match_id,
                            target_language,
                            batch_idx,
                            batch_duration,
                            batch_exc,
                        )
                        return []
                
                # Run all batch translations concurrently
                gather_start = time.time()
                logger.info(
                    "Starting concurrent execution of %d batches at %.3f",
                    len(batches),
                    gather_start
                )
                
                batch_results = await asyncio.gather(
                    *[translate_single_batch(batch, idx) for idx, batch in enumerate(batches)],
                    return_exceptions=False
                )
                
                gather_duration = time.time() - gather_start
                logger.info(
                    "All %d batches completed in %.2f seconds (concurrent execution)",
                    len(batches),
                    gather_duration
                )
                
                # Combine all successful batch results
                # asyncio.gather preserves order, so batch 0 (latest) comes first
                all_translated = []
                for batch_idx, batch_result in enumerate(batch_results):
                    if batch_result:
                        all_translated.extend(batch_result)
                        logger.debug(
                            "Combined batch %d: %d items (total so far: %d)",
                            batch_idx,
                            len(batch_result),
                            len(all_translated)
                        )
                
                translated_items = all_translated
                
                overall_duration = time.time() - overall_start
                logger.info(
                    "Translation completed: %d/%d items translated in %.2f seconds (match %s, lang %s)",
                    len(all_translated),
                    len(missing_entries),
                    overall_duration,
                    match_id,
                    target_language
                )
                
            except Exception as exc:
                logger.exception(
                    "Translation batching failed for match %s lang %s: %s",
                    "Translation batching failed for match %s lang %s: %s",
                    match_id,
                    target_language,
                    exc,
                )
                translated_items = []
                translated_items = []

            # If translator returned nothing or normalization failed, bail gracefully
            if not translated_items:
                logger.warning(
                    "No translated items returned for match %s lang %s (missing %d entries)",
                    match_id,
                    target_language,
                    len(missing_entries),
                )
            else:
                for translated in translated_items:
                    tid = str(translated.get("id")) if translated.get("id") is not None else None
                    text = translated.get("text") or ""
                    if not tid:
                        logger.debug("Skipping translated item without id: %r", translated)
                        continue
                    translations[tid] = text
                    # cache it
                    try:
                        await self.redis_service.cache_translation(
                            match_id,
                            target_language,
                            tid,
                            text,
                            ttl_seconds=self.cache_ttl,
                        )
                    except Exception:
                        # caching failure shouldn't break the pipeline
                        logger.exception(
                            "Failed caching translation for match %s lang %s id %s",
                            match_id,
                            target_language,
                            tid,
                        )

        # Build final translated lines using helper
        return self._build_translated_lines(lines, translations, target_language)

    def _normalize_translator_output(self, raw: Any) -> List[Dict[str, str]]:
        """
        Make sense of whatever the translator returned.

        Accepted/handled shapes:
        - already a list of dicts: [{"id": "...", "text": "..."}, ...]
        - a JSON string: '[{"id":"..","text":".."}, ...]'
        - a Bedrock-style wrapper string where content[0].text contains JSON
        - malformed but extractable text: try to regex out {"id":"...","text":"..."} objects

        Returns a list of dicts with keys "id" and "text" (may be empty list on failure).
        """
        if raw is None:
            return []

        # If translator already returned a Python list/dict, trust it if valid
        if isinstance(raw, (list, tuple)):
            items = []
            for it in raw:
                if isinstance(it, dict) and "id" in it and "text" in it:
                    items.append(
                        {
                            "id": it["id"],
                            "text": self._decode_unicode_escapes(it.get("text", "")),
                        }
                    )
            if items:
                return items

        if isinstance(raw, dict):
            # If it's a dict shaped response (maybe already parsed), try common places
            # e.g., {"content":[{"type":"text","text":"[...json...]"}]}
            try_candidates = []

            # try if dict itself is the list
            if "id" in raw and "text" in raw:
                return [
                    {
                        "id": raw["id"],
                        "text": self._decode_unicode_escapes(raw.get("text", "")),
                    }
                ]

            # common Bedrock wrapper
            content = raw.get("content")
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict) and "text" in first:
                    try_candidates.append(first["text"])

            # maybe there's a 'text' at top level
            if "text" in raw and isinstance(raw["text"], str):
                try_candidates.append(raw["text"])

            for candidate in try_candidates:
                parsed = self._try_parse_json_text(candidate)
                if parsed:
                    return parsed

        if isinstance(raw, str):
            # direct string: try parse as JSON first
            parsed = self._try_parse_json_text(raw)
            if parsed:
                return parsed

            # If not valid JSON, try to extract JSON payload inside common wrappers:
            # e.g. {"id": "...", "content":[{"type":"text","text":"[ ... ]"}]}
            # Look for a JSON array/object substring
            json_sub = self._extract_json_substring(raw)
            if json_sub:
                parsed = self._try_parse_json_text(json_sub)
                if parsed:
                    return parsed

            # Fallback: regex find {"id": "...", "text": "...."}
            items = []
            # DOTALL to allow newlines inside the text value
            pattern = re.compile(
                r'\{\s*"id"\s*:\s*"(?P<id>[^"]+)"\s*,\s*"text"\s*:\s*"(?P<text>.*?)"\s*\}',
                re.DOTALL,
            )
            for m in pattern.finditer(raw):
                gid = m.group("id")
                gtext = m.group("text")
                gtext_fixed = self._decode_unicode_escapes(gtext)
                items.append({"id": gid, "text": gtext_fixed})
            if items:
                return items

        # Nothing worked
        logger.debug("Unable to normalize translator output. raw type=%s content=%r", type(raw), raw)
        return []

    def _try_parse_json_text(self, text: str) -> Optional[List[Dict[str, str]]]:
        """
        Try to parse text as JSON. Accept either a JSON array of objects or a single object.
        Return list of dicts with keys id/text on success, or None on failure.
        """
        if not isinstance(text, str):
            return None
        s = text.strip()
        # strip Markdown backticks or stray quotes
        if s.startswith("```") and s.endswith("```"):
            s = s.strip("`")
        s = s.strip("`").strip()

        try:
            data = json.loads(s)
        except json.JSONDecodeError:
            return None

        if isinstance(data, list):
            out = []
            for item in data:
                if isinstance(item, dict) and "id" in item and "text" in item:
                    out.append({"id": item["id"], "text": item.get("text", "")})
            return out if out else None

        if isinstance(data, dict):
            # maybe dict -> wrap
            if "id" in data and "text" in data:
                return [
                    {
                        "id": data["id"],
                        "text": self._decode_unicode_escapes(data.get("text", "")),
                    }
                ]
            # maybe it's wrapper: {"content":[{"type":"text","text":"[...]"}]}
            content = data.get("content")
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict) and "text" in first:
                    return self._try_parse_json_text(first["text"])
        return None

    def _decode_unicode_escapes(self, text: str) -> str:
        """
        Convert sequences like '\\u0938' into real Unicode characters.
        Falls back to original text if decoding fails.
        """
        if not isinstance(text, str) or "\\u" not in text:
            return text
        try:
            return bytes(text, "utf-8").decode("unicode_escape")
        except Exception:
            return text

    def _extract_json_substring(self, text: str) -> Optional[str]:
        """
        Find a bracketed JSON substring (array or object) in `text`. Returns the substring or None.
        This is a best-effort helper: finds the first balanced [...] or {...} block.
        """
        if not isinstance(text, str):
            return None

        # try find first '[' ... matching ']' block
        start_idx = text.find("[")
        if start_idx != -1:
            # attempt to find matching closing bracket by scanning
            depth = 0
            for i in range(start_idx, len(text)):
                if text[i] == "[":
                    depth += 1
                elif text[i] == "]":
                    depth -= 1
                    if depth == 0:
                        return text[start_idx : i + 1]
        # fallback: try object {...}
        start_idx = text.find("{")
        if start_idx != -1:
            depth = 0
            for i in range(start_idx, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start_idx : i + 1]
        return None



