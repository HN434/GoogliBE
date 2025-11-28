"""
Translation services for live commentary.
Provides Bedrock-powered translation plus orchestration helpers that
cache translations and publish localized payloads.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Dict, List, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from services.redis_service import RedisService

logger = logging.getLogger(__name__)


class BedrockTranslationService:
    """Thin wrapper around AWS Bedrock runtime for batch translations."""

    def __init__(
        self,
        model_id: str,
        region: str,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 10240,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def _build_prompt(self, items: List[Dict[str, str]], source_lang: str, target_lang: str) -> str:
        payload = json.dumps(items, ensure_ascii=False)
        return (
            "You are a professional cricket commentator and translation engine. "
            f"Translate each entry from {source_lang} to {target_lang}. "
            "Preserve player names and cricket terms in English, keep tone conversational, "
            "and avoid transliteration for numbers. "
            "Return ONLY valid JSON in this format: "
            '[{"id": "<original id>", "text": "<translated text>"}]. '
            f"Input entries: {payload}"
        )

    def _invoke_bedrock(self, prompt: str) -> str:
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                ],
            }
        )
        response = self.client.invoke_model(modelId=self.model_id, body=body)
        payload = json.loads(response["body"].read())
        content = payload.get("output") or payload.get("content") or payload.get("message")
        if isinstance(content, list):
            # Anthropic returns {"output": [{"content": [{"text": "..."}]}]}
            content = content[0] if content else {}
        if isinstance(content, dict):
            text_blocks = content.get("content") or content.get("text") or []
            if isinstance(text_blocks, list):
                text = text_blocks[0].get("text") if text_blocks else ""
            else:
                text = text_blocks
        else:
            text = ""
        if not text and "content" in payload:
            # Claude responses sometimes live under payload["content"]
            chunks = payload["content"]
            if isinstance(chunks, list) and chunks:
                text = chunks[0].get("text", "")
        return text.strip()

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

        try:
            raw_response = await asyncio.to_thread(self._invoke_bedrock, prompt)
            translated_entries = json.loads(raw_response)
            if not isinstance(translated_entries, list):
                raise ValueError("Translation response is not a list")
            normalized = []
            for entry in translated_entries:
                if not isinstance(entry, dict):
                    continue
                entry_id = entry.get("id")
                text = entry.get("text")
                if entry_id and isinstance(text, str):
                    normalized.append({"id": entry_id, "text": text})
            return normalized
        except (BotoCoreError, ClientError, ValueError, json.JSONDecodeError) as exc:
            logger.error("Bedrock translation failed: %s", exc, exc_info=True)
            raise


class CommentaryTranslationService:
    """
    Coordinates translation for commentary snapshots:
    - reads cached translations from Redis
    - invokes Bedrock only for missing lines
    - publishes localized updates to language-specific channels
    """

    def __init__(
        self,
        redis_service: RedisService,
        translator: BedrockTranslationService,
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

        translations: Dict[str, str] = {}
        missing_entries: List[Dict[str, str]] = []

        for line in lines:
            line_id = line.get("id")
            if not line_id:
                continue
            cached = await self.redis_service.get_cached_translation(
                match_id, target_language, line_id
            )
            if cached:
                translations[line_id] = cached
            else:
                missing_entries.append({"id": line_id, "text": line.get("text", "")})

        if missing_entries:
            try:
                translated_items = await self.translator.translate_batch(
                    missing_entries,
                    source_lang=self.source_language,
                    target_lang=target_language,
                )
                for translated in translated_items:
                    translations[translated["id"]] = translated["text"]
                    await self.redis_service.cache_translation(
                        match_id,
                        target_language,
                        translated["id"],
                        translated["text"],
                        ttl_seconds=self.cache_ttl,
                    )
            except Exception as exc:
                logger.error(
                    "Failed translating commentary for match %s [lang=%s]: %s",
                    match_id,
                    target_language,
                    exc,
                )
                return

        translated_lines = []
        for line in lines:
            line_id = line.get("id")
            translated_text = translations.get(line_id) if line_id else None
            new_line = dict(line)
            if translated_text:
                new_line["text"] = translated_text
            new_line["language"] = target_language
            translated_lines.append(new_line)

        localized_payload = dict(update_payload)
        localized_payload["language"] = target_language
        localized_payload["lines"] = translated_lines

        await self.redis_service.publish_commentary(
            match_id, localized_payload, language=target_language
        )


