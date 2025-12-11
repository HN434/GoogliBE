"""
Video Analytics Service (Bedrock-powered)

Takes the aggregated pose metrics JSON for a single video (as stored in
`Video.metrics_jsonb`) and generates high-level coaching analytics that
match the structure needed by the frontend analysis page.

This service is intentionally thin: it builds a prompt that includes the
raw metrics JSON and asks Bedrock to respond with a *single* JSON object
containing key observations, improvement areas, suggested drills, and
an explanation section.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class VideoAnalyticsService:
    """Bedrock client wrapper for generating video-level batting analytics."""

    model_id: str
    region: str
    temperature: float = 0.4
    top_p: float = 0.9
    max_tokens: int = 2048

    def __post_init__(self) -> None:
        self.client = boto3.client("bedrock-runtime", region_name=self.region)
    
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

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate_analytics(self, video_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate high-level analytics JSON for a given video.

        Args:
            video_id: Video UUID as string
            metrics: Aggregated pose metrics (the exact structure produced by
                     `compute_video_pose_metrics`)

        Returns:
            Parsed JSON object produced by Bedrock. If the response cannot be
            parsed as JSON, this will raise a JSONDecodeError so callers can
            handle the failure.
        """
        prompt = self._build_prompt(video_id, metrics)
        raw_text = self._invoke_bedrock(prompt)
        return self._parse_json_response(raw_text)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_prompt(self, video_id: str, metrics: Dict[str, Any]) -> str:
        metrics_json = json.dumps(metrics, ensure_ascii=False)
        
        # Build interpretive context for metrics
        context = self._build_interpretive_context(metrics)
        
        # Extract shot classification if available
        shot_info = ""
        if "shot_classification" in metrics:
            shot_class = metrics["shot_classification"]
            primary_shot = shot_class.get("primary_shot")
            confidence = shot_class.get("confidence", 0.0)
            alternatives = shot_class.get("alternative_shots", [])
            
            if primary_shot:
                alt_text = ""
                if alternatives:
                    alt_shots = ', '.join([a.get('shot_type', '') for a in alternatives[:2]])
                    if alt_shots:
                        alt_text = f" Alternative possibilities: {alt_shots}."
                
                shot_info = (
                    f"\n\n=== SHOT TYPE ANALYSIS ===\n"
                    f"Primary shot detected: {primary_shot.replace('_', ' ').title()} "
                    f"(confidence: {confidence:.1%}){alt_text}\n"
                    f"Tailor your coaching advice specifically for this shot type.\n"
                )
        
        return (
            "You are an expert Level 3 cricket batting coach with 20+ years of experience "
            "coaching players from club to international level. You specialize in turning complex "
            "biomechanics into simple, encouraging coaching cues that any player can follow.\n\n"
            
            "=== LANGUAGE REQUIREMENT ===\n"
            "Use British English (UK) spelling and terminology throughout your response. "
            "Examples: 'colour' not 'color', 'realise' not 'realize', 'centre' not 'center', "
            "'optimise' not 'optimize', 'analyse' not 'analyze', 'favour' not 'favor'.\n\n"
            
            "=== YOUR TASK ===\n"
            "Analyze the following cricket batting video metrics and provide coaching feedback in plain, everyday language. "
            "Use the data to inform your advice, but explain it without technical jargon, degrees, or angles unless absolutely necessary. "
            "Focus on what the player should feel and do rather than quoting numbers.\n"
            
            + shot_info +
            
            "\n=== AVAILABLE METRICS ===\n"
            + context +
            
            "\n\n=== SCORING APPROACH ===\n"
            "Use the normalized measurements to understand the swing, but keep the explanation friendly and practical. "
            "Scores (0-10) are fine, yet the narrative should be light on numbers and heavy on clear guidance.\n\n"
            
            "**How to translate metrics into plain coaching:**\n"
            "• Turn stride, backlift, and weight-transfer data into simple body cues (e.g., \"step smoothly forward,\" \"lift the bat earlier\")\n"
            "• Use timing and consistency signals to comment on rhythm and repeatability, not raw figures\n"
            "• Mention a number only when it helps the player prioritize (e.g., calling out a very low or very high score)\n"
            "• Avoid degrees, angles, ratios, and fine-grained measurements unless essential for clarity\n\n"
            
            "**Scoring philosophy (kept simple):**\n"
            "- Apply your expert judgment to rate technique on a 0-10 scale\n"
            "- Celebrate what already works before suggesting fixes\n"
            "- Keep language approachable and confidence-building\n\n"
            
            "=== OUTPUT FORMAT ===\n"
            "Return coaching analytics as a SINGLE valid JSON object with this EXACT structure:\n\n"
            "{\n"
            '  "video_id": "<echo the input video_id>",\n'
            '  "summary": {\n'
            '    "overall_score": <number 0-10 based on overall technique>,\n'
            '    "skill_level": "<beginner|intermediate|advanced>",\n'
            '    "headline": "<one clear sentence summarizing the batting technique in plain language>"\n'
            "  },\n"
            '  "key_observations": [\n'
            '    {\n'
            '      "title": "Stance and Balance",\n'
            '      "score": <0-10>,\n'
            '      "description": "<2-3 sentences in everyday language about stability and setup; avoid angles/degree talk>"\n'
            "    },\n"
            '    {\n'
            '      "title": "Backlift and Bat Path",\n'
            '      "score": <0-10>,\n'
            '      "description": "<2-3 sentences about bat pickup and path using simple cues; no angle jargon>"\n'
            "    },\n"
            '    {\n'
            '      "title": "Footwork and Weight Transfer",\n'
            '      "score": <0-10>,\n'
            '      "description": "<2-3 sentences on movement and balance in plain terms; minimal numbers>"\n'
            "    },\n"
            '    {\n'
            '      "title": "Swing Path and Follow Through",\n'
            '      "score": <0-10>,\n'
            '      "description": "<2-3 sentences about swing flow and finish using relatable language>"\n'
            "    }\n"
            "  ],\n"
            '  "improvement_areas": [\n'
            '    {\n'
            '      "title": "<specific technical aspect>",\n'
            '      "detail": "<2-3 sentences with actionable, easy-to-follow advice; avoid degree/angle references>",\n'
            '      "priority": "<high|medium|low>"\n'
            "    }\n"
            '    // Include 2-4 improvement areas, prioritized by impact\n'
            "  ],\n"
            '  "suggested_drills": [\n'
            '    {\n'
            '      "name": "<drill name>",\n'
            '      "focus_area": "<bat_control|balance|footwork|alignment|weight_transfer>",\n'
            '      "description": "<2-3 sentences explaining the drill in simple terms and the feeling it builds>"\n'
            "    }\n"
            '    // Include 2-4 specific drills targeted at the identified weaknesses\n'
            "  ],\n"
            '  "explanation": {\n'
            '    "long_form": "<3-4 sentences giving an encouraging recap in layman’s terms; call out only the most important numbers if needed>",\n'
            '    "notes": ["<actionable tip 1 in plain language>", "<actionable tip 2>", "<actionable tip 3>"]\n'
            "  }\n"
            "}\n\n"
            
            "=== CRITICAL RULES ===\n"
            "1. Use British English (UK) spelling and terminology throughout—this is mandatory.\n"
            "2. Use the metrics to inform guidance but explain them in everyday language; avoid degrees, angles, and raw ratios unless essential.\n"
            "3. Keep numbers to the required scores and the rare case where a figure truly clarifies priority.\n"
            "4. Be constructive, encouraging, and balanced—acknowledge strengths while identifying growth areas.\n"
            "5. Tailor advice to the detected shot type and player's apparent skill level.\n"
            "6. Return ONLY valid JSON - no markdown, no backticks, no explanatory text outside the JSON.\n"
            "7. Make improvement areas and drills specific, actionable, and easy to follow without technical jargon.\n\n"
            
            f"=== INPUT DATA ===\n"
            f"Video ID: {video_id}\n\n"
            f"Pose Metrics (JSON):\n{metrics_json}\n"
        )
    
    def _build_interpretive_context(self, metrics: Dict[str, Any]) -> str:
        """Build human-readable context explaining what the metrics mean."""
        stance = metrics.get("stance_metrics", {})
        backlift = metrics.get("backlift_metrics", {})
        swing = metrics.get("swing_metrics", {})
        footwork = metrics.get("footwork_metrics", {})
        weight = metrics.get("weight_transfer_metrics", {})
        biomech = metrics.get("biomechanical_metrics", {})
        bat_metrics = metrics.get("bat_metrics", {})
        frame_quality = metrics.get("frame_quality", {})
        
        context_parts = []
        
        context_parts.append("\n🎯 KEY NORMALIZED METRICS (scale-invariant, camera zoom independent):")
        context_parts.append("="*60)
        
        # Data quality
        mean_conf = frame_quality.get("mean_keypoint_confidence", 0)
        if mean_conf:
            context_parts.append(f"• Data quality: {mean_conf:.2f} confidence (1.0 = perfect tracking)")
        
        # NORMALIZED FOOTWORK (scale-invariant!)
        stride_norm = footwork.get("front_foot_stride_normalized", 0)
        if stride_norm:
            context_parts.append(f"• Front foot stride (normalized): {stride_norm:.2f} × stance width (measures forward movement)")
        
        footwork_timing = footwork.get("footwork_timing_score", 0)
        if footwork_timing:
            context_parts.append(f"• Footwork timing: {footwork_timing:.2f} score (0.9 = ideal timing, <0.6 = too early/late)")
        
        # NORMALIZED BACKLIFT (scale-invariant!)
        backlift_norm = backlift.get("backlift_height_normalized", 0)
        if backlift_norm:
            context_parts.append(f"• Backlift height (normalized): {backlift_norm:.2f} × torso length (measures bat pickup)")
        
        consistency = backlift.get("backlift_consistency_score", 0)
        bat_angle = backlift.get("bat_angle_avg_deg", 0)
        if consistency and bat_angle:
            context_parts.append(f"• Backlift consistency: {consistency:.2f}, angle: {bat_angle:.1f}° (higher consistency = repeatable)")
        
        # NORMALIZED WEIGHT TRANSFER (scale-invariant!)
        com_shift_norm = weight.get("com_shift_normalized", 0)
        if com_shift_norm:
            context_parts.append(f"• Weight transfer (normalized): {com_shift_norm:.2f} × torso length (positive = into shot)")
        
        wt_velocity = weight.get("weight_transfer_velocity", 0)
        wt_smoothness = weight.get("weight_transfer_smoothness", 0)
        if wt_velocity or wt_smoothness:
            context_parts.append(f"• Weight transfer dynamics: velocity={wt_velocity:.3f}, smoothness={wt_smoothness:.2f} (higher = better)")
        
        balance_stability = weight.get("balance_stability_score", 0)
        balance_ratio = weight.get("left_right_balance_ratio", 0.5)
        if balance_stability or balance_ratio:
            context_parts.append(f"• Balance: stability={balance_stability:.2f}, L/R ratio={balance_ratio:.2f} (0.5 = centered)")
        
        # STANCE & POSTURE
        head_stability = stance.get("head_stability_score", 0)
        if head_stability:
            context_parts.append(f"• Head stability: {head_stability:.2f} (higher = steadier head position)")
        
        shoulder_tilt = stance.get("shoulder_tilt_avg_deg", 0)
        spine_lean = stance.get("spine_lean_avg_deg", 0)
        if shoulder_tilt or spine_lean:
            context_parts.append(f"• Posture: shoulder tilt={shoulder_tilt:.1f}°, spine lean={spine_lean:.1f}°")
        
        # SWING PATH
        smoothness = swing.get("downswing_smoothness", 0)
        if smoothness:
            context_parts.append(f"• Downswing smoothness: {smoothness:.2f} (higher = more fluid, controlled)")
        
        context_parts.append("\n⚡ ADVANCED BIOMECHANICAL METRICS:")
        context_parts.append("="*60)
        
        # Hip-shoulder separation (X-factor)
        x_factor_avg = biomech.get("hip_shoulder_separation_avg_deg", 0)
        x_factor_max = biomech.get("hip_shoulder_separation_max_deg", 0)
        if x_factor_avg or x_factor_max:
            context_parts.append(f"• X-Factor (hip-shoulder separation): avg={x_factor_avg:.1f}°, max={x_factor_max:.1f}° (measures coil/rotation)")
        
        # Kinetic chain efficiency
        kinetic_chain = biomech.get("kinetic_chain_efficiency", 0)
        if kinetic_chain:
            context_parts.append(f"• Kinetic chain efficiency: {kinetic_chain:.2f} (0.9 = excellent, hips lead shoulders)")
        
        # Rotation metrics
        rotation_range = biomech.get("upper_body_rotation_range_deg", 0)
        rotation_timing = biomech.get("rotation_timing_score", 0)
        if rotation_range or rotation_timing:
            context_parts.append(f"• Rotation: range={rotation_range:.1f}°, timing={rotation_timing:.2f} (measures body rotation)")
        
        # BAT DETECTION METRICS (if available)
        if bat_metrics:
            context_parts.append("\n🏏 BAT DETECTION METRICS:")
            context_parts.append("="*60)
            
            bat_detected = bat_metrics.get("bat_detected", False)
            if bat_detected:
                detection_rate = bat_metrics.get("detection_rate", 0)
                avg_conf = bat_metrics.get("avg_confidence", 0)
                context_parts.append(f"• Bat detected: YES ({detection_rate:.1%} of frames, avg confidence={avg_conf:.2f})")
                
                bat_angle_avg = bat_metrics.get("bat_angle_avg_deg", 0)
                bat_angle_range = bat_metrics.get("bat_angle_range_deg", 0)
                if bat_angle_avg:
                    context_parts.append(f"• Bat angle: avg={bat_angle_avg:.1f}°, range={bat_angle_range:.1f}° (rotation during shot)")
                
                bat_movement = bat_metrics.get("bat_movement_velocity", 0)
                if bat_movement:
                    context_parts.append(f"• Bat movement velocity: {bat_movement:.1f} px/frame (bat speed)")
            else:
                context_parts.append("• Bat detected: NO (bat not visible or detection disabled)")
        
        context_parts.append("\n💡 ANALYSIS GUIDANCE:")
        context_parts.append("="*60)
        context_parts.append("• Focus on NORMALIZED metrics (stride, backlift, weight transfer) - these are accurate")
        context_parts.append("• Pixel values (px) should be interpreted carefully - they depend on camera distance")
        context_parts.append("• Biomechanical metrics (X-factor, kinetic chain) show technique sophistication")
        context_parts.append("• Timing scores (footwork, rotation) indicate coordination and sequencing")
        context_parts.append("• Bat detection metrics (if available) provide direct bat tracking data")
        
        return "\n".join(context_parts) if context_parts else "Limited interpretive context available from metrics."

    def _invoke_bedrock(self, prompt: str) -> str:
        """Call Bedrock and return the model's raw textual output."""
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

        try:
            response = self.client.invoke_model(modelId=self.model_id, body=body)
        except (BotoCoreError, ClientError):
            logger.exception("Bedrock invoke_model for video analytics failed")
            raise

        streaming_body = response.get("body")
        if hasattr(streaming_body, "iter_chunks"):
            chunks = []
            for chunk in streaming_body.iter_chunks():
                if chunk:
                    chunks.append(chunk)
            raw_body = b"".join(chunks)
        else:
            raw_body = streaming_body.read() if streaming_body else b""

        if isinstance(raw_body, (bytes, bytearray)):
            raw_body = raw_body.decode("utf-8", errors="ignore")

        logger.debug("Bedrock analytics raw response (truncated): %s", raw_body[:500])
        
        # For Amazon Nova models, extract text from JSON response if needed
        if self._is_amazon_nova_model():
            try:
                parsed = json.loads(raw_body)
                # Nova models return text in output.message.content or output.text
                if isinstance(parsed, dict):
                    output = parsed.get("output")
                    if isinstance(output, dict):
                        # Check for output.message.content (Nova format)
                        message = output.get("message")
                        if isinstance(message, dict):
                            content = message.get("content")
                            if isinstance(content, str):
                                return content
                            elif isinstance(content, list) and content:
                                first = content[0]
                                if isinstance(first, dict) and "text" in first:
                                    return first["text"]
                        # Check for output.text directly
                        if "text" in output and isinstance(output["text"], str):
                            return output["text"]
                    elif isinstance(output, str):
                        return output
                    # Check for content as string directly
                    if "content" in parsed and isinstance(parsed["content"], str):
                        return parsed["content"]
            except json.JSONDecodeError:
                # If not JSON, return as-is
                pass
        
        return raw_body.strip()

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """
        Try hard to parse the model output as JSON.

        We expect the model to return one JSON object, but we defensively:
        - strip markdown fences/backticks
        - if parsing fails, try to extract the first {...} block
        """
        cleaned = text.strip()
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned.strip("`").strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Best-effort: find first balanced {...} block
            start = cleaned.find("{")
            if start == -1:
                raise
            depth = 0
            for i in range(start, len(cleaned)):
                if cleaned[i] == "{":
                    depth += 1
                elif cleaned[i] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = cleaned[start : i + 1]
                        return json.loads(candidate)
            # If we get here, re-raise the original error
            raise


_video_analytics_instance: Optional[VideoAnalyticsService] = None


def get_video_analytics_service() -> VideoAnalyticsService:
    """
    Get or create a singleton VideoAnalyticsService.
    Requires BEDROCK_REGION and BEDROCK_MODEL_ID to be configured.
    """
    global _video_analytics_instance

    if _video_analytics_instance is None:
        if not settings.BEDROCK_REGION or not settings.BEDROCK_MODEL_ID:
            raise RuntimeError(
                "Bedrock settings not configured. "
                "Set BEDROCK_REGION and BEDROCK_MODEL_ID in the environment."
            )

        _video_analytics_instance = VideoAnalyticsService(
            model_id=settings.BEDROCK_MODEL_ID,
            region=settings.BEDROCK_REGION,
            temperature=settings.BEDROCK_TEMPERATURE,
            top_p=settings.BEDROCK_TOP_P,
            max_tokens=settings.BEDROCK_MAX_TOKENS,
        )

    return _video_analytics_instance



