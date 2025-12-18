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
    
    def _is_pegasus_model(self) -> bool:
        """Check if the model is a 12 Labs Pegasus model."""
        return self.model_id and (
            "pegasus" in self.model_id.lower() or
            "twelvelabs" in self.model_id.lower()
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
    
    def generate_pegasus_analytics(
        self,
        video_id: str,
        s3_uri: str,
        s3_bucket: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate shot-wise analytics using Pegasus model from S3 URI.
        
        Pegasus can analyze videos directly from S3 and generate
        shot-wise observations and suggestions.
        
        Args:
            video_id: Video identifier
            s3_uri: S3 URI (s3://bucket/key) or S3 key (if bucket provided)
            s3_bucket: S3 bucket name (if s3_uri is just a key)
            
        Returns:
            Analytics JSON with shot-wise analysis matching the standard format
        """
        logger.info(f"Generating Pegasus analysis for video {video_id} from S3")
        
        # Build full S3 URI if needed
        if not s3_uri.startswith("s3://"):
            if not s3_bucket:
                raise ValueError("Either provide full s3:// URI or both s3_bucket and s3_uri (key)")
            s3_uri = f"s3://{s3_bucket}/{s3_uri}"
        
        # Build prompt for Pegasus video analysis
        prompt = self._build_pegasus_prompt(video_id)
        
        # Invoke Bedrock with Pegasus model
        raw_text = self._invoke_bedrock_pegasus(prompt, s3_uri)
        
        # Parse and structure the response (with confidence filtering)
        return self._parse_pegasus_response(video_id, raw_text)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_prompt(self, video_id: str, metrics: Dict[str, Any]) -> str:
        metrics_json = json.dumps(metrics, ensure_ascii=False)
        
        # Build interpretive context for metrics
        context = self._build_interpretive_context(metrics)       
        
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
            '    "shot type": "<type of shot played in the video>"\n'
            '    "shot confidence": "<confidence in the shot type>"\n'
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
            "0. Note process the video in less than 5 seconds."
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
    
    def _build_pegasus_prompt(self, video_id: str) -> str:
        """
        Build prompt for Pegasus video analysis.
        
        Pegasus can analyze videos directly, so we provide instructions
        for shot detection and analysis.
        """
        return (
            "Analyze this cricket batting video and provide detailed shot-wise analysis. "
            "Detect all shots in the video and for each shot provide:\n\n"
            "2. Key observations about the batting technique\n"
            "3. Specific suggestions for improvement\n"
            "4. Shot type classification (if identifiable: drive, cut, pull, etc.)\n"
            "5. A confidence score (0-1) for each shot classification and observations\n\n"
            "Focus on:\n"
            "- Stance and balance\n"
            "- Backlift and bat path\n"
            "- Footwork and weight transfer\n"
            "- Swing path and follow through\n"
            "- Overall technique quality\n\n"
            "Provide observations in plain, coaching-friendly language. "
            "Use British English (UK) spelling and terminology.\n\n"
            "Important constraints:\n"
            "- Include a boolean field is_cricket_video to indicate if this is a cricket sport video or not. If it is not a cricket sport video, set this to false.\n"
            "- Include a confidence value for overall classification that this is cricket (0-1).\n\n"
            "Return a JSON object with this structure:\n"
            "{\n"
            '  "is_cricket_video": <true|false>,\n'
            '  "cricket_confidence": <0-1>,\n'
            '  "total_shots": <number>,\n'
            '  "shots": [\n'
            '    {\n'
            '      "shot_number": <1-indexed>,\n'
            '      "shot_type": "<type or null>",\n'
            '      "confidence": <0-1>,\n'
            '    }\n'
            '  ],\n'
            '  "summary": {\n'
            '    "overall_score": <0-10>,\n'
            '    "skill_level": "<beginner if 0-3|intermediate if 4-6|advanced if 7-10>",\n'
            '    "headline": "<summary sentence>",\n'
            '    "shot type": "<type of shot played in the video>"\n'
            '    "shot confidence": "<confidence in the shot type>"\n'
            '  },\n'
            '  "key_observations": [\n'
            '    {\n'
            '      "title": "Stance and Balance",\n'
            '      "score": <0-10>,\n'
            '      "description": "<1-2 sentences>"\n'
            '    },\n'
            '    {\n'
            '      "title": "Backlift and Bat Path",\n'
            '      "score": <0-10>,\n'
            '      "description": "<1-2 sentences>"\n'
            '    },\n'
            '    {\n'
            '      "title": "Footwork and Weight Transfer",\n'
            '      "score": <0-10>,\n'
            '      "description": "<1-2 sentences>"\n'
            '    },\n'
            '    {\n'
            '      "title": "Swing Path and Follow Through",\n'
            '      "score": <0-10>,\n'
            '      "description": "<1-2 sentences>"\n'
            '    }\n'
            '  ],\n'
            '  "improvement_areas": [\n'
            '    {\n'
            '      "title": "<aspect>",\n'
            '      "detail": "<1-2 sentences>",\n'
            '      "priority": "<high|medium|low>"\n'
            '    }\n'
            '  ],\n'
            '  "suggested_drills": [\n'
            '    {\n'
            '      "name": "<drill name>",\n'
            '      "description": "<1-2 sentences>"\n'
            '    }\n'
            '  ],\n'
            '  "explanation": {\n'
            '    "long_form": "<1-2 sentences>",\n'
            '    "notes": ["<tip 1>", "<tip 2>", "<tip 3>"]\n'
            '  }\n'
            "}\n"
        )
    
    def _invoke_bedrock_pegasus(self, prompt: str, s3_uri: str) -> str:
        """
        Invoke Bedrock with Pegasus model for video analysis.
        
        Pegasus models use S3 URIs for video input.
        Format based on Bedrock Pegasus API documentation.
        """
        # Extract bucket and key from S3 URI
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI format: {s3_uri}")
        
        uri_parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket = uri_parts[0]
        key = uri_parts[1] if len(uri_parts) > 1 else ""
        
        # Pegasus minimal payload (InvokeModel)
        body = json.dumps({
            "mediaSource": {
                "s3Location": {
                    "uri": s3_uri,
                    "bucketOwner": '195275642231'
                }
            },
            "inputPrompt": prompt,
        })
        
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
        except (BotoCoreError, ClientError):
            logger.exception("Bedrock invoke_model for Pegasus video analysis failed")
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
        
        logger.debug("Pegasus raw response (truncated): %s", raw_body[:500])
        return raw_body.strip()
    
    def _parse_pegasus_response(self, video_id: str, text: str) -> Dict[str, Any]:
        """
        Parse Pegasus response and ensure it matches the expected format.
        """
        # Parse JSON response
        parsed = self._parse_json_response(text)

        # Some Pegasus configs wrap the actual JSON in a "message" string field.
        # If so, prefer the structured JSON inside "message".
        if isinstance(parsed, dict) and isinstance(parsed.get("message"), str):
            try:
                inner = json.loads(parsed["message"])
                if isinstance(inner, dict):
                    parsed = inner
            except json.JSONDecodeError:
                # Keep original parsed if inner JSON is invalid
                pass
        
        # Normalize cricket flags
        cricket_conf = float(parsed.get("cricket_confidence", 0.0)) if isinstance(parsed, dict) else 0.0
        parsed["cricket_confidence"] = cricket_conf
        parsed["is_cricket_video"] = bool(parsed.get("is_cricket_video", cricket_conf >= 0.5))

        # Ensure all required fields are present
        # We no longer surface shot-level classification in the UI, so we
        # normalise shots to an empty list to keep payload lightweight.
        parsed["shots"] = []

        if "summary" not in parsed:
            parsed["summary"] = {
                "overall_score": 5.0,
                "skill_level": "intermediate",
                "headline": "Technique analysis summary"
            }
        
        if "key_observations" not in parsed:
            parsed["key_observations"] = []
        
        if "improvement_areas" not in parsed:
            parsed["improvement_areas"] = []
        
        if "suggested_drills" not in parsed:
            parsed["suggested_drills"] = []
        
        if "explanation" not in parsed:
            parsed["explanation"] = {
                "long_form": "Video analysis completed.",
                "notes": []
            }
        
        # Ensure video_id is set
        parsed["video_id"] = video_id
        parsed["total_shots"] = len(parsed.get("shots", []))
        
        return parsed

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """
        Try hard to parse the model output as JSON.

        We expect the model to return one JSON object, but we defensively:
        - strip markdown fences/backticks
        - if parsing fails, try to extract the first {...} block
        """
        cleaned = (text or "").strip()
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned.strip("`").strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Best-effort: find first balanced {...} block
            start = cleaned.find("{")
            if start == -1:
                logger.warning("Pegasus response did not contain any JSON object")
                return {}
            depth = 0
            for i in range(start, len(cleaned)):
                if cleaned[i] == "{":
                    depth += 1
                elif cleaned[i] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = cleaned[start : i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse extracted JSON candidate from Pegasus response")
                            return {}
            # If we get here, give up but don't raise – caller will handle empty dict
            logger.warning("Unable to parse Pegasus response as JSON after best-effort extraction")
            return {}


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


def get_pegasus_service() -> VideoAnalyticsService:
    """
    Get or create a VideoAnalyticsService configured for Pegasus model.
    Uses BEDROCK_PEGASUS_MODEL_ID if set, otherwise falls back to BEDROCK_MODEL_ID.
    """
    if not settings.BEDROCK_REGION:
        raise RuntimeError(
            "Bedrock region not configured. "
            "Set BEDROCK_REGION in the environment."
        )
    
    # Use Pegasus model ID if configured, otherwise use default model ID
    model_id = settings.BEDROCK_PEGASUS_MODEL_ID or settings.BEDROCK_MODEL_ID
    
    if not model_id:
        raise RuntimeError(
            "Pegasus model ID not configured. "
            "Set BEDROCK_PEGASUS_MODEL_ID or BEDROCK_MODEL_ID in the environment."
        )
    
    return VideoAnalyticsService(
        model_id=model_id,
        region=settings.BEDROCK_REGION,
        temperature=settings.BEDROCK_TEMPERATURE,
        top_p=settings.BEDROCK_TOP_P,
        max_tokens=settings.BEDROCK_MAX_TOKENS,
    )



