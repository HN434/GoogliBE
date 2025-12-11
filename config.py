"""
Configuration settings for Cricket Pose Analysis Backend
"""
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file
    """

    # ===== Server Settings =====
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True

    # ===== File Storage =====
    UPLOAD_DIR: str = "../uploads"
    PROCESSED_DIR: str = "../uploads/processed"
    MAX_UPLOAD_SIZE_MB: int = 500

    # ===== GPU & Performance =====
    USE_GPU: bool = True
    DEVICE: str = "cuda"  # cuda, cpu, or mps (for Mac M1/M2)
    BATCH_SIZE: int = 16
    NUM_WORKERS: int = 4

    # ===== Pose Detection Models =====
    POSE_MODEL: str = "yolov8x-pose.pt"  # yolov8n/s/m/l/x-pose.pt
    POSE_CONFIDENCE_THRESHOLD: float = 0.5
    POSE_IOU_THRESHOLD: float = 0.45
    USE_MEDIAPIPE_FALLBACK: bool = True
    MEDIAPIPE_MODEL_COMPLEXITY: int = 2  # 0, 1, or 2

    # ===== RTMPose Inference =====
    RTMPOSE_CONFIG_PATH: str = os.getenv(
        "RTMPOSE_CONFIG_PATH",
        "models/rtmpose-s/rtmpose-s_256x192.py",
    )
    RTMPOSE_CHECKPOINT_PATH: str = os.getenv(
        "RTMPOSE_CHECKPOINT_PATH",
        "models/rtmpose-s/rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.pth",
    )
    RTMPOSE_DEVICE: str = os.getenv("RTMPOSE_DEVICE", "cuda")

    # ===== RF-DETR Bat Detection =====
    BAT_DETECTION_ENABLED: bool = os.getenv("BAT_DETECTION_ENABLED", "true").lower() == "true"
    RTDETR_MODEL_PATH: str = os.getenv(
        "RTDETR_MODEL_PATH",
        "checkpoint_best_regular.pth",  # Fine-tuned RF-DETR checkpoint for bat detection
    )
    BAT_CONFIDENCE_THRESHOLD: float = float(os.getenv("BAT_CONFIDENCE_THRESHOLD", "0.5"))
    BAT_NMS_THRESHOLD: float = float(os.getenv("BAT_NMS_THRESHOLD", "0.45"))
    # Size-based filtering (bats are smaller than people in cricket videos)
    # Note: Bats can be vertical (tall) or horizontal (wide) depending on angle
    BAT_MAX_HEIGHT: int = int(os.getenv("BAT_MAX_HEIGHT", "300"))  # Max height in pixels for bat bbox
    BAT_MAX_WIDTH: int = int(os.getenv("BAT_MAX_WIDTH", "150"))  # Max width in pixels for bat bbox  
    BAT_MIN_HEIGHT: int = int(os.getenv("BAT_MIN_HEIGHT", "30"))  # Min height in pixels
    BAT_MIN_WIDTH: int = int(os.getenv("BAT_MIN_WIDTH", "20"))  # Min width in pixels
    RTDETR_DEVICE: str = os.getenv("RTDETR_DEVICE", "cuda" if os.getenv("USE_GPU", "true").lower() == "true" else "cpu")

    # ===== Tracking =====
    TRACKING_METHOD: str = "deepsort"  # deepsort or bytetrack
    MAX_TRACKING_AGE: int = 30  # frames
    MIN_TRACKING_HITS: int = 3
    TRACKING_IOU_THRESHOLD: float = 0.3

    # ===== Video Processing =====
    SAMPLING_RATE_FPS: int = 10  # frames per second to process
    MAX_VIDEO_DURATION_SECONDS: int = 600  # 10 minutes
    OUTPUT_VIDEO_CODEC: str = "mp4v"
    OUTPUT_VIDEO_FPS: int = 30

    # ===== Processing Queue =====
    MAX_CONCURRENT_JOBS: int = 3
    JOB_TIMEOUT_SECONDS: int = 3600  # 1 hour
    
    # ===== WebSocket Settings =====
    WEBSOCKET_TIMEOUT_SECONDS: int = 3600  # 1 hour - timeout for WebSocket receive operations
    WEBSOCKET_PING_INTERVAL_SECONDS: int = 30  # Send ping every 30 seconds to keep connection alive
    
    # ===== WebSocket Settings =====
    WEBSOCKET_TIMEOUT_SECONDS: int = 3600  # 1 hour - timeout for WebSocket receive operations
    WEBSOCKET_PING_INTERVAL_SECONDS: int = 30  # Send ping every 30 seconds to keep connection alive
    ENABLE_JOB_QUEUE: bool = True
    
    # ===== CORS Settings =====
    CORS_ALLOW_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]  # Can be overridden via CORS_ALLOW_ORIGINS env var (comma-separated)

    # ===== Shot Classification =====
    SHOT_CLASSIFICATION_ENABLED: bool = True
    SHORT_PITCH_DETECTION_ENABLED: bool = True
    MIN_SHOT_CONFIDENCE: float = 0.6

    # ===== Metrics Computation =====
    COMPUTE_JOINT_ANGLES: bool = True
    COMPUTE_VELOCITIES: bool = True
    COMPUTE_BAT_ANGLE: bool = True
    VELOCITY_SMOOTHING_WINDOW: int = 5  # frames

    # ===== Visualization =====
    DRAW_SKELETON: bool = True
    DRAW_BBOXES: bool = True
    DRAW_PLAYER_IDS: bool = True
    DRAW_EVENT_LABELS: bool = True
    SKELETON_THICKNESS: int = 2
    BBOX_THICKNESS: int = 2
    FONT_SCALE: float = 0.6

    # ===== Export Options =====
    EXPORT_JSON_REPORT: bool = True
    EXPORT_CSV_TIMESERIES: bool = True
    EXPORT_ANNOTATED_VIDEO: bool = True
    INCLUDE_POSE_DATA_IN_JSON: bool = True  # Can be large

    # ===== Logging =====
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    LOG_TO_FILE: bool = False
    LOG_FILE_PATH: str = "../logs/pose_analysis.log"

    # ===== Advanced =====
    USE_TENSORRT: bool = False  # TensorRT acceleration (requires setup)
    USE_HALF_PRECISION: bool = False  # FP16 for faster inference
    CLEAR_GPU_CACHE: bool = True

    # ===== Commentary System =====
    REDIS_URL: str = os.getenv("REDIS_URL")
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")  # Optional password for server environments
    RAPIDAPI_KEY: Optional[str] = None  # Legacy: single key (for backward compatibility)
    RAPIDAPI_KEYS: Optional[str] = None  # Comma-separated list of API keys (e.g., "key1,key2,key3")
    RAPIDAPI_HOST: str = "cricbuzz-cricket.p.rapidapi.com"
    RAPIDAPI_BASE_URL: str = "https://cricbuzz-cricket.p.rapidapi.com"
    COMMENTARY_FETCH_INTERVAL: int = 20  # seconds
    WORKER_CLEANUP_INTERVAL: int = 60  # seconds
    ENABLE_HISTORICAL_COMMENTARY: bool = False
    COMMENTARY_DEFAULT_LANGUAGE: str = "en"
    COMMENTARY_SUPPORTED_LANGUAGES: str = "en,hi"
    TRANSLATION_ENABLED: bool = False
    TRANSLATION_CACHE_TTL_SECONDS: int = 900
    BEDROCK_REGION: Optional[str] = None
    BEDROCK_MODEL_ID: Optional[str] = None
    BEDROCK_TEMPERATURE: float = 0.2
    BEDROCK_TOP_P: float = 0.9
    BEDROCK_MAX_TOKENS: int = 3072  # Increased for more detailed analysis
    
    # ===== Googli AI Chat Settings =====
    SERPER_API_KEY: Optional[str] = os.getenv("SERPER_API_KEY")
    SERPER_API_URL: str = "https://google.serper.dev/search"
    CHAT_MODEL_ID: str = os.getenv("CHAT_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0")
    # Inference profile ARN (optional, overrides CHAT_MODEL_ID if set)
    # Format: arn:aws:bedrock:{region}::inference-profile/{model-id}
    # Example: arn:aws:bedrock:us-east-1::inference-profile/anthropic.claude-3-5-sonnet-20241022-v2:0
    CHAT_INFERENCE_PROFILE_ARN: Optional[str] = os.getenv("CHAT_INFERENCE_PROFILE_ARN")
    CHAT_TEMPERATURE: float = float(os.getenv("CHAT_TEMPERATURE", "0.7"))
    CHAT_TOP_P: float = float(os.getenv("CHAT_TOP_P", "0.9"))
    CHAT_MAX_TOKENS: int = int(os.getenv("CHAT_MAX_TOKENS", "512"))  # Allows for more complete responses (~350-400 words)
    CHAT_SYSTEM_PROMPT: str = """
        You are Googli AI, a friendly and knowledgeable cricket expert chatbot. Your primary role is to answer questions about cricket and ONLY cricket, You are like a cricket commentator with a lot of knowledge about cricket, You analyze the image and generate proper answer for user queries related to that image.

        CURRENT DATE AND TIME: {current_datetime}
        
        RESPONSE TONE: {tone}
        RESPONSE LANGUAGE: {language}
        
        TONE INSTRUCTIONS:
        - Strictly Generate Answers in {language} language.
        - Adapt your communication style to match the specified tone: {tone}
        - Professional: Use formal language, clear structure, and authoritative statements. Maintain a balanced, informative approach.
        - Casual: Use relaxed, conversational language. Feel free to use contractions and friendly expressions.
        - Enthusiastic: Show excitement and passion. Use exclamations and energetic language while staying informative.
        - Analytical: Focus on data, statistics, and detailed breakdowns. Use precise terminology and structured explanations.
        - Friendly: Be warm, approachable, and personable. Use a conversational tone with a helpful attitude.
        - Formal: Use very structured, respectful language with proper grammar and formal expressions.
        - Regardless of tone, always maintain accuracy and provide valuable cricket information.

        CRITICAL TIME-AWARENESS INSTRUCTIONS:
        - The current date and time shown above is the ACTUAL present moment.
        - ALWAYS use this current date/time as your reference point when discussing matches, events, schedules, or any time-sensitive information
        - When using the search tool, ensure your responses reflect information current as of the date/time shown above
        - NEVER refer to dates in the past as if they are in the future
        - If search results contain dates, compare them with the current date/time to determine if information is recent or outdated
        - When discussing "today", "yesterday", "this week", "this month", etc., base these relative terms on the current date/time provided
        - Always prioritize the most recent information from search results that matches the current date/time context

        Guidelines:
        DO NOT CALL SEARCH TOOL FOR MORE THAN 1 TIME.
        -1. Keep an open mind while analyzing the image based questions.
        0. Do not use the search tool for image related questions.
        1. ONLY answer questions related to cricket (matches, players, teams, rules, history, statistics, tournaments, etc.)
        2. If a question is not about cricket, politely decline and redirect the user to ask cricket-related questions
        3. Be enthusiastic and passionate about cricket, using cricket terminology naturally
        4. Use the real-time search tool when you need current information about:
        - Live match scores and updates
        - Recent match results and statistics
        - Current team rankings and player performances
        - Recent news and developments in cricket
        - Upcoming fixtures and schedules
        5. When using search results, ALWAYS verify dates against the current date/time provided above to ensure accuracy
        6. If search results contain outdated information (dates before the current date), explicitly note this and use the search tool again with more specific time-based queries
        7. Provide accurate, detailed, and well-structured responses based on the LATEST available information
        8. If you're not certain about historical facts, use the search tool to verify
        9. Be conversational and engaging while maintaining accuracy
        10. CRITICAL: Keep your responses concise and short, keep it to the point, Be direct and to the point while still being informative and helpful.

        Remember: You are Googli AI - a cricket specialist. Stay on topic, use current information, and make every conversation about cricket informative and enjoyable!
    """
    
    def get_rapidapi_keys(self) -> List[str]:
        """
        Get list of RapidAPI keys for rotation.
        Supports both RAPIDAPI_KEYS (comma-separated) and RAPIDAPI_KEY (single key for backward compatibility)
        
        Returns:
            List of API keys
        """
        keys = []
        
        # First, try RAPIDAPI_KEYS (comma-separated list)
        if self.RAPIDAPI_KEYS:
            keys = [key.strip() for key in self.RAPIDAPI_KEYS.split(",") if key.strip()]
        
        # Fallback to RAPIDAPI_KEY for backward compatibility
        if not keys and self.RAPIDAPI_KEY:
            keys = [self.RAPIDAPI_KEY]
        
        return keys
    
    def get_supported_commentary_languages(self) -> List[str]:
        """
        Return normalized list of supported commentary languages.
        Ensures the default language is always included and first in the list.
        """
        langs = [
            lang.strip().lower()
            for lang in (self.COMMENTARY_SUPPORTED_LANGUAGES or "").split(",")
            if lang.strip()
        ]
        default_lang = (self.COMMENTARY_DEFAULT_LANGUAGE or "en").lower()
        if default_lang not in langs:
            langs.insert(0, default_lang)
        # Deduplicate while preserving order
        seen = set()
        normalized = []
        for lang in langs:
            if lang not in seen:
                normalized.append(lang)
                seen.add(lang)
        return normalized
    
    def is_translation_enabled(self) -> bool:
        """Check if translation is enabled and properly configured."""
        return bool(
            self.TRANSLATION_ENABLED
            and self.BEDROCK_REGION
            and self.BEDROCK_MODEL_ID
        )
    
    # ===== Logging =====
    COMMENTARY_LOG_LEVEL: str = "DEBUG"  # DEBUG, INFO, WARNING, ERROR
    COMMENTARY_LOG_TO_FILE: bool = True
    
    # ===== Database =====
    DB_HOST: str = os.getenv("DB_HOST")
    DB_PORT: int = os.getenv("DB_PORT")
    DB_USER: str = os.getenv("DB_USER")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD")
    DB_NAME: str = os.getenv("DB_NAME")
    DB_ECHO: bool = False  # Set to True for SQL query logging

    # ===== Storage Configuration =====
    STORAGE_TYPE: str = os.getenv("STORAGE_TYPE", "local")  # "local" or "s3"
    VIDEO_STORAGE_DIR: str = os.getenv("VIDEO_STORAGE_DIR", "uploads/videos")  # Local storage directory
    
    # ===== AWS S3 Configuration (Optional) =====
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET_NAME: Optional[str] = os.getenv("S3_BUCKET_NAME")
    S3_PRESIGNED_URL_EXPIRATION: int = 3600  # 1 hour in seconds
    USE_S3: bool = os.getenv("USE_S3", "false").lower() == "true"  # Enable S3 if credentials are provided

    # ===== Storage Configuration =====
    STORAGE_TYPE: str = os.getenv("STORAGE_TYPE", "local")  # "local" or "s3"
    VIDEO_STORAGE_DIR: str = os.getenv("VIDEO_STORAGE_DIR", "uploads/videos")  # Local storage directory
    
    # ===== AWS S3 Configuration (Optional) =====
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET_NAME: Optional[str] = os.getenv("S3_BUCKET_NAME")
    S3_PRESIGNED_URL_EXPIRATION: int = 3600  # 1 hour in seconds
    USE_S3: bool = os.getenv("USE_S3", "false").lower() == "true"  # Enable S3 if credentials are provided

    # Pydantic v2 model configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra environment variables not defined in Settings
    )


# Create singleton settings instance
settings = Settings()


# Ensure directories exist
def setup_directories():
    """Create necessary directories if they don't exist"""
    upload_dir = Path(settings.UPLOAD_DIR)
    processed_dir = Path(settings.PROCESSED_DIR)
    video_storage_dir = Path(settings.VIDEO_STORAGE_DIR)
    video_storage_dir = Path(settings.VIDEO_STORAGE_DIR)

    upload_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    video_storage_dir.mkdir(parents=True, exist_ok=True)
    video_storage_dir.mkdir(parents=True, exist_ok=True)

    if settings.LOG_TO_FILE:
        log_dir = Path(settings.LOG_FILE_PATH).parent
        log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[OK] Directories setup complete:")
    print(f"   - Upload: {upload_dir.absolute()}")
    print(f"   - Processed: {processed_dir.absolute()}")
    print(f"   - Video Storage: {video_storage_dir.absolute()}")
    print(f"   - Video Storage: {video_storage_dir.absolute()}")


# Auto-setup on import
setup_directories()
