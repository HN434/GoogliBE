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
    ENABLE_JOB_QUEUE: bool = True

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
    REDIS_URL: str = "redis://localhost:6379"
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
    BEDROCK_MAX_TOKENS: int = 1024
    
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

    upload_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if settings.LOG_TO_FILE:
        log_dir = Path(settings.LOG_FILE_PATH).parent
        log_dir.mkdir(parents=True, exist_ok=True)

    print(f"✅ Directories setup complete:")
    print(f"   - Upload: {upload_dir.absolute()}")
    print(f"   - Processed: {processed_dir.absolute()}")


# Auto-setup on import
setup_directories()
