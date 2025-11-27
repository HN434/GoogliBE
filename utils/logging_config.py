"""
Logging configuration for the commentary system
Sets up file and console logging with proper formatting
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Create logs directory if it doesn't exist
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Log file path
LOG_FILE = LOG_DIR / f"commentary_{datetime.now().strftime('%Y%m%d')}.log"


def setup_logging(level: str = "INFO", log_to_file: bool = True):
    """
    Setup logging configuration for commentary system
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to log to file in addition to console
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter with detailed information
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Simpler formatter for console
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Console handler (always enabled)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if enabled)
    if log_to_file:
        # Rotating file handler (max 10MB per file, keep 5 backups)
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        
        logging.info(f"📝 Logging to file: {LOG_FILE.absolute()}")
    
    # Set specific loggers to appropriate levels
    commentary_loggers = {
        'services.redis_service': logging.DEBUG,
        'services.commentary_service': logging.DEBUG,
        'workers.match_worker': logging.DEBUG,
        'workers.worker_supervisor': logging.DEBUG,
        'ws_manager.connection_manager': logging.DEBUG,
        'app.commentary_router': logging.DEBUG,
    }
    
    for logger_name, logger_level in commentary_loggers.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(logger_level)
    
    # Reduce noise from some libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('redis').setLevel(logging.WARNING)
    
    logging.info(f"✅ Logging configured at level {level}")
    if log_to_file:
        logging.info(f"   Console: Enabled")
        logging.info(f"   File: {LOG_FILE}")
    
    return root_logger
