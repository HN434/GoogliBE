"""
RQ Worker for Video Processing
Runs on GPU EC2 machine to process video analysis jobs
"""

import logging
import sys
from redis import Redis
from arq import Worker as ArqWorker
from arq.connections import RedisSettings, create_pool
from config import settings
from worker.inference import get_pose_estimator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger(__name__)


async def main():
    """Main worker entry point (ARQ async worker)"""
    logger.info("=" * 50)
    logger.info("🚀 Starting ARQ Worker for Video Processing")
    logger.info("=" * 50)

    # Warm up RTMPose so inference graph is resident before jobs arrive
    try:
        pose_estimator = get_pose_estimator()
        logger.info(
            "✅ RTMPose ready (device=%s, config=%s)",
            pose_estimator.device,
            pose_estimator.config_path,
        )
    except Exception as pose_error:
        logger.error("❌ Failed to initialize RTMPose: %s", pose_error, exc_info=True)
        sys.exit(1)
    
    # Parse Redis URL or use defaults
    redis_url = settings.REDIS_URL or "redis://localhost:6379/0"
    password = None
    
    # Extract host, port, db, and password from URL
    if redis_url.startswith("redis://"):
        # Handle URL with password: redis://:password@host:port/db
        url_without_protocol = redis_url.replace("redis://", "")
        
        # Check if password is in URL
        if "@" in url_without_protocol:
            # Format: :password@host:port/db or username:password@host:port/db
            auth_and_rest = url_without_protocol.split("@")
            auth_part = auth_and_rest[0]
            rest = auth_and_rest[1]
            
            # Extract password (format: :password or username:password)
            if ":" in auth_part:
                password = auth_part.split(":")[-1] if auth_part.startswith(":") else auth_part.split(":")[1]
            
            # Parse host, port, db from rest
            parts = rest.split("/")
            host_port = parts[0].split(":")
            host = host_port[0] if len(host_port) > 0 else "localhost"
            port = int(host_port[1]) if len(host_port) > 1 else 6379
            db = int(parts[1]) if len(parts) > 1 else 0
        else:
            # No password in URL, parse normally
            parts = url_without_protocol.split("/")
            host_port = parts[0].split(":")
            host = host_port[0] if len(host_port) > 0 else "localhost"
            port = int(host_port[1]) if len(host_port) > 1 else 6379
            db = int(parts[1]) if len(parts) > 1 else 0
            
            # Use password from settings if not in URL (for server environments)
            if settings.REDIS_PASSWORD:
                password = settings.REDIS_PASSWORD
    else:
        host = "localhost"
        port = 6379
        db = 0
        # Use password from settings if provided (for server environments)
        if settings.REDIS_PASSWORD:
            password = settings.REDIS_PASSWORD
    
    logger.info(f"Connecting to Redis at {host}:{port}/{db}")

    try:
        # ARQ uses its own Redis connection pool
        redis_settings = RedisSettings(
            host=host,
            port=port,
            database=db,
        )
        # Add password if available
        if password:
            redis_settings.password = password
        redis = await create_pool(redis_settings)
        logger.info("✅ Redis connection pool established for ARQ")

        # Configure and start ARQ worker
        worker = ArqWorker(
            functions=["worker.jobs.analyze_video.analyze_video_job_async"],
            redis_settings=redis_settings,
            queue_name="video-processing",
            burst=False,
        )
        logger.info("=" * 50)
        logger.info(f"✅ ARQ Worker started (queue: video-processing)")
        logger.info("=" * 50)
        logger.info("Worker is ready to process jobs...")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 50)

        await worker.async_run()

    except KeyboardInterrupt:
        logger.info("\n🛑 Worker stopped by user")
    except Exception as e:
        logger.error(f"❌ Worker error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

