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
    
    # Extract host, port, db from URL
    if redis_url.startswith("redis://"):
        parts = redis_url.replace("redis://", "").split("/")
        host_port = parts[0].split(":")
        host = host_port[0] if len(host_port) > 0 else "localhost"
        port = int(host_port[1]) if len(host_port) > 1 else 6379
        db = int(parts[1]) if len(parts) > 1 else 0
    else:
        host = "localhost"
        port = 6379
        db = 0
    
    logger.info(f"Connecting to Redis at {host}:{port}/{db}")

    try:
        # ARQ uses its own Redis connection pool
        redis_settings = RedisSettings(
            host=host,
            port=port,
            database=db,
        )
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

