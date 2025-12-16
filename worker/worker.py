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


async def main(num_workers: int = 2):
    """Main worker entry point (ARQ async worker)

    Args:
        num_workers: Number of ARQ worker instances to run in this process.
            Each worker pulls jobs from the same Redis queue ("video-processing").
            Jobs are claimed atomically by ARQ, so a given job/video will only be
            processed by a single worker, even when multiple workers are running.
    """
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
        # Establish a connection pool once to validate connectivity; individual
        # workers will use redis_settings to manage their own connections.
        await create_pool(redis_settings)
        logger.info("✅ Redis connection pool established for ARQ")

        # Configure and start one or more ARQ workers in this process.
        # ARQ/Redis ensure that each enqueued job is claimed by exactly one worker,
        # so multiple workers will NOT process the same video job concurrently.
        workers = [
            ArqWorker(
                functions=["worker.jobs.analyze_video.analyze_video_job_async"],
                redis_settings=redis_settings,
                queue_name="video-processing",
                burst=False,
            )
            for _ in range(max(1, int(num_workers)))
        ]

        logger.info("=" * 50)
        logger.info(
            "✅ ARQ Workers started (count=%d, queue=%s)",
            len(workers),
            "video-processing",
        )
        logger.info("=" * 50)
        logger.info("Workers are ready to process jobs...")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 50)

        import asyncio as _asyncio

        await _asyncio.gather(*(w.async_run() for w in workers))

    except KeyboardInterrupt:
        logger.info("\n🛑 Worker stopped by user")
    except Exception as e:
        logger.error(f"❌ Worker error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    import asyncio
    # Optional CLI argument: number of workers to run in this process.
    # Usage:
    #   python -m worker.worker           # starts 2 workers (default)
    #   python -m worker.worker 3         # starts 3 workers
    default_workers = 2
    num_workers = default_workers
    if len(sys.argv) > 1:
        try:
            num_workers = int(sys.argv[1])
        except ValueError:
            logger.warning(
                "Invalid num_workers argument %r, falling back to default=%d",
                sys.argv[1],
                default_workers,
            )

    asyncio.run(main(num_workers=num_workers))

