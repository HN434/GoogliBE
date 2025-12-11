# Video Processing Worker

This worker runs on a dedicated GPU EC2 machine to process video analysis jobs from the Redis queue and ships with CPU-only RTMPose scaffolding (see `docs/pose_inference.md`).

## Setup

1. Install dependencies:
```bash
pip install -r ../requirements.txt
```

2. Configure environment variables (same as main backend):
```env
REDIS_URL=redis://localhost:6379/0
DB_HOST=your-db-host
DB_PORT=5432
DB_USER=your-db-user
DB_PASSWORD=your-db-password
DB_NAME=your-db-name
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-bucket-name
```

## Running the Worker

From the project root:
```bash
python -m worker.worker
```

Or from the worker directory:
```bash
python worker.py
```

The worker will:
- Connect to Redis
- Listen on the "video-processing" queue
- Process jobs as they arrive
- Update database with progress and results

## Worker Process

The worker processes videos through these steps:

1. **Download** video from S3
2. **Extract** metadata (duration, fps, width, height)
3. **Run** pose detection model (RTMPose - see `docs/pose_inference.md`)
4. **Generate** keypoints JSON
5. **Create** overlay video (optional)
6. **Upload** results to S3
7. **Update** database with outputs and metrics

## Monitoring

- Check worker logs for job processing status
- Monitor Redis queue: `rq info video-processing`
- Check database for video status updates

## Job Timeout

Jobs have a 10-minute timeout. For longer videos, adjust `job_timeout` in `services/video_analysis_queue.py`.

For detailed setup instructions (CPU dependencies, RTMPose asset download, and the sanity-check script), read `docs/pose_inference.md`.

