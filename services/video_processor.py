"""
Video Processing Service
Handles video file reading, frame extraction, and metadata extraction
"""

import cv2
import numpy as np
from typing import List, Tuple, Generator, Optional
from pathlib import Path
from dataclasses import dataclass

from config import settings
from models.schemas import VideoMetadata


@dataclass
class VideoInfo:
    """Video file information"""
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    codec: str


class VideoProcessor:
    """
    Handles video file processing operations
    """

    def __init__(self, video_path: str):
        """
        Initialize video processor

        Args:
            video_path: Path to video file
        """
        self.video_path = Path(video_path)

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        self.info = self._extract_video_info()

    def _extract_video_info(self) -> VideoInfo:
        """Extract video metadata"""
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        # Get codec (fourcc)
        fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

        return VideoInfo(
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration,
            codec=codec
        )

    def get_metadata(self, filename: str, processed_frames: int) -> VideoMetadata:
        """
        Get video metadata for API response

        Args:
            filename: Original filename
            processed_frames: Number of frames actually processed

        Returns:
            VideoMetadata object
        """
        return VideoMetadata(
            filename=filename,
            duration_seconds=self.info.duration_seconds,
            fps=self.info.fps,
            resolution=[self.info.width, self.info.height],
            total_frames=self.info.total_frames,
            processed_frames=processed_frames,
            sampling_rate=settings.SAMPLING_RATE_FPS
        )

    def frame_generator(
        self,
        sample_rate: Optional[int] = None,
        start_frame: int = 0,
        end_frame: Optional[int] = None
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generate frames from video with optional sampling

        Args:
            sample_rate: Sample every Nth frame (uses settings if None)
            start_frame: Starting frame number
            end_frame: Ending frame number (None = end of video)

        Yields:
            Tuple of (frame_number, frame_image)
        """
        if sample_rate is None:
            # Calculate sample rate based on FPS
            sample_rate = max(1, int(self.info.fps / settings.SAMPLING_RATE_FPS))

        if end_frame is None:
            end_frame = self.info.total_frames

        # Set starting position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_num = start_frame

        while frame_num < end_frame:
            ret, frame = self.cap.read()

            if not ret:
                break

            # Yield frame if it matches sampling rate
            if (frame_num - start_frame) % sample_rate == 0:
                yield frame_num, frame

            frame_num += 1

    def get_batch_frames(
        self,
        batch_size: Optional[int] = None,
        sample_rate: Optional[int] = None
    ) -> Generator[List[Tuple[int, np.ndarray]], None, None]:
        """
        Generate batches of frames for efficient processing

        Args:
            batch_size: Number of frames per batch (uses settings if None)
            sample_rate: Sample every Nth frame (uses settings if None)

        Yields:
            List of (frame_number, frame_image) tuples
        """
        if batch_size is None:
            batch_size = settings.BATCH_SIZE

        batch = []

        for frame_num, frame in self.frame_generator(sample_rate=sample_rate):
            batch.append((frame_num, frame))

            if len(batch) >= batch_size:
                yield batch
                batch = []

        # Yield remaining frames
        if batch:
            yield batch

    def get_frame_at_time(self, time_seconds: float) -> Optional[np.ndarray]:
        """
        Get frame at specific timestamp

        Args:
            time_seconds: Timestamp in seconds

        Returns:
            Frame image or None if failed
        """
        frame_num = int(time_seconds * self.info.fps)

        if frame_num >= self.info.total_frames:
            return None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()

        return frame if ret else None

    def get_frame_at_index(self, frame_num: int) -> Optional[np.ndarray]:
        """
        Get frame at specific index

        Args:
            frame_num: Frame number

        Returns:
            Frame image or None if failed
        """
        if frame_num >= self.info.total_frames or frame_num < 0:
            return None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()

        return frame if ret else None

    def reset(self):
        """Reset video to beginning"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def cleanup(self):
        """Release video capture resources"""
        if self.cap:
            self.cap.release()

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()


class VideoWriter:
    """
    Handles writing annotated video output
    """

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float
    ):
        """
        Initialize video writer

        Args:
            output_path: Output video file path
            width: Frame width
            height: Frame height
            fps: Frames per second
        """
        self.output_path = Path(output_path)
        self.width = width
        self.height = height
        self.fps = fps

        # Create output directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get codec
        fourcc = cv2.VideoWriter_fourcc(*settings.OUTPUT_VIDEO_CODEC)

        # Create video writer
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            fps,
            (width, height)
        )

        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to create video writer: {output_path}")

    def write_frame(self, frame: np.ndarray):
        """
        Write a single frame to output video

        Args:
            frame: Frame image (numpy array)
        """
        # Ensure frame is correct size
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))

        self.writer.write(frame)

    def write_frames(self, frames: List[np.ndarray]):
        """
        Write multiple frames to output video

        Args:
            frames: List of frame images
        """
        for frame in frames:
            self.write_frame(frame)

    def cleanup(self):
        """Release video writer resources"""
        if self.writer:
            self.writer.release()

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()


def validate_video_file(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate video file

    Args:
        file_path: Path to video file

    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(file_path)

    # Check file exists
    if not path.exists():
        return False, "File not found"

    # Check file size
    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > settings.MAX_UPLOAD_SIZE_MB:
        return False, f"File too large ({file_size_mb:.1f}MB > {settings.MAX_UPLOAD_SIZE_MB}MB)"

    # Try to open video
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return False, "Failed to open video file"

        # Check duration
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = total_frames / fps if fps > 0 else 0

        cap.release()

        if duration > settings.MAX_VIDEO_DURATION_SECONDS:
            return False, f"Video too long ({duration:.1f}s > {settings.MAX_VIDEO_DURATION_SECONDS}s)"

        return True, None

    except Exception as e:
        return False, f"Error validating video: {str(e)}"
