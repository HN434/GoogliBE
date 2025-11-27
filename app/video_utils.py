"""
Video processing utilities
Extract frames, get video info, etc.
"""

import cv2
import numpy as np
from typing import Iterator, Dict


class VideoProcessor:
    """Process video files and extract frames"""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

    def get_video_info(self) -> Dict:
        """
        Get video metadata

        Returns:
            Dictionary with fps, frame_count, duration, resolution
        """

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0

        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": duration
        }

    def extract_frames(self, skip_frames: int = 0) -> Iterator[np.ndarray]:
        """
        Extract frames from video

        Args:
            skip_frames: Number of frames to skip between extractions (0 = all frames)

        Yields:
            numpy array: Frame in BGR format
        """

        # Reset to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_idx = 0

        while True:
            ret, frame = self.cap.read()

            if not ret:
                break

            # Skip frames if specified
            if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                frame_idx += 1
                continue

            yield frame
            frame_idx += 1

    def extract_frame_at_time(self, timestamp: float) -> np.ndarray:
        """
        Extract single frame at specific timestamp

        Args:
            timestamp: Time in seconds

        Returns:
            Frame as numpy array
        """

        # Set position
        self.cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

        ret, frame = self.cap.read()

        if not ret:
            raise ValueError(f"Failed to extract frame at {timestamp}s")

        return frame

    def __del__(self):
        """Release video capture on deletion"""
        if hasattr(self, 'cap'):
            self.cap.release()
