"""
3D Pose & Mesh Generation Pipeline
Complete pipeline: Video/Images -> RTM Pose -> 3D SMPL -> Mesh -> Props -> glTF
"""

import logging
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json

from config import settings
from worker.inference.pose_estimator import get_pose_estimator, PoseEstimatorResult
from services.smpl_estimator import SMPLEstimator, create_smpl_estimator
from services.mesh_generator import MeshGenerator, create_mesh_generator
from services.prop_attacher import PropAttacher, create_prop_attacher
from services.gltf_exporter import GLTFExporter, create_gltf_exporter

logger = logging.getLogger(__name__)


class Pipeline3D:
    """
    Complete 3D pose and mesh generation pipeline.
    
    Pipeline stages:
    1. Video/Images -> RTM Pose (2D keypoints)
    2. RTM Pose -> 3D pose & shape estimation (SMPL/SMPL-X)
    3. SMPL parameters -> 3D mesh + skeleton
    4. (Optional) Attach props to hand joints
    5. Export to glTF (.glb)
    """
    
    def __init__(
        self,
        smpl_model_path: Optional[str] = None,
        props_dir: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Initialize 3D pipeline.
        
        Args:
            smpl_model_path: Path to SMPL-X model files
            props_dir: Directory containing 3D prop models
            device: Device to run on ("cpu" or "cuda")
        """
        self.device = device
        
        # Initialize components
        logger.info("Initializing 3D pipeline components...")
        
        # RTM Pose estimator (lazy loaded)
        self.pose_estimator = None
        
        # SMPL estimator
        try:
            self.smpl_estimator = create_smpl_estimator(
                model_path=smpl_model_path,
                device=device,
            )
            logger.info("SMPL estimator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize SMPL estimator: {e}")
            self.smpl_estimator = None
        
        # Mesh generator
        self.mesh_generator = create_mesh_generator()
        logger.info("Mesh generator initialized")
        
        # Prop attacher
        self.prop_attacher = create_prop_attacher(props_dir=props_dir)
        logger.info("Prop attacher initialized")
        
        # glTF exporter
        self.gltf_exporter = create_gltf_exporter()
        logger.info("glTF exporter initialized")
    
    def _get_pose_estimator(self):
        """Lazy load RTM Pose estimator."""
        if self.pose_estimator is None:
            self.pose_estimator = get_pose_estimator()
        return self.pose_estimator
    
    def process_video(
        self,
        video_path: str,
        output_dir: str,
        attach_props: bool = True,
        prop_names: Optional[List[str]] = None,
        export_format: str = "glb",
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict:
        """
        Process video through the complete 3D pipeline.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save outputs
            attach_props: Whether to attach props (bat, etc.)
            prop_names: List of prop names to attach (default: ["bat"])
            export_format: Export format ("glb" or "gltf")
            fps: Frames per second (if None, uses video FPS)
            max_frames: Maximum number of frames to process (None for all)
            progress_callback: Optional callback function(progress: float, message: str)
        
        Returns:
            Dictionary with processing results and output paths
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        logger.info(f"Starting 3D pipeline for video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if fps is None:
            fps = video_fps
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        logger.info(
            f"Video info: {width}x{height} @ {video_fps} fps, "
            f"{total_frames} frames"
        )
        
        # Initialize components
        pose_estimator = self._get_pose_estimator()
        
        # Process frames
        frame_index = 0
        all_mesh_frames = []
        all_estimations = []
        
        if progress_callback:
            progress_callback(0.0, "Processing video frames...")
        
        try:
            while True:
                if max_frames and frame_index >= max_frames:
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Stage 1: RTM Pose - Extract 2D keypoints
                pose_results = pose_estimator.infer(frame)
                
                if not pose_results:
                    logger.debug(f"Frame {frame_index}: No pose detected")
                    frame_index += 1
                    continue
                
                # Process first person (can be extended for multi-person)
                pose_result = pose_results[0]
                keypoints_2d = pose_result.keypoints
                keypoint_scores = pose_result.scores
                
                # Stage 2: 3D pose & shape estimation (SMPL)
                if self.smpl_estimator:
                    estimation = self.smpl_estimator.estimate_from_2d_keypoints(
                        keypoints_2d=keypoints_2d,
                        keypoint_scores=keypoint_scores,
                        image_shape=(height, width),
                    )
                else:
                    logger.warning("SMPL estimator not available, using simplified 3D")
                    # Create a temporary simplified estimator for fallback
                    from services.smpl_estimator import SMPLEstimator
                    temp_estimator = SMPLEstimator(model_path=None, device=self.device)
                    estimation = temp_estimator._estimate_simplified(
                        keypoints_2d=keypoints_2d,
                        keypoint_scores=keypoint_scores,
                        image_shape=(height, width),
                    )
                
                all_estimations.append({
                    "frame_index": frame_index,
                    "estimation": estimation,
                })
                
                # Stage 3: Generate 3D mesh + skeleton
                mesh_data = self.mesh_generator.generate_mesh(
                    estimation,
                    frame_index=frame_index,
                )
                
                # Stage 4: Attach props (optional)
                if attach_props:
                    if prop_names is None:
                        prop_names = ["bat"]
                    
                    # Get hand joints
                    hand_joints = self.smpl_estimator.get_hand_joints(estimation) if self.smpl_estimator else {}
                    
                    # Attach props
                    for prop_name in prop_names:
                        if "bat" in prop_name.lower():
                            mesh_data = self.prop_attacher.attach_bat_to_hands(
                                mesh_data,
                                hand_joints,
                                bat_name=prop_name,
                            )
                        else:
                            # Generic prop attachment
                            if "right_hand" in hand_joints and hand_joints["right_hand"] is not None:
                                mesh_data = self.prop_attacher.attach_prop_to_hand(
                                    mesh_data,
                                    hand_joints["right_hand"],
                                    prop_name=prop_name,
                                    hand_side="right",
                                )
                
                all_mesh_frames.append(mesh_data)
                
                # Progress update
                if progress_callback and frame_index % 10 == 0:
                    progress = (frame_index + 1) / total_frames * 100
                    progress_callback(progress, f"Processed {frame_index + 1}/{total_frames} frames")
                
                frame_index += 1
        
        finally:
            cap.release()
        
        logger.info(f"Processed {frame_index} frames")
        
        if progress_callback:
            progress_callback(50.0, "Generating animation...")
        
        # Stage 5: Create animation
        animation_data = self.mesh_generator.create_animation(
            all_mesh_frames,
            fps=fps,
        )
        
        if progress_callback:
            progress_callback(75.0, "Exporting to glTF...")
        
        # Stage 6: Export to glTF
        output_name = video_path.stem
        gltf_output_path = output_dir / f"{output_name}_3d_animation.{export_format}"
        
        gltf_path = self.gltf_exporter.export_animation(
            animation_data,
            str(gltf_output_path),
            format=export_format,
        )
        
        # Also export per-frame meshes if needed
        frame_meshes_dir = output_dir / f"{output_name}_frames"
        frame_meshes_dir.mkdir(exist_ok=True)
        
        for i, mesh_frame in enumerate(all_mesh_frames):
            frame_path = frame_meshes_dir / f"frame_{i:06d}.{export_format}"
            try:
                self.gltf_exporter.export_mesh(
                    mesh_frame,
                    str(frame_path),
                    format=export_format,
                )
            except Exception as e:
                logger.warning(f"Failed to export frame {i}: {e}")
        
        if progress_callback:
            progress_callback(100.0, "Complete!")
        
        # Save metadata
        metadata = {
            "video_path": str(video_path),
            "total_frames": frame_index,
            "fps": fps,
            "video_fps": video_fps,
            "resolution": {"width": width, "height": height},
            "gltf_path": gltf_path,
            "frames_dir": str(frame_meshes_dir),
            "props_attached": attach_props,
            "prop_names": prop_names if attach_props else [],
        }
        
        metadata_path = output_dir / f"{output_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"3D pipeline complete. Output: {gltf_path}")
        
        return {
            "success": True,
            "gltf_path": gltf_path,
            "metadata_path": str(metadata_path),
            "frames_dir": str(frame_meshes_dir),
            "metadata": metadata,
        }
    
    def process_images(
        self,
        image_paths: List[str],
        output_dir: str,
        attach_props: bool = True,
        prop_names: Optional[List[str]] = None,
        export_format: str = "glb",
        fps: float = 30.0,
        progress_callback: Optional[callable] = None,
    ) -> Dict:
        """
        Process a sequence of images through the 3D pipeline.
        
        Args:
            image_paths: List of paths to input images
            output_dir: Directory to save outputs
            attach_props: Whether to attach props
            prop_names: List of prop names to attach
            export_format: Export format ("glb" or "gltf")
            fps: Frames per second for animation
            progress_callback: Optional callback function
        
        Returns:
            Dictionary with processing results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing {len(image_paths)} images")
        
        pose_estimator = self._get_pose_estimator()
        all_mesh_frames = []
        
        if progress_callback:
            progress_callback(0.0, "Processing images...")
        
        for i, image_path in enumerate(image_paths):
            image_path = Path(image_path)
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}, skipping")
                continue
            
            # Load image
            frame = cv2.imread(str(image_path))
            if frame is None:
                logger.warning(f"Failed to load image: {image_path}, skipping")
                continue
            
            height, width = frame.shape[:2]
            
            # Stage 1: RTM Pose
            pose_results = pose_estimator.infer(frame)
            if not pose_results:
                continue
            
            pose_result = pose_results[0]
            keypoints_2d = pose_result.keypoints
            keypoint_scores = pose_result.scores
            
            # Stage 2: SMPL estimation
            if self.smpl_estimator:
                estimation = self.smpl_estimator.estimate_from_2d_keypoints(
                    keypoints_2d=keypoints_2d,
                    keypoint_scores=keypoint_scores,
                    image_shape=(height, width),
                )
            else:
                logger.warning("SMPL estimator not available, using simplified 3D")
                # Create a temporary simplified estimator for fallback
                from services.smpl_estimator import SMPLEstimator
                temp_estimator = SMPLEstimator(model_path=None, device=self.device)
                estimation = temp_estimator._estimate_simplified(
                    keypoints_2d=keypoints_2d,
                    keypoint_scores=keypoint_scores,
                    image_shape=(height, width),
                )
            
            # Stage 3: Generate mesh
            mesh_data = self.mesh_generator.generate_mesh(
                estimation,
                frame_index=i,
            )
            
            # Stage 4: Attach props
            if attach_props:
                if prop_names is None:
                    prop_names = ["bat"]
                
                hand_joints = self.smpl_estimator.get_hand_joints(estimation) if self.smpl_estimator else {}
                
                for prop_name in prop_names:
                    if "bat" in prop_name.lower():
                        mesh_data = self.prop_attacher.attach_bat_to_hands(
                            mesh_data,
                            hand_joints,
                            bat_name=prop_name,
                        )
            
            all_mesh_frames.append(mesh_data)
            
            if progress_callback:
                progress = (i + 1) / len(image_paths) * 100
                progress_callback(progress, f"Processed {i + 1}/{len(image_paths)} images")
        
        # Create animation and export
        animation_data = self.mesh_generator.create_animation(
            all_mesh_frames,
            fps=fps,
        )
        
        output_name = Path(image_paths[0]).stem if image_paths else "images"
        gltf_output_path = output_dir / f"{output_name}_3d_animation.{export_format}"
        
        gltf_path = self.gltf_exporter.export_animation(
            animation_data,
            str(gltf_output_path),
            format=export_format,
        )
        
        return {
            "success": True,
            "gltf_path": gltf_path,
            "num_frames": len(all_mesh_frames),
        }


def create_pipeline_3d(
    smpl_model_path: Optional[str] = None,
    props_dir: Optional[str] = None,
    device: str = "cpu",
) -> Pipeline3D:
    """
    Factory function to create 3D pipeline instance.
    
    Args:
        smpl_model_path: Path to SMPL-X model files
        props_dir: Directory containing 3D prop models
        device: Device to run on
    
    Returns:
        Pipeline3D instance
    """
    return Pipeline3D(
        smpl_model_path=smpl_model_path,
        props_dir=props_dir,
        device=device,
    )

