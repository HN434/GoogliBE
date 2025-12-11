"""
API Router for 3D Pose & Mesh Generation Pipeline
"""

import logging
import os
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from config import settings
from services.pipeline_3d import create_pipeline_3d

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pipeline-3d", tags=["3D Pipeline"])


class Pipeline3DRequest(BaseModel):
    """Request model for 3D pipeline processing."""
    attach_props: bool = True
    prop_names: Optional[List[str]] = None
    export_format: str = "glb"  # "glb" or "gltf"
    max_frames: Optional[int] = None
    fps: Optional[float] = None


class Pipeline3DResponse(BaseModel):
    """Response model for 3D pipeline processing."""
    success: bool
    message: str
    gltf_path: Optional[str] = None
    metadata_path: Optional[str] = None
    frames_dir: Optional[str] = None
    job_id: Optional[str] = None


# Global pipeline instance (lazy loaded)
_pipeline_3d = None


def get_pipeline_3d():
    """Get or create 3D pipeline instance."""
    global _pipeline_3d
    if _pipeline_3d is None:
        smpl_model_path = settings.SMPL_MODEL_PATH
        props_dir = settings.PROPS_DIR
        device = settings.PIPELINE_3D_DEVICE if settings.USE_GPU else "cpu"
        
        _pipeline_3d = create_pipeline_3d(
            smpl_model_path=smpl_model_path,
            props_dir=props_dir,
            device=device,
        )
    return _pipeline_3d


@router.post("/process-video", response_model=Pipeline3DResponse)
async def process_video_3d(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    attach_props: bool = True,
    prop_names: Optional[str] = None,  # Comma-separated string
    export_format: str = "glb",
    max_frames: Optional[int] = None,
    fps: Optional[float] = None,
):
    """
    Process video through 3D pose & mesh generation pipeline.
    
    Pipeline stages:
    1. Video -> RTM Pose (2D keypoints)
    2. RTM Pose -> 3D pose & shape (SMPL/SMPL-X)
    3. SMPL -> 3D mesh + skeleton
    4. (Optional) Attach props to hand joints
    5. Export to glTF (.glb)
    
    Args:
        video: Video file to process
        attach_props: Whether to attach props (bat, etc.)
        prop_names: Comma-separated list of prop names (e.g., "bat,bat_handle")
        export_format: Export format ("glb" or "gltf")
        max_frames: Maximum number of frames to process
        fps: Frames per second (if None, uses video FPS)
    
    Returns:
        Response with job ID and status
    """
    try:
        # Parse prop names
        prop_names_list = None
        if prop_names:
            prop_names_list = [name.strip() for name in prop_names.split(",") if name.strip()]
        
        # Save uploaded video
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        video_path = upload_dir / video.filename
        with open(video_path, "wb") as f:
            content = await video.read()
            f.write(content)
        
        logger.info(f"Received video: {video.filename}, size: {len(content)} bytes")
        
        # Create output directory
        output_dir = Path(settings.PROCESSED_DIR) / "3d_pipeline" / video_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process video
        pipeline = get_pipeline_3d()
        
        def progress_callback(progress: float, message: str):
            """Progress callback for pipeline."""
            logger.info(f"Progress: {progress:.1f}% - {message}")
        
        result = pipeline.process_video(
            video_path=str(video_path),
            output_dir=str(output_dir),
            attach_props=attach_props,
            prop_names=prop_names_list,
            export_format=export_format,
            fps=fps,
            max_frames=max_frames,
            progress_callback=progress_callback,
        )
        
        return Pipeline3DResponse(
            success=True,
            message="Video processed successfully",
            gltf_path=result.get("gltf_path"),
            metadata_path=result.get("metadata_path"),
            frames_dir=result.get("frames_dir"),
        )
    
    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.post("/process-images", response_model=Pipeline3DResponse)
async def process_images_3d(
    images: List[UploadFile] = File(...),
    attach_props: bool = True,
    prop_names: Optional[str] = None,
    export_format: str = "glb",
    fps: float = 30.0,
):
    """
    Process a sequence of images through 3D pipeline.
    
    Args:
        images: List of image files to process
        attach_props: Whether to attach props
        prop_names: Comma-separated list of prop names
        export_format: Export format ("glb" or "gltf")
        fps: Frames per second for animation
    
    Returns:
        Response with output paths
    """
    try:
        # Parse prop names
        prop_names_list = None
        if prop_names:
            prop_names_list = [name.strip() for name in prop_names.split(",") if name.strip()]
        
        # Save uploaded images
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        image_paths = []
        for image in images:
            image_path = upload_dir / image.filename
            with open(image_path, "wb") as f:
                content = await image.read()
                f.write(content)
            image_paths.append(str(image_path))
        
        logger.info(f"Received {len(image_paths)} images")
        
        # Create output directory
        output_dir = Path(settings.PROCESSED_DIR) / "3d_pipeline" / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process images
        pipeline = get_pipeline_3d()
        
        result = pipeline.process_images(
            image_paths=image_paths,
            output_dir=str(output_dir),
            attach_props=attach_props,
            prop_names=prop_names_list,
            export_format=export_format,
            fps=fps,
        )
        
        return Pipeline3DResponse(
            success=True,
            message=f"Processed {len(image_paths)} images successfully",
            gltf_path=result.get("gltf_path"),
        )
    
    except Exception as e:
        logger.error(f"Error processing images: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("/download/{filename:path}")
async def download_output(filename: str):
    """
    Download output file (glTF, metadata, etc.).
    
    Args:
        filename: Name of the file to download (relative to processed/3d_pipeline/)
    
    Returns:
        File response
    """
    try:
        file_path = Path(settings.PROCESSED_DIR) / "3d_pipeline" / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        if not file_path.is_file():
            raise HTTPException(status_code=400, detail="Not a file")
        
        return FileResponse(
            path=str(file_path),
            filename=file_path.name,
            media_type="application/octet-stream",
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@router.get("/status")
async def get_pipeline_status():
    """
    Get status of 3D pipeline and available components.
    
    Returns:
        Status information
    """
    try:
        pipeline = get_pipeline_3d()
        
        status = {
            "pipeline_initialized": pipeline is not None,
            "smpl_available": pipeline.smpl_estimator is not None if pipeline else False,
            "mesh_generator_available": pipeline.mesh_generator is not None if pipeline else False,
            "prop_attacher_available": pipeline.prop_attacher is not None if pipeline else False,
            "gltf_exporter_available": pipeline.gltf_exporter is not None if pipeline else False,
            "device": pipeline.device if pipeline else None,
        }
        
        return JSONResponse(content=status)
    
    except Exception as e:
        logger.error(f"Error getting status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

