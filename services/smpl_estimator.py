"""
3D Pose & Shape Estimation using SMPL/SMPL-X
Converts 2D keypoints from RTM Pose to 3D body shape and pose parameters.
"""

import logging
import numpy as np
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

try:
    import smplx
    SMPLX_AVAILABLE = True
except ImportError:
    SMPLX_AVAILABLE = False
    logger.warning("smplx not available. Install with: pip install smplx")

try:
    from scipy.spatial.transform import Rotation as R
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available. Install with: pip install scipy")


class SMPLEstimator:
    """
    Estimates 3D body shape and pose from 2D keypoints using SMPL-X model.
    
    This class takes 2D keypoints from RTM Pose and estimates:
    - Body pose parameters (joint rotations)
    - Shape parameters (beta - body shape coefficients)
    - Global translation and rotation
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "smplx",  # "smpl", "smplh", or "smplx"
        gender: str = "neutral",  # "neutral", "male", or "female"
        device: str = "cpu",
        use_hands: bool = True,
        use_face: bool = False,
        num_betas: int = 10,  # Number of shape parameters
    ):
        """
        Initialize SMPL-X estimator.
        
        Args:
            model_path: Path to SMPL-X model files (.pkl or .npz)
            model_type: Type of model ("smpl", "smplh", "smplx")
            gender: Gender of the model
            device: Device to run on ("cpu" or "cuda")
            use_hands: Whether to use hand joints
            use_face: Whether to use face joints
            num_betas: Number of shape parameters
        """
        if not SMPLX_AVAILABLE:
            raise ImportError(
                "smplx is required for 3D pose estimation. "
                "Install with: pip install smplx"
            )
        
        self.model_type = model_type
        self.device = device
        self.gender = gender
        self.use_hands = use_hands
        self.use_face = use_face
        self.num_betas = num_betas
        
        # Default model path (user should download SMPL-X models)
        if model_path is None:
            model_path = self._get_default_model_path()
        
        self.model_path = Path(model_path) if model_path else None
        
        # Initialize SMPL-X model
        self.model = None
        if self.model_path and self.model_path.exists():
            try:
                self.model = smplx.create(
                    model_path=str(self.model_path.parent),
                    model_type=model_type,
                    gender=gender,
                    use_hands=use_hands,
                    use_face=use_face,
                    num_betas=num_betas,
                    ext="npz" if model_path.suffix == ".npz" else "pkl",
                )
                self.model = self.model.to(device)
                logger.info(f"SMPL-X model loaded from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load SMPL-X model: {e}. Using simplified estimation.")
                self.model = None
        else:
            logger.warning(
                f"SMPL-X model not found at {model_path}. "
                "Using simplified 3D estimation without full body shape."
            )
    
    def _get_default_model_path(self) -> Optional[str]:
        """Get default path for SMPL-X models."""
        # Check common locations
        possible_paths = [
            "models/smplx",
            "models/smpl",
            "../models/smplx",
            "../models/smpl",
        ]
        for path in possible_paths:
            p = Path(path)
            if p.exists():
                # Look for model file
                for ext in [".pkl", ".npz"]:
                    model_file = p / f"SMPLX_{self.gender.upper()}{ext}"
                    if model_file.exists():
                        return str(model_file)
        return None
    
    def estimate_from_2d_keypoints(
        self,
        keypoints_2d: np.ndarray,
        keypoint_scores: Optional[np.ndarray] = None,
        image_shape: Optional[Tuple[int, int]] = None,
        initial_pose: Optional[Dict] = None,
    ) -> Dict:
        """
        Estimate 3D pose and shape from 2D keypoints.
        
        Args:
            keypoints_2d: 2D keypoints array of shape (N, 2) where N is number of keypoints
            keypoint_scores: Confidence scores for each keypoint, shape (N,)
            image_shape: (height, width) of the input image
            initial_pose: Optional initial pose parameters for optimization
        
        Returns:
            Dictionary containing:
            - pose: Body pose parameters (joint rotations)
            - betas: Shape parameters
            - global_orient: Global rotation
            - transl: Global translation
            - joints_3d: 3D joint positions
            - vertices: 3D mesh vertices (if model available)
            - faces: Mesh faces (if model available)
        """
        if keypoints_2d is None or len(keypoints_2d) == 0:
            raise ValueError("keypoints_2d cannot be empty")
        
        # Map RTM Pose keypoints to SMPL-X joint structure
        # RTM Pose typically has 17 keypoints (COCO format)
        # SMPL-X has more joints including hands and face
        
        if self.model is not None:
            # Use full SMPL-X model for accurate estimation
            return self._estimate_with_model(
                keypoints_2d, keypoint_scores, image_shape, initial_pose
            )
        else:
            # Fallback: Simplified 3D estimation without full body model
            return self._estimate_simplified(
                keypoints_2d, keypoint_scores, image_shape
            )
    
    def _estimate_with_model(
        self,
        keypoints_2d: np.ndarray,
        keypoint_scores: Optional[np.ndarray],
        image_shape: Optional[Tuple[int, int]],
        initial_pose: Optional[Dict],
    ) -> Dict:
        """Estimate using full SMPL-X model (requires optimization)."""
        # This is a simplified version - in practice, you'd use optimization
        # to fit SMPL-X parameters to 2D keypoints (e.g., using SPIN, EFT, etc.)
        
        # For now, return a basic structure
        # In production, integrate with optimization frameworks like:
        # - SPIN (SMPL oPtimization IN the loop)
        # - EFT (End-to-end fitting)
        # - PyMAF (Pyramid Mesh Alignment Feedback)
        
        batch_size = 1
        device = self.device
        
        # Initialize parameters
        betas = torch.zeros(batch_size, self.num_betas, device=device)
        global_orient = torch.zeros(batch_size, 3, device=device)
        body_pose = torch.zeros(batch_size, 21 * 3, device=device)  # 21 joints * 3 (axis-angle)
        
        if self.use_hands:
            left_hand_pose = torch.zeros(batch_size, 15 * 3, device=device)
            right_hand_pose = torch.zeros(batch_size, 15 * 3, device=device)
        else:
            left_hand_pose = None
            right_hand_pose = None
        
        # Forward pass through model
        output = self.model(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
        )
        
        # Extract results
        vertices = output.vertices[0].detach().cpu().numpy()
        joints_3d = output.joints[0].detach().cpu().numpy()
        
        return {
            "pose": body_pose[0].detach().cpu().numpy(),
            "betas": betas[0].detach().cpu().numpy(),
            "global_orient": global_orient[0].detach().cpu().numpy(),
            "transl": np.array([0.0, 0.0, 0.0]),  # Would be estimated during optimization
            "joints_3d": joints_3d,
            "vertices": vertices,
            "faces": self.model.faces,
        }
    
    def _estimate_simplified(
        self,
        keypoints_2d: np.ndarray,
        keypoint_scores: Optional[np.ndarray],
        image_shape: Optional[Tuple[int, int]],
    ) -> Dict:
        """
        Simplified 3D estimation without full SMPL-X model.
        Uses basic geometric assumptions to create 3D skeleton.
        """
        # Map 2D keypoints to 3D using depth estimation
        # This is a placeholder - in practice, you'd use more sophisticated methods
        
        num_keypoints = len(keypoints_2d)
        
        # Estimate depth using keypoint relationships
        # Simple heuristic: use body proportions to estimate depth
        if image_shape:
            height, width = image_shape
            scale = max(height, width)
        else:
            scale = 1.0
        
        # Create 3D keypoints by estimating depth
        keypoints_3d = np.zeros((num_keypoints, 3))
        keypoints_3d[:, :2] = keypoints_2d
        
        # Simple depth estimation based on keypoint positions
        # Center of body (hip) is at depth 0
        if num_keypoints >= 17:  # COCO format
            # Use hip keypoints (11, 12 in COCO) as reference
            hip_indices = [11, 12] if num_keypoints >= 13 else [0]
            hip_center = np.mean(keypoints_2d[hip_indices], axis=0) if len(hip_indices) > 0 else keypoints_2d[0]
            
            # Estimate depth based on distance from hip center
            for i, kp in enumerate(keypoints_2d):
                dist = np.linalg.norm(kp - hip_center)
                # Normalize depth estimate
                depth = dist / scale * 0.5  # Arbitrary scaling factor
                keypoints_3d[i, 2] = depth
        
        # Create basic skeleton structure
        # This would map to SMPL-X joint structure
        joints_3d = keypoints_3d  # Simplified: use keypoints as joints
        
        return {
            "pose": np.zeros(63),  # 21 joints * 3 (axis-angle format)
            "betas": np.zeros(self.num_betas),
            "global_orient": np.zeros(3),
            "transl": np.array([0.0, 0.0, 0.0]),
            "joints_3d": joints_3d,
            "vertices": None,  # No mesh without model
            "faces": None,
        }
    
    def get_hand_joints(self, estimation_result: Dict) -> Dict[str, np.ndarray]:
        """
        Extract hand joint positions from estimation result.
        
        Args:
            estimation_result: Result from estimate_from_2d_keypoints
        
        Returns:
            Dictionary with 'left_hand' and 'right_hand' joint positions
        """
        joints_3d = estimation_result.get("joints_3d")
        if joints_3d is None:
            return {"left_hand": None, "right_hand": None}
        
        # In SMPL-X, hand joints are typically at specific indices
        # This is a simplified mapping - actual indices depend on model structure
        # Left wrist is typically around index 20, right wrist around 21
        # Hand joints follow after that
        
        # For simplified version, use wrist positions
        if len(joints_3d) >= 22:
            left_wrist = joints_3d[20] if len(joints_3d) > 20 else None
            right_wrist = joints_3d[21] if len(joints_3d) > 21 else None
        else:
            # Fallback: use last available joints or estimate
            left_wrist = joints_3d[-2] if len(joints_3d) >= 2 else None
            right_wrist = joints_3d[-1] if len(joints_3d) >= 1 else None
        
        return {
            "left_hand": left_wrist,
            "right_hand": right_wrist,
        }


def create_smpl_estimator(
    model_path: Optional[str] = None,
    device: str = "cpu",
) -> SMPLEstimator:
    """
    Factory function to create SMPL estimator instance.
    
    Args:
        model_path: Path to SMPL-X model files
        device: Device to run on
    
    Returns:
        SMPLEstimator instance
    """
    return SMPLEstimator(
        model_path=model_path,
        device=device,
    )

