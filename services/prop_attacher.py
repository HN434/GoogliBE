"""
Prop Attachment Module
Attaches 3D props (bat, bat-handle, bat model) to hand joints.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    logger.warning("trimesh not available. Install with: pip install trimesh")


class PropAttacher:
    """
    Attaches 3D props (like cricket bat) to hand joints in the mesh.
    """
    
    def __init__(self, props_dir: Optional[str] = None):
        """
        Initialize prop attacher.
        
        Args:
            props_dir: Directory containing 3D prop models (e.g., .obj, .glb files)
        """
        self.props_dir = Path(props_dir) if props_dir else Path("models/props")
        self.loaded_props = {}
    
    def load_prop(self, prop_name: str, prop_path: Optional[str] = None) -> Optional[Dict]:
        """
        Load a 3D prop model.
        
        Args:
            prop_name: Name of the prop (e.g., "bat", "bat_handle")
            prop_path: Optional path to prop file. If None, searches in props_dir
        
        Returns:
            Dictionary containing prop mesh data or None if not found
        """
        if prop_name in self.loaded_props:
            return self.loaded_props[prop_name]
        
        # Try to find prop file
        if prop_path:
            prop_file = Path(prop_path)
        else:
            # Search in props directory
            possible_names = [
                f"{prop_name}.obj",
                f"{prop_name}.glb",
                f"{prop_name}.gltf",
                f"{prop_name}.ply",
            ]
            prop_file = None
            for name in possible_names:
                candidate = self.props_dir / name
                if candidate.exists():
                    prop_file = candidate
                    break
        
        if prop_file is None or not prop_file.exists():
            logger.warning(f"Prop file not found for {prop_name}. Creating placeholder.")
            return self._create_placeholder_prop(prop_name)
        
        # Load prop mesh
        try:
            if TRIMESH_AVAILABLE:
                mesh = trimesh.load(str(prop_file))
                if isinstance(mesh, trimesh.Scene):
                    # Extract first mesh from scene
                    mesh = list(mesh.geometry.values())[0]
                
                prop_data = {
                    "vertices": np.array(mesh.vertices),
                    "faces": np.array(mesh.faces),
                    "name": prop_name,
                    "path": str(prop_file),
                }
                
                # Calculate bounding box and center
                bounds = mesh.bounds
                center = mesh.centroid
                prop_data["bounds"] = bounds
                prop_data["center"] = center
                
                self.loaded_props[prop_name] = prop_data
                logger.info(f"Loaded prop: {prop_name} from {prop_file}")
                return prop_data
            else:
                logger.warning("trimesh not available, creating placeholder prop")
                return self._create_placeholder_prop(prop_name)
        except Exception as e:
            logger.error(f"Failed to load prop {prop_name}: {e}")
            return self._create_placeholder_prop(prop_name)
    
    def _create_placeholder_prop(self, prop_name: str) -> Dict:
        """
        Create a placeholder prop (simple geometric shape) when actual model is not available.
        
        Args:
            prop_name: Name of the prop
        
        Returns:
            Dictionary with placeholder prop data
        """
        if "bat" in prop_name.lower():
            # Create a simple bat shape (rectangular prism)
            # Cricket bat dimensions (approximate, in meters)
            length = 0.96  # ~38 inches
            width = 0.108  # ~4.25 inches
            thickness = 0.038  # ~1.5 inches
            
            # Create vertices for a rectangular bat
            vertices = np.array([
                # Handle (narrower)
                [-thickness/2, -length/2, -width/4],
                [thickness/2, -length/2, -width/4],
                [thickness/2, -length/2, width/4],
                [-thickness/2, -length/2, width/4],
                [-thickness/2, -length/4, -width/4],
                [thickness/2, -length/4, -width/4],
                [thickness/2, -length/4, width/4],
                [-thickness/2, -length/4, width/4],
                # Blade (wider)
                [-thickness/2, -length/4, -width/2],
                [thickness/2, -length/4, -width/2],
                [thickness/2, -length/4, width/2],
                [-thickness/2, -length/4, width/2],
                [-thickness/2, length/2, -width/2],
                [thickness/2, length/2, -width/2],
                [thickness/2, length/2, width/2],
                [-thickness/2, length/2, width/2],
            ])
            
            # Create faces (simplified - would need proper triangulation)
            faces = np.array([
                # Handle faces
                [0, 1, 2], [0, 2, 3],
                [4, 5, 6], [4, 6, 7],
                [0, 4, 5], [0, 5, 1],
                [3, 7, 6], [3, 6, 2],
                # Blade faces
                [8, 9, 10], [8, 10, 11],
                [12, 13, 14], [12, 14, 15],
                [8, 12, 13], [8, 13, 9],
                [11, 15, 14], [11, 14, 10],
            ])
        else:
            # Generic placeholder (cube)
            vertices = np.array([
                [-0.05, -0.05, -0.05],
                [0.05, -0.05, -0.05],
                [0.05, 0.05, -0.05],
                [-0.05, 0.05, -0.05],
                [-0.05, -0.05, 0.05],
                [0.05, -0.05, 0.05],
                [0.05, 0.05, 0.05],
                [-0.05, 0.05, 0.05],
            ])
            faces = np.array([
                [0, 1, 2], [0, 2, 3],
                [4, 7, 6], [4, 6, 5],
                [0, 4, 5], [0, 5, 1],
                [2, 6, 7], [2, 7, 3],
                [0, 3, 7], [0, 7, 4],
                [1, 5, 6], [1, 6, 2],
            ])
        
        prop_data = {
            "vertices": vertices,
            "faces": faces,
            "name": prop_name,
            "path": None,  # Placeholder
            "bounds": np.array([
                vertices.min(axis=0),
                vertices.max(axis=0),
            ]),
            "center": vertices.mean(axis=0),
        }
        
        self.loaded_props[prop_name] = prop_data
        return prop_data
    
    def attach_prop_to_hand(
        self,
        mesh_data: Dict,
        hand_joint: np.ndarray,
        prop_name: str = "bat",
        hand_side: str = "right",  # "left" or "right"
        prop_rotation: Optional[np.ndarray] = None,
        prop_scale: float = 1.0,
    ) -> Dict:
        """
        Attach a prop to a hand joint.
        
        Args:
            mesh_data: Mesh data dictionary from MeshGenerator
            hand_joint: 3D position of hand joint (wrist or hand)
            prop_name: Name of prop to attach
            hand_side: Which hand ("left" or "right")
            prop_rotation: Optional rotation matrix or euler angles for prop
            prop_scale: Scale factor for prop
        
        Returns:
            Updated mesh data with attached prop
        """
        # Load prop
        prop = self.load_prop(prop_name)
        if prop is None:
            logger.warning(f"Could not load prop {prop_name}")
            return mesh_data
        
        # Get prop vertices and faces
        prop_vertices = prop["vertices"].copy()
        prop_faces = prop["faces"].copy()
        
        # Center prop at origin
        prop_center = prop.get("center", prop_vertices.mean(axis=0))
        prop_vertices = prop_vertices - prop_center
        
        # Apply rotation if provided
        if prop_rotation is not None:
            if prop_rotation.shape == (3, 3):
                # Rotation matrix
                R = prop_rotation
            elif prop_rotation.shape == (3,):
                # Euler angles
                from scipy.spatial.transform import Rotation
                R = Rotation.from_euler('xyz', prop_rotation).as_matrix()
            else:
                R = np.eye(3)
            
            prop_vertices = (R @ prop_vertices.T).T
        else:
            # Default rotation: align prop along hand direction
            # For bat, typically held vertically or at angle
            # This is simplified - would need hand orientation estimation
            R = np.eye(3)
        
        # Apply scale
        prop_vertices = prop_vertices * prop_scale
        
        # Translate to hand joint position
        prop_vertices = prop_vertices + hand_joint
        
        # Merge with main mesh
        main_vertices = mesh_data.get("vertices")
        main_faces = mesh_data.get("faces")
        
        if main_vertices is not None:
            # Offset prop faces by number of main vertices
            prop_faces_offset = prop_faces + len(main_vertices)
            
            # Combine vertices
            combined_vertices = np.vstack([main_vertices, prop_vertices])
            
            # Combine faces
            if main_faces is not None:
                combined_faces = np.vstack([main_faces, prop_faces_offset])
            else:
                combined_faces = prop_faces_offset
            
            # Update mesh data
            mesh_data["vertices"] = combined_vertices
            mesh_data["faces"] = combined_faces
            
            # Store prop attachment info
            if "attached_props" not in mesh_data:
                mesh_data["attached_props"] = []
            
            mesh_data["attached_props"].append({
                "name": prop_name,
                "hand_side": hand_side,
                "attachment_point": hand_joint.tolist(),
                "vertex_start": len(main_vertices),
                "vertex_count": len(prop_vertices),
                "face_start": len(main_faces) if main_faces is not None else 0,
                "face_count": len(prop_faces),
            })
        else:
            # No main mesh, just use prop
            mesh_data["vertices"] = prop_vertices
            mesh_data["faces"] = prop_faces
        
        return mesh_data
    
    def attach_bat_to_hands(
        self,
        mesh_data: Dict,
        hand_joints: Dict[str, np.ndarray],
        bat_name: str = "bat",
    ) -> Dict:
        """
        Attach bat to both hands (if both are available) or single hand.
        
        Args:
            mesh_data: Mesh data dictionary
            hand_joints: Dictionary with 'left_hand' and/or 'right_hand' positions
            bat_name: Name of bat prop
        
        Returns:
            Updated mesh data with bat attached
        """
        # Prefer right hand (typically dominant hand for batting)
        if "right_hand" in hand_joints and hand_joints["right_hand"] is not None:
            mesh_data = self.attach_prop_to_hand(
                mesh_data,
                hand_joints["right_hand"],
                prop_name=bat_name,
                hand_side="right",
            )
        elif "left_hand" in hand_joints and hand_joints["left_hand"] is not None:
            mesh_data = self.attach_prop_to_hand(
                mesh_data,
                hand_joints["left_hand"],
                prop_name=bat_name,
                hand_side="left",
            )
        
        return mesh_data


def create_prop_attacher(props_dir: Optional[str] = None) -> PropAttacher:
    """Factory function to create prop attacher."""
    return PropAttacher(props_dir=props_dir)

