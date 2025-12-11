"""
glTF Export Module
Exports 3D meshes and animations to glTF (.glb) format.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

logger = logging.getLogger(__name__)

try:
    import pygltflib
    PYGLTF_AVAILABLE = True
except ImportError:
    PYGLTF_AVAILABLE = False
    logger.warning("pygltflib not available. Install with: pip install pygltflib")

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    logger.warning("trimesh not available. Install with: pip install trimesh")


class GLTFExporter:
    """
    Exports 3D meshes and animations to glTF format (.glb or .gltf).
    """
    
    def __init__(self):
        """Initialize glTF exporter."""
        if not PYGLTF_AVAILABLE:
            logger.warning(
                "pygltflib not available. glTF export will use fallback method. "
                "Install with: pip install pygltflib"
            )
    
    def export_mesh(
        self,
        mesh_data: Dict,
        output_path: str,
        format: str = "glb",  # "glb" or "gltf"
        include_animation: bool = False,
    ) -> str:
        """
        Export a single mesh frame to glTF.
        
        Args:
            mesh_data: Mesh data dictionary from MeshGenerator
            output_path: Path to save the glTF file
            format: Export format ("glb" or "gltf")
            include_animation: Whether to include animation data
        
        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        
        if PYGLTF_AVAILABLE:
            return self._export_with_pygltf(mesh_data, output_path, format, include_animation)
        else:
            # Fallback: use trimesh if available
            if TRIMESH_AVAILABLE:
                return self._export_with_trimesh(mesh_data, output_path, format)
            else:
                raise ImportError(
                    "Neither pygltflib nor trimesh is available. "
                    "Install one of them: pip install pygltflib or pip install trimesh"
                )
    
    def export_animation(
        self,
        animation_data: Dict,
        output_path: str,
        format: str = "glb",
    ) -> str:
        """
        Export animation (multiple frames) to glTF.
        
        Args:
            animation_data: Animation data from MeshGenerator.create_animation
            output_path: Path to save the glTF file
            format: Export format ("glb" or "gltf")
        
        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        
        if PYGLTF_AVAILABLE:
            return self._export_animation_with_pygltf(animation_data, output_path, format)
        else:
            # Fallback: export first frame as static mesh
            logger.warning("pygltflib not available, exporting first frame only")
            if animation_data.get("frames"):
                return self.export_mesh(
                    animation_data["frames"][0],
                    output_path,
                    format,
                    include_animation=False,
                )
            else:
                raise ValueError("No frames in animation data")
    
    def _export_with_pygltf(
        self,
        mesh_data: Dict,
        output_path: Path,
        format: str,
        include_animation: bool,
    ) -> str:
        """Export using pygltflib."""
        vertices = mesh_data.get("vertices")
        faces = mesh_data.get("faces")
        
        if vertices is None:
            raise ValueError("No vertices in mesh data")
        
        # Create glTF structure
        gltf = pygltflib.GLTF2()
        scene = pygltflib.Scene()
        gltf.scenes.append(scene)
        gltf.scene = 0
        
        # Create mesh
        mesh = pygltflib.Mesh()
        primitive = pygltflib.Primitive()
        
        # Prepare vertex data
        vertices_bytes = vertices.astype(np.float32).tobytes()
        
        # Prepare index data
        if faces is not None:
            # Flatten faces for indexing
            indices = faces.flatten().astype(np.uint32)
            indices_bytes = indices.tobytes()
        else:
            indices = None
            indices_bytes = None
        
        # Create buffer
        buffer = pygltflib.Buffer()
        buffer_data = bytearray()
        
        # Add vertex buffer
        vertex_buffer_view = pygltflib.BufferView()
        vertex_buffer_view.buffer = 0
        vertex_buffer_view.byteOffset = len(buffer_data)
        vertex_buffer_view.byteLength = len(vertices_bytes)
        vertex_buffer_view.target = pygltflib.ARRAY_BUFFER
        buffer_data.extend(vertices_bytes)
        gltf.bufferViews.append(vertex_buffer_view)
        
        # Add index buffer if available
        if indices_bytes:
            index_buffer_view = pygltflib.BufferView()
            index_buffer_view.buffer = 0
            index_buffer_view.byteOffset = len(buffer_data)
            index_buffer_view.byteLength = len(indices_bytes)
            index_buffer_view.target = pygltflib.ELEMENT_ARRAY_BUFFER
            buffer_data.extend(indices_bytes)
            gltf.bufferViews.append(index_buffer_view)
        
        # Create accessors
        vertex_accessor = pygltflib.Accessor()
        vertex_accessor.bufferView = 0
        vertex_accessor.componentType = pygltflib.FLOAT
        vertex_accessor.count = len(vertices)
        vertex_accessor.type = pygltflib.VEC3
        vertex_accessor.min = vertices.min(axis=0).tolist()
        vertex_accessor.max = vertices.max(axis=0).tolist()
        gltf.accessors.append(vertex_accessor)
        
        if indices_bytes:
            index_accessor = pygltflib.Accessor()
            index_accessor.bufferView = 1
            index_accessor.componentType = pygltflib.UNSIGNED_INT
            index_accessor.count = len(indices)
            index_accessor.type = pygltflib.SCALAR
            gltf.accessors.append(index_accessor)
            primitive.indices = 1
        
        # Set primitive attributes
        primitive.attributes.POSITION = 0
        mesh.primitives.append(primitive)
        gltf.meshes.append(mesh)
        
        # Create node
        node = pygltflib.Node()
        node.mesh = 0
        gltf.nodes.append(node)
        
        # Add node to scene
        scene.nodes.append(0)
        
        # Set buffer
        buffer.uri = "data:application/octet-stream;base64," + pygltflib.base64_encode(buffer_data)
        buffer.byteLength = len(buffer_data)
        gltf.buffers.append(buffer)
        
        # Export
        if format == "glb":
            output_path = output_path.with_suffix(".glb")
            gltf.save_binary(str(output_path))
        else:
            output_path = output_path.with_suffix(".gltf")
            gltf.save(str(output_path))
        
        logger.info(f"Exported mesh to {output_path}")
        return str(output_path)
    
    def _export_animation_with_pygltf(
        self,
        animation_data: Dict,
        output_path: Path,
        format: str,
    ) -> str:
        """Export animation using pygltflib."""
        frames = animation_data.get("frames", [])
        fps = animation_data.get("fps", 30.0)
        
        if not frames:
            raise ValueError("No frames in animation data")
        
        # For now, export as a sequence or use first frame
        # Full animation support would require creating samplers and channels
        # This is a simplified version
        
        # Export first frame as base mesh
        base_mesh = frames[0]
        output_path = output_path.with_suffix(".glb")
        
        # Create glTF with multiple meshes (one per frame)
        # In production, you'd create proper animation channels
        gltf = pygltflib.GLTF2()
        scene = pygltflib.Scene()
        gltf.scenes.append(scene)
        gltf.scene = 0
        
        # Export first frame as main mesh
        # Additional frames could be added as separate nodes or animation
        vertices = base_mesh.get("vertices")
        faces = base_mesh.get("faces")
        
        if vertices is None:
            raise ValueError("No vertices in first frame")
        
        # Similar to _export_with_pygltf but with animation structure
        # This is simplified - full implementation would include:
        # - Animation samplers for keyframe data
        # - Animation channels linking to node transforms
        # - Proper timing and interpolation
        
        # For now, export static mesh
        logger.info(f"Exporting animation with {len(frames)} frames (simplified)")
        return self._export_with_pygltf(base_mesh, output_path, format, include_animation=False)
    
    def _export_with_trimesh(
        self,
        mesh_data: Dict,
        output_path: Path,
        format: str,
    ) -> str:
        """Export using trimesh (fallback)."""
        if not TRIMESH_AVAILABLE:
            raise ImportError("trimesh not available")
        
        vertices = mesh_data.get("vertices")
        faces = mesh_data.get("faces")
        
        if vertices is None:
            raise ValueError("No vertices in mesh data")
        
        # Create trimesh object
        if faces is not None:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        else:
            # Create point cloud
            mesh = trimesh.PointCloud(vertices=vertices)
        
        # Export
        if format == "glb":
            output_path = output_path.with_suffix(".glb")
            mesh.export(str(output_path), file_type="glb")
        else:
            output_path = output_path.with_suffix(".gltf")
            mesh.export(str(output_path), file_type="gltf")
        
        logger.info(f"Exported mesh to {output_path} using trimesh")
        return str(output_path)


def create_gltf_exporter() -> GLTFExporter:
    """Factory function to create glTF exporter."""
    return GLTFExporter()

