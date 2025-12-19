"""
Test script to convert MediaPipe pose detection to 3D skeleton GLB file.

This script:
1. Processes a video using MediaPipe Pose
2. Extracts 3D world landmarks (already in 3D from MediaPipe)
3. Creates a skeleton structure with bone connections
4. Exports to GLB format
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import List, Tuple, Optional
import argparse
import sys

# Try to import GLB export libraries
try:
    import pygltflib
    PYGLTF_AVAILABLE = True
except ImportError:
    PYGLTF_AVAILABLE = False
    print("Warning: pygltflib not available. Trying trimesh...")
    try:
        import trimesh
        TRIMESH_AVAILABLE = True
    except ImportError:
        TRIMESH_AVAILABLE = False
        print("Error: Neither pygltflib nor trimesh is available.")
        print("Install one with: pip install pygltflib or pip install trimesh")
        sys.exit(1)


# MediaPipe Pose landmark indices
class MediaPipeLandmarks:
    """MediaPipe Pose 33 landmark indices"""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# MediaPipe Pose bone connections (pairs of landmark indices)
# These define the skeleton structure
BONE_CONNECTIONS = [
    # Face
    (MediaPipeLandmarks.NOSE, MediaPipeLandmarks.LEFT_EYE_INNER),
    (MediaPipeLandmarks.LEFT_EYE_INNER, MediaPipeLandmarks.LEFT_EYE),
    (MediaPipeLandmarks.LEFT_EYE, MediaPipeLandmarks.LEFT_EYE_OUTER),
    (MediaPipeLandmarks.LEFT_EYE_OUTER, MediaPipeLandmarks.LEFT_EAR),
    (MediaPipeLandmarks.NOSE, MediaPipeLandmarks.RIGHT_EYE_INNER),
    (MediaPipeLandmarks.RIGHT_EYE_INNER, MediaPipeLandmarks.RIGHT_EYE),
    (MediaPipeLandmarks.RIGHT_EYE, MediaPipeLandmarks.RIGHT_EYE_OUTER),
    (MediaPipeLandmarks.RIGHT_EYE_OUTER, MediaPipeLandmarks.RIGHT_EAR),
    (MediaPipeLandmarks.MOUTH_LEFT, MediaPipeLandmarks.MOUTH_RIGHT),
    
    # Upper body - left arm
    (MediaPipeLandmarks.LEFT_SHOULDER, MediaPipeLandmarks.LEFT_ELBOW),
    (MediaPipeLandmarks.LEFT_ELBOW, MediaPipeLandmarks.LEFT_WRIST),
    (MediaPipeLandmarks.LEFT_WRIST, MediaPipeLandmarks.LEFT_INDEX),
    (MediaPipeLandmarks.LEFT_WRIST, MediaPipeLandmarks.LEFT_PINKY),
    (MediaPipeLandmarks.LEFT_WRIST, MediaPipeLandmarks.LEFT_THUMB),
    
    # Upper body - right arm
    (MediaPipeLandmarks.RIGHT_SHOULDER, MediaPipeLandmarks.RIGHT_ELBOW),
    (MediaPipeLandmarks.RIGHT_ELBOW, MediaPipeLandmarks.RIGHT_WRIST),
    (MediaPipeLandmarks.RIGHT_WRIST, MediaPipeLandmarks.RIGHT_INDEX),
    (MediaPipeLandmarks.RIGHT_WRIST, MediaPipeLandmarks.RIGHT_PINKY),
    (MediaPipeLandmarks.RIGHT_WRIST, MediaPipeLandmarks.RIGHT_THUMB),
    
    # Torso
    (MediaPipeLandmarks.LEFT_SHOULDER, MediaPipeLandmarks.RIGHT_SHOULDER),
    (MediaPipeLandmarks.LEFT_SHOULDER, MediaPipeLandmarks.LEFT_HIP),
    (MediaPipeLandmarks.RIGHT_SHOULDER, MediaPipeLandmarks.RIGHT_HIP),
    (MediaPipeLandmarks.LEFT_HIP, MediaPipeLandmarks.RIGHT_HIP),
    
    # Lower body - left leg
    (MediaPipeLandmarks.LEFT_HIP, MediaPipeLandmarks.LEFT_KNEE),
    (MediaPipeLandmarks.LEFT_KNEE, MediaPipeLandmarks.LEFT_ANKLE),
    (MediaPipeLandmarks.LEFT_ANKLE, MediaPipeLandmarks.LEFT_HEEL),
    (MediaPipeLandmarks.LEFT_ANKLE, MediaPipeLandmarks.LEFT_FOOT_INDEX),
    
    # Lower body - right leg
    (MediaPipeLandmarks.RIGHT_HIP, MediaPipeLandmarks.RIGHT_KNEE),
    (MediaPipeLandmarks.RIGHT_KNEE, MediaPipeLandmarks.RIGHT_ANKLE),
    (MediaPipeLandmarks.RIGHT_ANKLE, MediaPipeLandmarks.RIGHT_HEEL),
    (MediaPipeLandmarks.RIGHT_ANKLE, MediaPipeLandmarks.RIGHT_FOOT_INDEX),
]


def extract_3d_pose_from_video(
    video_path: str,
    max_frames: Optional[int] = None,
    min_visibility: float = 0.5
) -> List[np.ndarray]:
    """
    Extract 3D pose landmarks from video using MediaPipe.
    
    Args:
        video_path: Path to input video file
        max_frames: Maximum number of frames to process (None for all)
        min_visibility: Minimum visibility threshold for landmarks
    
    Returns:
        List of 3D landmark arrays, one per frame (shape: [33, 3])
    """
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,  # Highest accuracy
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    all_poses = []
    frame_count = 0
    
    print(f"Processing video: {video_path}")
    
    try:
        while True:
            if max_frames and frame_count >= max_frames:
                break
                
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = pose.process(image_rgb)
            
            if results.pose_world_landmarks:
                # Extract 3D world landmarks
                landmarks_3d = []
                for landmark in results.pose_world_landmarks.landmark:
                    # MediaPipe world landmarks are in meters, relative to hip
                    if landmark.visibility >= min_visibility:
                        landmarks_3d.append([
                            float(landmark.x),
                            float(landmark.y),
                            float(landmark.z)
                        ])
                    else:
                        # Use NaN for low visibility landmarks
                        landmarks_3d.append([np.nan, np.nan, np.nan])
                
                all_poses.append(np.array(landmarks_3d, dtype=np.float32))
            else:
                # No pose detected, add NaN array
                all_poses.append(np.full((33, 3), np.nan, dtype=np.float32))
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"  Processed {frame_count} frames...")
    
    finally:
        cap.release()
        pose.close()
    
    print(f"Extracted {len(all_poses)} frames with pose data")
    return all_poses


def create_skeleton_mesh(
    landmarks_3d: np.ndarray,
    bone_connections: List[Tuple[int, int]],
    bone_radius: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a mesh representation of the skeleton from 3D landmarks.
    
    Args:
        landmarks_3d: 3D landmark positions (33, 3)
        bone_connections: List of (start_idx, end_idx) tuples
        bone_radius: Radius of bone cylinders
    
    Returns:
        vertices: Mesh vertices (N, 3)
        faces: Mesh faces (M, 3)
    """
    vertices = []
    faces = []
    vertex_offset = 0
    
    # Add joint spheres
    joint_radius = bone_radius * 1.5
    joint_segments = 8
    
    for i, landmark in enumerate(landmarks_3d):
        if np.any(np.isnan(landmark)):
            continue
        
        # Create sphere for joint
        center = landmark
        for u in range(joint_segments + 1):
            for v in range(joint_segments + 1):
                u_angle = u * np.pi / joint_segments
                v_angle = v * 2 * np.pi / joint_segments
                
                x = center[0] + joint_radius * np.sin(u_angle) * np.cos(v_angle)
                y = center[1] + joint_radius * np.sin(u_angle) * np.sin(v_angle)
                z = center[2] + joint_radius * np.cos(u_angle)
                
                vertices.append([x, y, z])
        
        # Create faces for sphere (simplified - would need proper triangulation)
        # For now, we'll just use the vertices and create bones
    
    # Add bone cylinders
    for start_idx, end_idx in bone_connections:
        if start_idx >= len(landmarks_3d) or end_idx >= len(landmarks_3d):
            continue
        
        start = landmarks_3d[start_idx]
        end = landmarks_3d[end_idx]
        
        if np.any(np.isnan(start)) or np.any(np.isnan(end)):
            continue
        
        # Create cylinder between start and end
        direction = end - start
        length = np.linalg.norm(direction)
        
        if length < 1e-6:
            continue
        
        direction = direction / length
        
        # Create cylinder vertices (simplified - 8 segments)
        cylinder_segments = 8
        for i in range(cylinder_segments):
            angle = 2 * np.pi * i / cylinder_segments
            
            # Perpendicular vectors
            if abs(direction[2]) < 0.9:
                perp1 = np.array([-direction[1], direction[0], 0])
            else:
                perp1 = np.array([0, -direction[2], direction[1]])
            
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(direction, perp1)
            
            # Create circle vertices at start and end
            for t in [0, 1]:
                pos = start + t * direction * length
                offset = bone_radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
                vertices.append((pos + offset).tolist())
    
    vertices = np.array(vertices, dtype=np.float32)
    
    # Create simple faces (this is a simplified version)
    # For a proper implementation, you'd need proper triangulation
    # For now, we'll export as a point cloud or use a simpler approach
    
    return vertices, None  # Return None for faces to use point cloud


def export_skeleton_to_glb_pygltf(
    landmarks_3d: np.ndarray,
    output_path: Path,
    bone_connections: List[Tuple[int, int]],
    frame_index: int = 0
) -> str:
    """
    Export 3D skeleton to GLB using pygltflib.
    
    Args:
        landmarks_3d: 3D landmark positions (33, 3)
        output_path: Output file path
        bone_connections: List of bone connections
        frame_index: Frame index for naming
    
    Returns:
        Path to exported GLB file
    """
    if not PYGLTF_AVAILABLE:
        raise ImportError("pygltflib not available")
    
    # Filter out NaN landmarks
    valid_landmarks = []
    valid_indices = []
    for i, landmark in enumerate(landmarks_3d):
        if not np.any(np.isnan(landmark)):
            valid_landmarks.append(landmark)
            valid_indices.append(i)
    
    if len(valid_landmarks) < 2:
        raise ValueError("Not enough valid landmarks to create skeleton")
    
    valid_landmarks = np.array(valid_landmarks, dtype=np.float32)
    
    # Create mapping from original indices to valid indices
    idx_map = {orig: new for new, orig in enumerate(valid_indices)}
    
    # Use joint positions as vertices
    vertices = valid_landmarks
    
    # Create bone connections as line indices
    bone_indices = []
    for start_idx, end_idx in bone_connections:
        if start_idx in idx_map and end_idx in idx_map:
            start_vertex = idx_map[start_idx]
            end_vertex = idx_map[end_idx]
            
            # Add line indices (using existing joint vertices)
            bone_indices.append([start_vertex, end_vertex])
    
    # Create glTF structure
    gltf = pygltflib.GLTF2()
    scene = pygltflib.Scene()
    gltf.scenes.append(scene)
    gltf.scene = 0
    
    # Prepare vertex data
    vertices_bytes = vertices.astype(np.float32).tobytes()
    
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
    
    # Create accessor for vertices
    vertex_accessor = pygltflib.Accessor()
    vertex_accessor.bufferView = 0
    vertex_accessor.componentType = pygltflib.FLOAT
    vertex_accessor.count = len(vertices)
    vertex_accessor.type = pygltflib.VEC3
    vertex_accessor.min = vertices.min(axis=0).tolist()
    vertex_accessor.max = vertices.max(axis=0).tolist()
    gltf.accessors.append(vertex_accessor)
    
    # Create mesh with points primitive (for joints)
    mesh = pygltflib.Mesh()
    
    # Add points primitive for joints
    points_primitive = pygltflib.Primitive()
    points_primitive.attributes.POSITION = 0
    points_primitive.mode = pygltflib.POINTS
    mesh.primitives.append(points_primitive)
    
    # Add lines primitive for bones
    if bone_indices:
        # Flatten bone indices (each bone is a pair of vertices)
        line_indices = np.array(bone_indices, dtype=np.uint32).flatten()
        indices_bytes = line_indices.tobytes()
        
        # Add index buffer
        index_buffer_view = pygltflib.BufferView()
        index_buffer_view.buffer = 0
        index_buffer_view.byteOffset = len(buffer_data)
        index_buffer_view.byteLength = len(indices_bytes)
        index_buffer_view.target = pygltflib.ELEMENT_ARRAY_BUFFER
        buffer_data.extend(indices_bytes)
        gltf.bufferViews.append(index_buffer_view)
        
        # Create index accessor
        index_accessor = pygltflib.Accessor()
        index_accessor.bufferView = 1
        index_accessor.componentType = pygltflib.UNSIGNED_INT
        index_accessor.count = len(indices)
        index_accessor.type = pygltflib.SCALAR
        gltf.accessors.append(index_accessor)
        
        # Add lines primitive
        lines_primitive = pygltflib.Primitive()
        lines_primitive.attributes.POSITION = 0
        lines_primitive.indices = 1
        lines_primitive.mode = pygltflib.LINES
        mesh.primitives.append(lines_primitive)
    
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
    output_path = output_path.with_suffix(".glb")
    gltf.save_binary(str(output_path))
    
    print(f"Exported skeleton to {output_path}")
    return str(output_path)


def export_skeleton_to_glb_trimesh(
    landmarks_3d: np.ndarray,
    output_path: Path,
    bone_connections: List[Tuple[int, int]]
) -> str:
    """
    Export 3D skeleton to GLB using trimesh (fallback).
    
    Args:
        landmarks_3d: 3D landmark positions (33, 3)
        output_path: Output file path
        bone_connections: List of bone connections
    
    Returns:
        Path to exported GLB file
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh not available")
    
    # Filter out NaN landmarks
    valid_landmarks = []
    valid_indices = []
    for i, landmark in enumerate(landmarks_3d):
        if not np.any(np.isnan(landmark)):
            valid_landmarks.append(landmark)
            valid_indices.append(i)
    
    if len(valid_landmarks) < 2:
        raise ValueError("Not enough valid landmarks to create skeleton")
    
    valid_landmarks = np.array(valid_landmarks, dtype=np.float32)
    
    # Create mapping
    idx_map = {orig: new for new, orig in enumerate(valid_indices)}
    
    # Create line segments for bones
    lines = []
    for start_idx, end_idx in bone_connections:
        if start_idx in idx_map and end_idx in idx_map:
            start_vertex = valid_landmarks[idx_map[start_idx]]
            end_vertex = valid_landmarks[idx_map[end_idx]]
            lines.append([start_vertex, end_vertex])
    
    if not lines:
        raise ValueError("No valid bone connections found")
    
    # Create trimesh line collection
    line_collection = trimesh.load_path(lines)
    
    # Export
    output_path = output_path.with_suffix(".glb")
    line_collection.export(str(output_path), file_type="glb")
    
    print(f"Exported skeleton to {output_path} using trimesh")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert MediaPipe pose detection to 3D skeleton GLB file"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to input video file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output GLB file path (default: <video_name>_skeleton.glb)"
    )
    parser.add_argument(
        "-f", "--frame",
        type=int,
        default=0,
        help="Frame index to export (default: 0, first frame with pose)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process"
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.5,
        help="Minimum visibility threshold for landmarks (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    # Validate input
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = video_path.parent / f"{video_path.stem}_skeleton.glb"
    
    # Extract 3D poses from video
    print("Extracting 3D poses from video...")
    all_poses = extract_3d_pose_from_video(
        str(video_path),
        max_frames=args.max_frames,
        min_visibility=args.min_visibility
    )
    
    if not all_poses:
        print("Error: No poses detected in video")
        sys.exit(1)
    
    # Find first frame with valid pose
    selected_frame = args.frame
    if selected_frame >= len(all_poses):
        selected_frame = 0
        print(f"Warning: Frame {args.frame} out of range, using frame 0")
    
    # Find frame with valid pose
    for i in range(selected_frame, len(all_poses)):
        if not np.all(np.isnan(all_poses[i])):
            selected_frame = i
            break
    else:
        # If no valid pose found, try from beginning
        for i in range(len(all_poses)):
            if not np.all(np.isnan(all_poses[i])):
                selected_frame = i
                break
        else:
            print("Error: No valid poses found in video")
            sys.exit(1)
    
    landmarks_3d = all_poses[selected_frame]
    print(f"Using frame {selected_frame} for export")
    
    # Export to GLB
    print(f"Exporting skeleton to {output_path}...")
    try:
        if PYGLTF_AVAILABLE:
            export_skeleton_to_glb_pygltf(
                landmarks_3d,
                output_path,
                BONE_CONNECTIONS,
                frame_index=selected_frame
            )
        elif TRIMESH_AVAILABLE:
            export_skeleton_to_glb_trimesh(
                landmarks_3d,
                output_path,
                BONE_CONNECTIONS
            )
        else:
            print("Error: No GLB export library available")
            sys.exit(1)
        
        print(f"\n✅ Successfully exported skeleton to: {output_path}")
        print(f"   Frame: {selected_frame}")
        print(f"   Valid landmarks: {np.sum(~np.any(np.isnan(landmarks_3d), axis=1))}/33")
        
    except Exception as e:
        print(f"Error exporting skeleton: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

