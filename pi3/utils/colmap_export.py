# Copyright (c) Meta Platforms, Inc. and affiliates.
# This implementation is inspired by VGGT's np_to_pycolmap.py
# Reference: https://github.com/facebookresearch/vggt/blob/main/vggt/dependency/np_to_pycolmap.py
# 
# Note: This implementation is adapted for pycolmap 3.13+ API

import os
import numpy as np
import torch

try:
    import pycolmap
    PYCOLMAP_AVAILABLE = True
except ImportError:
    PYCOLMAP_AVAILABLE = False


def pi3_to_colmap(
    points3d,
    camera_poses,
    images,
    conf=None,
    image_size=None,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
    conf_threshold=0.1,
    min_track_length=2,
    points_rgb=None,
):
    """
    Convert Pi3 output to COLMAP reconstruction format.
    
    Args:
        points3d (torch.Tensor or np.ndarray): Global 3D points, shape (B, N, H, W, 3) or (N, H, W, 3)
        camera_poses (torch.Tensor or np.ndarray): Camera-to-world transformation matrices (4x4 in OpenCV format), 
                                                    shape (B, N, 4, 4) or (N, 4, 4)
        images (torch.Tensor or np.ndarray): RGB images, shape (B, N, 3, H, W) or (N, 3, H, W)
        conf (torch.Tensor or np.ndarray, optional): Confidence scores, shape (B, N, H, W, 1) or (N, H, W, 1)
        image_size (tuple, optional): (width, height) of images. If None, inferred from images tensor
        shared_camera (bool): Whether all frames share the same camera intrinsics
        camera_type (str): COLMAP camera model type ("SIMPLE_PINHOLE", "PINHOLE")
        conf_threshold (float): Confidence threshold for filtering points
        min_track_length (int): Minimum number of views for a valid 3D point
        points_rgb (np.ndarray, optional): RGB colors for points, shape matching points3d
    
    Returns:
        reconstruction (pycolmap.Reconstruction): COLMAP reconstruction object
        intrinsics (np.ndarray): Estimated camera intrinsics for each frame, shape (N, 3, 3)
    """
    if not PYCOLMAP_AVAILABLE:
        raise ImportError("pycolmap is required for COLMAP export. Install it with: pip install pycolmap")
    
    # Convert tensors to numpy if needed
    if torch.is_tensor(points3d):
        points3d = points3d.detach().cpu().numpy()
    if torch.is_tensor(camera_poses):
        camera_poses = camera_poses.detach().cpu().numpy()
    if torch.is_tensor(images):
        images = images.detach().cpu().numpy()
    if conf is not None and torch.is_tensor(conf):
        conf = conf.detach().cpu().numpy()
    
    # Handle batch dimension
    if points3d.ndim == 5:
        points3d = points3d[0]  # (N, H, W, 3)
    if camera_poses.ndim == 3:
        camera_poses = camera_poses[0]  # (N, 4, 4)
    if images.ndim == 5:
        images = images[0]  # (N, 3, H, W)
    if conf is not None and conf.ndim == 5:
        conf = conf[0]  # (N, H, W, 1)
    
    N, H, W, _ = points3d.shape
    
    # Infer image size if not provided
    if image_size is None:
        image_size = np.array([W, H], dtype=np.int32)
    else:
        image_size = np.array(image_size, dtype=np.int32)
    
    # Estimate camera intrinsics from image size
    # Using a simple heuristic: focal length = max(W, H) * 1.2
    intrinsics = _estimate_intrinsics(image_size, N, camera_type)
    
    # Apply confidence filtering if provided
    if conf is not None:
        # Flatten points and get valid mask
        conf_squeezed = conf.squeeze(-1)  # (N, H, W)
        valid_mask = conf_squeezed > conf_threshold
    else:
        valid_mask = np.ones((N, H, W), dtype=bool)
    
    # Convert Pi3 output to COLMAP format
    # We need to create tracks (2D-3D correspondences)
    # Each 3D point corresponds to pixel locations across multiple views
    
    # Create pixel coordinates
    u_coords, v_coords = np.meshgrid(np.arange(W), np.arange(H))
    
    # Build tracks: for each valid 3D point, find its 2D projections
    reconstruction = pycolmap.Reconstruction()
    
    # Flatten the points for easier processing
    points3d_flat = points3d.reshape(N, -1, 3)  # (N, H*W, 3)
    valid_mask_flat = valid_mask.reshape(N, -1)  # (N, H*W)
    
    # Get RGB colors if not provided
    if points_rgb is None and images is not None:
        # images shape: (N, 3, H, W) -> (N, H, W, 3)
        images_hwc = np.transpose(images, (0, 2, 3, 1))
        # Normalize to 0-255 range
        if images_hwc.max() <= 1.0:
            images_hwc = (images_hwc * 255).astype(np.uint8)
        points_rgb = images_hwc.reshape(N, -1, 3)  # (N, H*W, 3)
    
    # Find valid points that appear in multiple views
    point_validity = valid_mask_flat.sum(axis=0)  # (H*W,)
    valid_point_indices = np.where(point_validity >= min_track_length)[0]
    
    # Add 3D points to reconstruction
    point_id_map = {}
    for idx, point_idx in enumerate(valid_point_indices):
        # Use the mean of 3D points across valid views
        valid_views = valid_mask_flat[:, point_idx]
        point3d = points3d_flat[valid_views, point_idx].mean(axis=0)
        
        # Get RGB color (use first valid view)
        first_valid_view = np.where(valid_views)[0][0]
        if points_rgb is not None:
            rgb = points_rgb[first_valid_view, point_idx]
        else:
            rgb = np.array([128, 128, 128], dtype=np.uint8)
        
        # Add point to reconstruction
        point3D_id = idx + 1  # COLMAP uses 1-indexed IDs
        reconstruction.add_point3D(point3d, pycolmap.Track(), rgb)
        point_id_map[point_idx] = point3D_id
    
    # Add cameras and images
    for frame_idx in range(N):
        # Set camera
        if frame_idx == 0 or not shared_camera:
            pycolmap_intri = _build_pycolmap_intri(frame_idx, intrinsics, camera_type)
            camera = pycolmap.Camera(
                model=camera_type,
                width=image_size[0],
                height=image_size[1],
                params=pycolmap_intri,
                camera_id=frame_idx + 1
            )
            reconstruction.add_camera(camera)
        
        # Convert camera_to_world to world_to_camera for COLMAP
        # COLMAP uses cam_from_world (world to camera)
        cam_from_world_matrix = np.linalg.inv(camera_poses[frame_idx])
        
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(cam_from_world_matrix[:3, :3]),
            cam_from_world_matrix[:3, 3]
        )
        
        # Create image using the correct API
        image = pycolmap.Image(
            name=f"image_{frame_idx:04d}.jpg",
            camera_id=camera.camera_id if shared_camera and frame_idx > 0 else frame_idx + 1,
            image_id=frame_idx + 1
        )
        image.cam_from_world = cam_from_world
        
        # Add 2D points and establish tracks
        points2D_list = []
        point2D_idx = 0
        
        for point_idx in valid_point_indices:
            if valid_mask_flat[frame_idx, point_idx]:
                # Calculate 2D pixel coordinates
                v = point_idx // W
                u = point_idx % W
                point2D_xy = np.array([u + 0.5, v + 0.5], dtype=np.float64)
                
                point3D_id = point_id_map[point_idx]
                points2D_list.append(pycolmap.Point2D(point2D_xy, point3D_id))
                
                # Add to track
                track = reconstruction.points3D[point3D_id].track
                track.add_element(frame_idx + 1, point2D_idx)
                point2D_idx += 1
        
        # Set points for image
        if points2D_list:
            image.points2D = pycolmap.ListPoint2D(points2D_list)
        
        reconstruction.add_image(image)
    
    return reconstruction, intrinsics


def pi3_to_colmap_simple(
    points3d,
    camera_poses,
    image_size,
    intrinsics=None,
    points_rgb=None,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
):
    """
    Simplified conversion from Pi3 output to COLMAP format without tracks.
    This is useful for visualization and initialization but should NOT be used for bundle adjustment.
    
    Note: This function creates a frame-based reconstruction compatible with pycolmap 3.13+.
    Camera poses are stored in frames and can be accessed via reconstruction.frames[frame_id].
    Images are not separately persisted in this simplified format - use pi3_to_colmap() if you need
    full image metadata with 2D-3D correspondences.
    
    Args:
        points3d (np.ndarray): 3D points, shape (P, 3) where P is number of points
        camera_poses (np.ndarray): Camera-to-world matrices, shape (N, 4, 4) or (B, N, 4, 4)
        image_size (np.ndarray): Image size [width, height]
        intrinsics (np.ndarray, optional): Camera intrinsics, shape (N, 3, 3) or (B, N, 3, 3)
        points_rgb (np.ndarray, optional): RGB colors for points, shape (P, 3)
        shared_camera (bool): Whether all frames share the same camera
        camera_type (str): COLMAP camera model type
    
    Returns:
        reconstruction (pycolmap.Reconstruction): COLMAP reconstruction object
    """
    if not PYCOLMAP_AVAILABLE:
        raise ImportError("pycolmap is required for COLMAP export. Install it with: pip install pycolmap")
    
    # Convert tensors to numpy if needed
    if torch.is_tensor(points3d):
        points3d = points3d.detach().cpu().numpy()
    if torch.is_tensor(camera_poses):
        camera_poses = camera_poses.detach().cpu().numpy()
    if points_rgb is not None and torch.is_tensor(points_rgb):
        points_rgb = points_rgb.detach().cpu().numpy()
    
    # Handle batch dimension - camera_poses should be (N, 4, 4) or (B, N, 4, 4)
    # ndim == 4 means (B, N, 4, 4), ndim == 3 means (N, 4, 4) - no batch
    if camera_poses.ndim == 4:
        camera_poses = camera_poses[0]  # Remove batch dimension if present: (B, N, 4, 4) -> (N, 4, 4)
    
    # Reshape points if needed
    if points3d.ndim > 2:
        original_shape = points3d.shape
        points3d = points3d.reshape(-1, 3)
        if points_rgb is not None:
            points_rgb = points_rgb.reshape(-1, 3)
    
    N = len(camera_poses)
    P = len(points3d)
    
    # Estimate intrinsics if not provided
    if intrinsics is None:
        intrinsics = _estimate_intrinsics(image_size, N, camera_type)
    elif torch.is_tensor(intrinsics):
        intrinsics = intrinsics.detach().cpu().numpy()
        if intrinsics.ndim == 4:
            intrinsics = intrinsics[0]  # Remove batch dimension, keep (N, 3, 3)
    elif intrinsics.ndim == 4:
        intrinsics = intrinsics[0]  # Remove batch dimension, keep (N, 3, 3)
    
    # Create reconstruction
    reconstruction = pycolmap.Reconstruction()
    
    # Add 3D points
    for point_idx in range(P):
        rgb = points_rgb[point_idx] if points_rgb is not None else np.array([128, 128, 128])
        # Ensure RGB is in 0-255 range
        if rgb.max() <= 1.0:
            rgb = (rgb * 255).astype(np.uint8)
        reconstruction.add_point3D(points3d[point_idx], pycolmap.Track(), rgb)
    
    # Create rig with sensors (required by pycolmap 3.13+)
    rig = pycolmap.Rig(rig_id=1)
    
    # Add cameras and register them with the rig as sensors
    if shared_camera:
        # Single camera for all frames
        pycolmap_intri = _build_pycolmap_intri(0, intrinsics, camera_type)
        camera = pycolmap.Camera(
            model=camera_type,
            width=int(image_size[0]),
            height=int(image_size[1]),
            params=pycolmap_intri,
            camera_id=1
        )
        reconstruction.add_camera(camera)
        
        # Add camera as reference sensor to rig
        sensor = pycolmap.sensor_t()
        sensor.type = pycolmap.SensorType.CAMERA
        sensor.id = 1
        rig.add_ref_sensor(sensor)
    else:
        # Separate camera for each frame
        for frame_idx in range(N):
            pycolmap_intri = _build_pycolmap_intri(frame_idx, intrinsics, camera_type)
            camera = pycolmap.Camera(
                model=camera_type,
                width=int(image_size[0]),
                height=int(image_size[1]),
                params=pycolmap_intri,
                camera_id=frame_idx + 1
            )
            reconstruction.add_camera(camera)
            
            # Add camera as sensor to rig
            sensor = pycolmap.sensor_t()
            sensor.type = pycolmap.SensorType.CAMERA
            sensor.id = frame_idx + 1
            
            if frame_idx == 0:
                # First camera is the reference sensor
                rig.add_ref_sensor(sensor)
            else:
                # Other cameras are non-reference sensors
                rig.add_sensor(sensor, pycolmap.Rigid3d())
    
    # Add rig to reconstruction
    reconstruction.add_rig(rig)
    
    # Add frames with poses and create corresponding images
    for frame_idx in range(N):
        camera_id = 1 if shared_camera else frame_idx + 1
        
        # Create frame
        frame = pycolmap.Frame(frame_id=frame_idx + 1, rig_id=1)
        
        # Create data_id - this links the frame to the camera/image
        data_id = pycolmap.data_t()
        data_id.sensor_id = pycolmap.sensor_t()
        data_id.sensor_id.type = pycolmap.SensorType.CAMERA
        data_id.sensor_id.id = camera_id
        data_id.id = frame_idx + 1  # This must match the image_id we'll create
        frame.add_data_id(data_id)
        
        reconstruction.add_frame(frame)
        
        # Set camera pose on the frame
        # Pi3 outputs camera-to-world, but COLMAP needs world-to-camera (cam_from_world)
        cam_from_world_matrix = np.linalg.inv(camera_poses[frame_idx])
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(cam_from_world_matrix[:3, :3]),
            cam_from_world_matrix[:3, 3]
        )
        reconstruction.frames[frame_idx + 1].set_cam_from_world(camera_id, cam_from_world)
        
        # Register the frame to make it active
        reconstruction.register_frame(frame_idx + 1)
        
        # Create and add image linked to this frame
        # The image_id must match the data_id.id set above
        image = pycolmap.Image(
            name=f"image_{frame_idx:04d}.jpg",
            camera_id=camera_id,
            image_id=frame_idx + 1  # Must match data_id.id
        )
        # Set frame_id to link image to frame
        image.frame_id = frame_idx + 1
        reconstruction.add_image(image)
    
    return reconstruction


def _estimate_intrinsics(image_size, num_frames, camera_type="SIMPLE_PINHOLE"):
    """
    Estimate camera intrinsics from image size.
    
    Args:
        image_size (np.ndarray): [width, height]
        num_frames (int): Number of frames
        camera_type (str): Camera model type
    
    Returns:
        intrinsics (np.ndarray): Camera intrinsics, shape (N, 3, 3)
    """
    W, H = image_size
    
    # Simple heuristic: focal length = 1.2 * max(W, H)
    focal = 1.2 * max(W, H)
    cx = W / 2.0
    cy = H / 2.0
    
    K = np.array([
        [focal, 0, cx],
        [0, focal, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Repeat for all frames
    intrinsics = np.repeat(K[np.newaxis], num_frames, axis=0)
    
    return intrinsics


def _build_pycolmap_intri(frame_idx, intrinsics, camera_type, extra_params=None):
    """
    Build camera parameters for pycolmap based on camera type.
    
    Args:
        frame_idx (int): Frame index
        intrinsics (np.ndarray): Camera intrinsics, shape (N, 3, 3)
        camera_type (str): Camera model type
        extra_params (np.ndarray, optional): Additional camera parameters
    
    Returns:
        pycolmap_intri (np.ndarray): Camera parameters for pycolmap
    """
    if camera_type == "PINHOLE":
        # fx, fy, cx, cy
        pycolmap_intri = np.array([
            intrinsics[frame_idx][0, 0],  # fx
            intrinsics[frame_idx][1, 1],  # fy
            intrinsics[frame_idx][0, 2],  # cx
            intrinsics[frame_idx][1, 2]   # cy
        ])
    elif camera_type == "SIMPLE_PINHOLE":
        # f, cx, cy
        focal = (intrinsics[frame_idx][0, 0] + intrinsics[frame_idx][1, 1]) / 2
        pycolmap_intri = np.array([
            focal,
            intrinsics[frame_idx][0, 2],  # cx
            intrinsics[frame_idx][1, 2]   # cy
        ])
    elif camera_type == "SIMPLE_RADIAL":
        # f, cx, cy, k1
        focal = (intrinsics[frame_idx][0, 0] + intrinsics[frame_idx][1, 1]) / 2
        k1 = extra_params[frame_idx][0] if extra_params is not None else 0.0
        pycolmap_intri = np.array([
            focal,
            intrinsics[frame_idx][0, 2],  # cx
            intrinsics[frame_idx][1, 2],  # cy
            k1
        ])
    else:
        raise ValueError(f"Unsupported camera type: {camera_type}")
    
    return pycolmap_intri


def _save_images(images, output_path, reconstruction):
    """
    Save images to the 'images' folder under output_path.
    
    Args:
        images (torch.Tensor or np.ndarray): Images to save, shape (N, 3, H, W) or (B, N, 3, H, W)
        output_path (str): Base output directory path
        reconstruction (pycolmap.Reconstruction): COLMAP reconstruction (used to get image names)
    """
    try:
        import cv2
    except ImportError:
        print("Warning: opencv-python not installed. Cannot save images.")
        return
    
    # Convert tensors to numpy if needed
    if torch.is_tensor(images):
        images = images.detach().cpu().numpy()
    
    # Handle batch dimension - images should be (N, 3, H, W) or (B, N, 3, H, W)
    if images.ndim == 5:
        images = images[0]  # Remove batch dimension: (B, N, 3, H, W) -> (N, 3, H, W)
    
    N = images.shape[0]
    
    # Create images folder
    images_dir = os.path.join(output_path, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Save each image
    for frame_idx in range(N):
        # Get image data: (3, H, W) -> (H, W, 3)
        img = images[frame_idx].transpose(1, 2, 0)
        
        # Convert from [0, 1] to [0, 255] if needed
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Get the image name from the reconstruction
        # Frame IDs in reconstruction are 1-indexed
        image_id = frame_idx + 1
        if image_id in reconstruction.images:
            image_name = reconstruction.images[image_id].name
        else:
            # Fallback to default naming
            image_name = f"image_{frame_idx:04d}.jpg"
        
        # Save the image
        image_path = os.path.join(images_dir, image_name)
        cv2.imwrite(image_path, img_bgr)
    
    print(f"Saved {N} images to: {images_dir}")


def save_colmap_reconstruction(reconstruction, output_path, images=None):
    """
    Save COLMAP reconstruction to disk.
    
    Args:
        reconstruction (pycolmap.Reconstruction): COLMAP reconstruction object
        output_path (str): Output directory path
        images (torch.Tensor or np.ndarray, optional): Images to save, shape (N, 3, H, W) or (B, N, 3, H, W)
                                                        with pixel values in range [0, 1]
    """
    if not PYCOLMAP_AVAILABLE:
        raise ImportError("pycolmap is required for COLMAP export. Install it with: pip install pycolmap")
    
    os.makedirs(output_path, exist_ok=True)
    
    # Save the reconstruction data
    reconstruction.write(output_path)
    print(f"COLMAP reconstruction saved to: {output_path}")
    
    # Save images if provided
    if images is not None:
        _save_images(images, output_path, reconstruction)
