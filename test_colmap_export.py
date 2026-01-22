#!/usr/bin/env python3
"""
Simple test to verify COLMAP export functionality works correctly.
"""
import numpy as np
import torch
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pi3.utils.colmap_export import pi3_to_colmap_simple, save_colmap_reconstruction

def test_colmap_export():
    """Test basic COLMAP export functionality with synthetic data."""
    print("Testing COLMAP export functionality...")
    
    # Create synthetic test data
    N = 3  # Number of frames
    H, W = 84, 112  # Image dimensions (small for testing)
    P = 100  # Number of 3D points
    
    # Generate random 3D points
    np.random.seed(42)
    points3d = np.random.randn(P, 3).astype(np.float32) * 2.0
    
    # Generate camera poses (identity with small translations)
    camera_poses = np.zeros((N, 4, 4), dtype=np.float32)
    for i in range(N):
        camera_poses[i] = np.eye(4)
        camera_poses[i][0, 3] = i * 0.5  # Translation along X axis
    
    # Generate colors
    points_rgb = (np.random.rand(P, 3) * 255).astype(np.uint8)
    
    # Image size
    image_size = np.array([W, H])
    
    print(f"  Points3D shape: {points3d.shape}")
    print(f"  Camera poses shape: {camera_poses.shape}")
    print(f"  Image size: {image_size}")
    
    # Test COLMAP export
    try:
        reconstruction = pi3_to_colmap_simple(
            points3d=points3d,
            camera_poses=camera_poses,
            image_size=image_size,
            points_rgb=points_rgb,
            shared_camera=False,
            camera_type="SIMPLE_PINHOLE"
        )
        
        print(f"\n✓ COLMAP reconstruction created successfully!")
        print(f"  - 3D points: {len(reconstruction.points3D)}")
        print(f"  - Images: {len(reconstruction.images)}")
        print(f"  - Cameras: {len(reconstruction.cameras)}")
        
        # Verify reconstruction contents
        assert len(reconstruction.points3D) == P, "Number of 3D points mismatch"
        assert len(reconstruction.images) == N, "Number of images mismatch"
        assert len(reconstruction.cameras) == N, "Number of cameras mismatch"
        
        # Test saving to disk
        output_path = "/tmp/test_colmap_reconstruction"
        save_colmap_reconstruction(reconstruction, output_path)
        
        # Verify files were created
        assert os.path.exists(output_path), "Output directory not created"
        assert os.path.exists(os.path.join(output_path, "cameras.bin")), "cameras.bin not created"
        assert os.path.exists(os.path.join(output_path, "images.bin")), "images.bin not created"
        assert os.path.exists(os.path.join(output_path, "points3D.bin")), "points3D.bin not created"
        
        print(f"\n✓ COLMAP reconstruction saved successfully to {output_path}")
        print(f"  - cameras.bin")
        print(f"  - images.bin")
        print(f"  - points3D.bin")
        
        # Clean up
        import shutil
        shutil.rmtree(output_path)
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_colmap_export()
    sys.exit(0 if success else 1)
