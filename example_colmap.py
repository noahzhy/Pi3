import torch
import argparse
import numpy as np
import os
from pi3.utils.basic import load_multimodal_data, write_ply
from pi3.utils.geometry import depth_edge
from pi3.utils.colmap_export import pi3_to_colmap_simple, save_colmap_reconstruction
from pi3.models.pi3x import Pi3X

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run inference with Pi3 and export to COLMAP format.")
    
    parser.add_argument("--data_path", type=str, default='examples/skating.mp4',
                        help="Path to the input image directory or a video file.")
    
    parser.add_argument("--conditions_path", type=str, default=None,
                        help="Optional path to a .npz file containing 'poses', 'depths', 'intrinsics'.")

    parser.add_argument("--save_path", type=str, default='examples/result.ply',
                        help="Path to save the output .ply file.")
    
    parser.add_argument("--colmap_path", type=str, default='examples/result_colmap',
                        help="Path to save the COLMAP reconstruction.")
    
    parser.add_argument("--interval", type=int, default=-1,
                        help="Interval to sample image. Default: 1 for images dir, 10 for video")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to the model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
                        
    args = parser.parse_args()
    if args.interval < 0:
        args.interval = 10 if args.data_path.endswith('.mp4') else 1
    print(f'Sampling interval: {args.interval}')

    # 1. Prepare model
    print(f"Loading model...")
    device = torch.device(args.device)
    if args.ckpt is not None:
        model = Pi3X().to(device).eval()
        if args.ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(args.ckpt)
        else:
            weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        
        model.load_state_dict(weight, strict=False)
    else:
        model = Pi3X.from_pretrained("yyfz233/Pi3X").to(device).eval()

    # 2. Prepare input data
    poses = None
    depths = None
    intrinsics = None

    if args.conditions_path is not None and os.path.exists(args.conditions_path):
        print(f"Loading conditions from {args.conditions_path}...")
        data_npz = np.load(args.conditions_path, allow_pickle=True)
        poses = data_npz['poses']             # Expected (N, 4, 4) OpenCV camera-to-world
        depths = data_npz['depths']           # Expected (N, H, W)
        intrinsics = data_npz['intrinsics']   # Expected (N, 3, 3)

    conditions = dict(
        intrinsics=intrinsics,
        poses=poses,
        depths=depths
    )

    # Load images
    imgs, conditions = load_multimodal_data(args.data_path, conditions, interval=args.interval, device=device) 

    # 3. Infer
    print("Running model inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            res = model(
                imgs=imgs, 
                **conditions
            )

    # 4. Process mask
    masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
    non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
    masks = torch.logical_and(masks, non_edge)[0]

    # 5. Save PLY point cloud
    print(f"Saving point cloud to: {args.save_path}")
    if os.path.dirname(args.save_path):
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        
    write_ply(res['points'][0][masks].cpu(), imgs[0].permute(0, 2, 3, 1)[masks], args.save_path)
    print("PLY point cloud saved.")

    # 6. Export to COLMAP format
    print(f"\nExporting to COLMAP format...")
    try:
        # Get image dimensions
        _, _, H, W = imgs.shape[1:]
        image_size = np.array([W, H])
        
        # Flatten the points for COLMAP export
        points3d_masked = res['points'][0][masks].cpu()
        colors_masked = imgs[0].permute(0, 2, 3, 1)[masks].cpu()
        
        # Convert colors to 0-255 range if needed
        if colors_masked.max() <= 1.0:
            colors_masked = (colors_masked * 255).numpy().astype(np.uint8)
        else:
            colors_masked = colors_masked.numpy().astype(np.uint8)
        
        # Use simplified COLMAP export
        reconstruction = pi3_to_colmap_simple(
            points3d=points3d_masked.numpy(),
            camera_poses=res['camera_poses'].cpu().numpy(),
            image_size=image_size,
            intrinsics=conditions.get('intrinsics', None),
            points_rgb=colors_masked,
            shared_camera=False,
            camera_type="SIMPLE_PINHOLE"
        )
        
        # Save COLMAP reconstruction
        save_colmap_reconstruction(reconstruction, args.colmap_path, images=imgs)
        print(f"COLMAP reconstruction exported successfully!")
        print(f"  - {len(reconstruction.points3D)} 3D points")
        print(f"  - {len(reconstruction.images)} images")
        print(f"  - {len(reconstruction.cameras)} cameras")
        
    except ImportError as e:
        print(f"Warning: Could not export to COLMAP format: {e}")
        print("Install pycolmap with: pip install pycolmap")
    except Exception as e:
        print(f"Error during COLMAP export: {e}")
        import traceback
        traceback.print_exc()

    print("\nDone.")
