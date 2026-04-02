"""
Convert an Isaac Sim recorded sequence to the Replica dataset format expected by SplaTAM.

Isaac Sim structure (input):
    data/IsaacSim/<scene>/Cameras_Camera/
        rgb/rgb_XXXXX.png
        distance_to_image_plane/distance_to_image_plane_XXXXX.npy
        camera_params/camera_params_XXXXX.json

Replica structure (output):
    data/IsaacSim/<scene>/
        results/frameXXXXXX.jpg
        results/depthXXXXXX.png   (uint16, depth_in_meters * png_depth_scale)
        traj.txt                   (one 4x4 c2w matrix per line, 16 space-separated floats)

Also generates a GradSLAM-compatible YAML config file.

Usage:
    python scripts/convert_isaac_to_replica.py --scene office0
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np
from natsort import natsorted


def load_camera_params(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def isaac_cvt_to_c2w(camera_params):
    """
    Extract camera-to-world matrix from Isaac Sim cameraViewTransform.

    The cameraViewTransform is stored transposed. After transposing we get
    the world-to-camera transform T_Ci_W0 in USD camera convention
    (X-right, Y-up, -Z forward = OpenGL convention).

    Replica uses the same OpenGL convention, so the c2w can be used directly.
    """
    cvt = np.array(camera_params["cameraViewTransform"]).reshape(4, 4)
    T_Ci_W0 = cvt.T  # world-to-camera
    T_W0_Ci = np.linalg.inv(T_Ci_W0)  # camera-to-world
    return T_W0_Ci


def compute_intrinsics(camera_params, h, w):
    """Compute fx, fy, cx, cy from Isaac Sim camera parameters."""
    focal_length = camera_params["cameraFocalLength"]
    aperture = camera_params["cameraAperture"]
    fx = focal_length / aperture[0] * w
    fy = focal_length / aperture[1] * h
    cx = w / 2.0
    cy = h / 2.0
    return fx, fy, cx, cy


def convert_depth(depth_npy_path, png_depth_scale, max_depth):
    """
    Convert float32 depth (meters) to uint16 PNG.
    Depths beyond max_depth and infinite/invalid values are set to 0.
    """
    depth = np.load(depth_npy_path)
    valid_mask = np.isfinite(depth) & (depth <= max_depth)
    depth_scaled = np.zeros_like(depth, dtype=np.float64)
    depth_scaled[valid_mask] = depth[valid_mask] * png_depth_scale
    depth_scaled = np.clip(depth_scaled, 0, 65535)
    return depth_scaled.astype(np.uint16)


def main():
    parser = argparse.ArgumentParser(description="Convert Isaac Sim sequence to Replica format")
    parser.add_argument("--basedir", type=str, default="./data/IsaacSim", help="Base directory of Isaac Sim data")
    parser.add_argument("--scene", type=str, required=True, help="Scene name (e.g., office0)")
    parser.add_argument("--png_depth_scale", type=float, default=6553.5,
                        help="Depth scale factor: depth_meters * scale = uint16 value. "
                             "Default 6553.5 matches Replica (~0.15mm precision, max ~10m).")
    parser.add_argument("--max_depth", type=float, default=10.0,
                        help="Maximum depth in meters. Pixels beyond this are set to 0.")
    args = parser.parse_args()

    scene_dir = os.path.join(args.basedir, args.scene)
    camera_dir = os.path.join(scene_dir, "Cameras_Camera")
    results_dir = os.path.join(scene_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Discover files
    rgb_paths = natsorted(glob.glob(os.path.join(camera_dir, "rgb", "rgb_*.png")))
    depth_paths = natsorted(glob.glob(os.path.join(camera_dir, "distance_to_image_plane", "*.npy")))
    param_paths = natsorted(glob.glob(os.path.join(camera_dir, "camera_params", "camera_params_*.json")))

    n_frames = len(rgb_paths)
    assert n_frames == len(depth_paths) == len(param_paths), (
        f"Mismatch: {len(rgb_paths)} rgb, {len(depth_paths)} depth, {len(param_paths)} params"
    )
    print(f"Found {n_frames} frames in {camera_dir}")

    # Extract intrinsics from first frame
    first_params = load_camera_params(param_paths[0])
    first_depth = np.load(depth_paths[0])
    h, w = first_depth.shape
    fx, fy, cx, cy = compute_intrinsics(first_params, h, w)
    print(f"Image size: {w}x{h}")
    print(f"Intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print(f"Depth scale: {args.png_depth_scale} (max representable: {65535/args.png_depth_scale:.2f}m)")
    print(f"Max depth: {args.max_depth}m (pixels beyond this → 0)")

    # Convert frames
    traj_lines = []
    for i in range(n_frames):
        if i % 100 == 0:
            print(f"  Processing frame {i}/{n_frames}...")

        # RGB: copy as jpg
        rgb = cv2.imread(rgb_paths[i], cv2.IMREAD_COLOR)
        out_rgb_path = os.path.join(results_dir, f"frame{i:06d}.jpg")
        cv2.imwrite(out_rgb_path, rgb)

        # Depth: convert npy float32 -> uint16 png
        depth_uint16 = convert_depth(depth_paths[i], args.png_depth_scale, args.max_depth)
        out_depth_path = os.path.join(results_dir, f"depth{i:06d}.png")
        cv2.imwrite(out_depth_path, depth_uint16)

        # Pose: extract c2w from cameraViewTransform
        params = load_camera_params(param_paths[i])
        c2w = isaac_cvt_to_c2w(params)
        # Write as 16 space-separated floats (row-major 4x4)
        traj_lines.append(" ".join(f"{v:.18e}" for v in c2w.flatten()))

    # Write trajectory
    traj_path = os.path.join(scene_dir, "traj.txt")
    with open(traj_path, "w") as f:
        f.write("\n".join(traj_lines) + "\n")
    print(f"Wrote {len(traj_lines)} poses to {traj_path}")

    # Write GradSLAM config yaml
    config_path = os.path.join(scene_dir, "dataconfig.yaml")
    with open(config_path, "w") as f:
        f.write(f"dataset_name: 'replica'\n")
        f.write(f"camera_params:\n")
        f.write(f"  image_height: {h}\n")
        f.write(f"  image_width: {w}\n")
        f.write(f"  fx: {fx}\n")
        f.write(f"  fy: {fy}\n")
        f.write(f"  cx: {cx}\n")
        f.write(f"  cy: {cy}\n")
        f.write(f"  png_depth_scale: {args.png_depth_scale}\n")
        f.write(f"  crop_edge: 0\n")
    print(f"Wrote config to {config_path}")
    print("Done!")


if __name__ == "__main__":
    main()
