"""
Render a single novel view from a SplaTAM checkpoint using a hardcoded pose offset.

Usage:
  python viz_scripts/render_single_novel_view.py configs/isaacsim/isaac_office0.py
  python viz_scripts/render_single_novel_view.py configs/replica/replica_room0.py --width 960 --height 540
"""

import argparse
import os
import sys
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

import numpy as np
import torch
import cv2

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

from utils.recon_helpers import setup_camera
from utils.slam_helpers import params2rendervar, params2depthplussilhouette


# Hardcoded novel view transform in world frame (meters).
# This is applied to the first checkpoint camera pose to create a nearby novel view.
HARDCODED_TRANSLATION_WORLD = np.array([0.0, 0.0, 2.5], dtype=np.float32)

# Hardcoded orientation offset in world frame (degrees).
# Convention here is intrinsic XYZ rotations: roll (X), pitch (Y), yaw (Z).
HARDCODED_RPY_DEG_WORLD = np.array([0.0, -120.0, 0.0], dtype=np.float32)


def load_config(config_path: str):
    experiment = SourceFileLoader(os.path.basename(config_path), config_path).load_module()
    return experiment.config


def load_checkpoint(params_path: str):
    raw = dict(np.load(params_path, allow_pickle=True))
    params = {k: torch.tensor(v).cuda().float() for k, v in raw.items()}
    return raw, params


def rpy_deg_to_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Convert roll-pitch-yaw (deg) to a rotation matrix."""
    r = np.deg2rad(roll_deg)
    p = np.deg2rad(pitch_deg)
    y = np.deg2rad(yaw_deg)

    cx, sx = np.cos(r), np.sin(r)
    cy, sy = np.cos(p), np.sin(p)
    cz, sz = np.cos(y), np.sin(y)

    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cx, -sx],
        [0.0, sx, cx],
    ], dtype=np.float32)
    ry = np.array([
        [cy, 0.0, sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0, cy],
    ], dtype=np.float32)
    rz = np.array([
        [cz, -sz, 0.0],
        [sz, cz, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    return rz @ ry @ rx


def build_hardcoded_novel_w2c(base_w2c: np.ndarray) -> np.ndarray:
    """Create a hardcoded novel pose from the first camera pose."""
    c2w = np.linalg.inv(base_w2c)
    c2w[:3, 3] += HARDCODED_TRANSLATION_WORLD

    rot_delta = rpy_deg_to_matrix(*HARDCODED_RPY_DEG_WORLD.tolist())
    c2w[:3, :3] = rot_delta @ c2w[:3, :3]

    novel_w2c = np.linalg.inv(c2w)
    return novel_w2c.astype(np.float32)


def make_white_bg_cam(cam):
    return Camera(
        image_height=cam.image_height,
        image_width=cam.image_width,
        tanfovx=cam.tanfovx,
        tanfovy=cam.tanfovy,
        bg=torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda"),
        scale_modifier=cam.scale_modifier,
        viewmatrix=cam.viewmatrix,
        projmatrix=cam.projmatrix,
        sh_degree=cam.sh_degree,
        campos=cam.campos,
        prefiltered=cam.prefiltered,
    )


def main():
    parser = argparse.ArgumentParser(description="Render one hardcoded novel view from SplaTAM map")
    parser.add_argument("experiment", type=str, help="Path to experiment config file")
    parser.add_argument("--width", type=int, default=None, help="Output width (default: checkpoint org_width)")
    parser.add_argument("--height", type=int, default=None, help="Output height (default: checkpoint org_height)")
    parser.add_argument("--near", type=float, default=0.01, help="Near plane")
    parser.add_argument("--far", type=float, default=100.0, help="Far plane")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    cfg = load_config(args.experiment)
    results_dir = os.path.join(cfg["workdir"], cfg["run_name"])
    params_path = os.path.join(results_dir, "params.npz")

    if args.output_dir is None:
        output_dir = os.path.join(results_dir, "novel_view")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading checkpoint: {params_path}")
    raw, params = load_checkpoint(params_path)

    org_w = int(raw["org_width"])
    org_h = int(raw["org_height"])
    w = int(args.width) if args.width is not None else org_w
    h = int(args.height) if args.height is not None else org_h

    k = raw["intrinsics"][:3, :3].astype(np.float32).copy()
    k[0, :] *= w / org_w
    k[1, :] *= h / org_h

    base_w2c = raw["w2c"].astype(np.float32)
    if base_w2c.ndim == 1:
        base_w2c = base_w2c.reshape(4, 4)

    novel_w2c = build_hardcoded_novel_w2c(base_w2c)
    print(f"Rendering novel view at resolution {w}x{h}")
    print(f"Hardcoded world translation: {HARDCODED_TRANSLATION_WORLD.tolist()}")
    print(f"Hardcoded world RPY (deg): {HARDCODED_RPY_DEG_WORLD.tolist()}")

    with torch.no_grad():
        cam = setup_camera(w, h, k, novel_w2c, near=args.near, far=args.far)
        white_bg_cam = make_white_bg_cam(cam)

        rendervar = params2rendervar(params)
        depth_rendervar = params2depthplussilhouette(
            params,
            torch.tensor(novel_w2c).cuda().float(),
        )

        rgb, _, depth_rgb = Renderer(raster_settings=white_bg_cam)(**rendervar)
        depth_sil, _, _ = Renderer(raster_settings=cam)(**depth_rendervar)

    rgb_np = (torch.clamp(rgb, 0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    depth_np = depth_sil[0].detach().cpu().numpy()
    sil_np = depth_sil[1].detach().cpu().numpy()

    valid = sil_np > 0.5
    if np.any(valid):
        dmin = float(np.percentile(depth_np[valid], 5))
        dmax = float(np.percentile(depth_np[valid], 95))
        dnorm = np.clip((depth_np - dmin) / max(dmax - dmin, 1e-6), 0.0, 1.0)
    else:
        dnorm = np.zeros_like(depth_np)
    depth_u8 = (dnorm * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)

    rgb_path = os.path.join(output_dir, "novel_rgb.png")
    depth_path = os.path.join(output_dir, "novel_depth.png")
    depth_npy_path = os.path.join(output_dir, "novel_depth.npy")
    pose_path = os.path.join(output_dir, "novel_w2c.npy")

    cv2.imwrite(rgb_path, cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(depth_path, depth_vis)
    np.save(depth_npy_path, depth_np)
    np.save(pose_path, novel_w2c)

    print("\nSaved:")
    print(f"  {rgb_path}")
    print(f"  {depth_path}")
    print(f"  {depth_npy_path}")
    print(f"  {pose_path}")


if __name__ == "__main__":
    main()
