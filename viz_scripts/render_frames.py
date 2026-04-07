"""
Render RGB and depth frames from a SplaTAM checkpoint, plus save GT frames.
Saves side-by-side comparison plots, per-frame metrics, poses, and a CSV index.

Usage:
  python viz_scripts/render_frames.py configs/replica/replica_room0.py [--eval_every 25] [--save_frames]
"""
import argparse
import csv
import sys
import os
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from datasets.gradslam_datasets import load_dataset_config, ReplicaDataset
from utils.eval_helpers import eval, METRIC_DEPTH_PNG_SCALE
from utils.slam_external import build_rotation


def get_eval_indices(num_frames: int, eval_every: int) -> list[int]:
    indices = []
    for time_idx in range(num_frames):
        if time_idx == 0 or (time_idx + 1) % eval_every == 0:
            indices.append(time_idx)
    return indices


def pose_to_str(c2w: np.ndarray) -> str:
    """Flatten 4x4 c2w matrix to space-separated string (traj.txt format)."""
    flat = c2w.reshape(-1)
    return " ".join(f"{v:.18e}" for v in flat)


def get_estimated_c2w(params: dict, frame_idx: int) -> np.ndarray:
    """Extract the checkpoint's estimated c2w (4x4) for a given frame index."""
    cam_rot = F.normalize(params['cam_unnorm_rots'][..., frame_idx].detach())
    cam_tran = params['cam_trans'][..., frame_idx].detach()
    w2c = torch.eye(4).cuda().float()
    w2c[:3, :3] = build_rotation(cam_rot)
    w2c[:3, 3] = cam_tran
    return torch.linalg.inv(w2c).cpu().numpy().astype(np.float64)


def save_traj_and_csv(params: dict, num_frames: int, eval_every: int, eval_dir: str):
    """Save traj.txt (estimated poses) and frames_index.csv."""
    indices = get_eval_indices(num_frames, eval_every)

    traj_path = os.path.join(eval_dir, "traj.txt")
    csv_path = os.path.join(eval_dir, "frames_index.csv")

    with open(traj_path, "w") as traj_f, \
         open(csv_path, "w", newline="") as csv_f:

        writer = csv.writer(csv_f)
        writer.writerow(["index", "gt_rgb_path", "gt_depth_path",
                         "gs_rgb_path", "gs_depth_path", "estimated_pose"])

        for frame_idx in indices:
            c2w = get_estimated_c2w(params, frame_idx)
            pose_str = pose_to_str(c2w)

            # traj.txt: one line per frame, 16 floats (c2w row-major)
            traj_f.write(pose_str + "\n")

            # CSV row with relative paths from eval_dir
            gt_rgb = os.path.join("rgb", f"gt_{frame_idx:04d}.png")
            gt_depth = os.path.join("depth_metric", f"gt_{frame_idx:04d}.png")
            gs_rgb = os.path.join("rendered_rgb", f"gs_{frame_idx:04d}.png")
            gs_depth = os.path.join("rendered_depth_metric", f"gs_{frame_idx:04d}.png")

            writer.writerow([frame_idx, gt_rgb, gt_depth, gs_rgb, gs_depth, pose_str])

    return traj_path, csv_path, len(indices)


def verify_metric_depth(eval_dir: str, indices: list[int]):
    """Compare rendered metric depth PNGs against GT metric depth PNGs.

    Both are uint16 PNG encoded as depth_meters * METRIC_DEPTH_PNG_SCALE.
    Reports L1 error in meters for valid pixels.
    """
    gt_dir = os.path.join(eval_dir, "depth_metric")
    render_dir = os.path.join(eval_dir, "rendered_depth_metric")

    if not os.path.isdir(gt_dir) or not os.path.isdir(render_dir):
        print("  Metric depth directories not found, skipping verification.")
        return

    errors = []
    for frame_idx in indices:
        gt_path = os.path.join(gt_dir, f"gt_{frame_idx:04d}.png")
        gs_path = os.path.join(render_dir, f"gs_{frame_idx:04d}.png")

        if not os.path.isfile(gt_path) or not os.path.isfile(gs_path):
            continue

        gt_u16 = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        gs_u16 = cv2.imread(gs_path, cv2.IMREAD_UNCHANGED)

        if gt_u16 is None or gs_u16 is None:
            continue

        gt_m = gt_u16.astype(np.float64) / METRIC_DEPTH_PNG_SCALE
        gs_m = gs_u16.astype(np.float64) / METRIC_DEPTH_PNG_SCALE

        valid = (gt_u16 > 0) & (gs_u16 > 0)
        if valid.sum() == 0:
            continue

        l1 = np.abs(gt_m[valid] - gs_m[valid]).mean()
        errors.append((frame_idx, l1, valid.sum()))

    if not errors:
        print("  No valid metric depth pairs found for verification.")
        return

    all_l1 = [e[1] for e in errors]
    avg_l1 = np.mean(all_l1)
    max_l1 = np.max(all_l1)
    print(f"  Metric depth verification ({len(errors)} frames):")
    print(f"    Avg L1: {avg_l1:.4f} m  |  Max L1: {max_l1:.4f} m")
    if max_l1 > 0.5:
        print("    WARNING: max L1 > 0.5m — metric depth may have issues.")
    else:
        print("    Metric depth looks consistent with GT.")


def get_dataset(cfg):
    dataset_config = load_dataset_config(cfg['data']['gradslam_data_cfg'])
    return ReplicaDataset(
        dataset_config,
        cfg['data']['basedir'],
        cfg['data']['sequence'],
        desired_image_height=cfg['data']['desired_image_height'],
        desired_image_width=cfg['data']['desired_image_width'],
        start=cfg['data']['start'],
        end=cfg['data']['end'],
        stride=cfg['data']['stride'],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Path to experiment config file")
    parser.add_argument("--eval_every", type=int, default=25, help="Render every Nth frame")
    parser.add_argument("--save_frames", action="store_true", help="Save individual rendered/GT RGB and depth images")
    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()
    cfg = experiment.config

    results_dir = os.path.join(cfg['workdir'], cfg['run_name'])
    scene_path = os.path.join(results_dir, "params.npz")
    eval_dir = os.path.join(results_dir, "eval")

    print(f"Loading checkpoint: {scene_path}")
    params = dict(np.load(scene_path, allow_pickle=True))
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}

    print(f"Loading dataset: {cfg['data']['sequence']}")
    dataset = get_dataset(cfg)
    num_frames = len(dataset)

    print(f"Rendering {num_frames} frames (eval_every={args.eval_every}, save_frames={args.save_frames})")
    with torch.no_grad():
        eval(
            dataset, params, num_frames, eval_dir,
            sil_thres=cfg['mapping']['sil_thres'],
            mapping_iters=cfg['mapping']['num_iters'],
            add_new_gaussians=cfg['mapping']['add_new_gaussians'],
            eval_every=args.eval_every,
            save_frames=args.save_frames,
        )

    print(f"\nResults saved to: {eval_dir}/")
    print(f"  plots/        — side-by-side GT vs rendered comparisons")
    print(f"  metrics.png   — PSNR and depth L1 over time")
    print(f"  psnr.txt      — per-frame PSNR values")
    if args.save_frames:
        print(f"  rendered_rgb/           — rendered RGB frames")
        print(f"  rendered_depth/         — rendered depth maps (colorized)")
        print(f"  rendered_depth_metric/  — rendered depth maps (uint16 metric PNG)")
        print(f"  rgb/                    — GT RGB frames")
        print(f"  depth/                  — GT depth maps (colorized)")
        print(f"  depth_metric/           — GT depth maps (uint16 metric PNG)")

        # Save traj.txt and CSV index
        traj_path, csv_path, n_frames = save_traj_and_csv(
            params=params,
            num_frames=num_frames,
            eval_every=args.eval_every,
            eval_dir=eval_dir,
        )
        print(f"  traj.txt                — {n_frames} estimated camera poses (c2w 4x4)")
        print(f"  frames_index.csv        — index with paths and estimated poses")

        # Save camera intrinsics and image size
        _, _, intrinsics, _ = dataset[0]
        intrinsics = intrinsics[:3, :3].cpu().numpy()
        sample_img = cv2.imread(os.path.join(eval_dir, "rendered_rgb", "gs_0000.png"))
        h, w = sample_img.shape[:2]
        camera_info = {
            "image_height": h,
            "image_width": w,
            "fx": float(intrinsics[0, 0]),
            "fy": float(intrinsics[1, 1]),
            "cx": float(intrinsics[0, 2]),
            "cy": float(intrinsics[1, 2]),
        }
        import json as _json
        camera_info_path = os.path.join(eval_dir, "camera_info.json")
        with open(camera_info_path, "w") as f:
            _json.dump(camera_info, f, indent=2)
        print(f"  camera_info.json        — intrinsics and image size")

        # Verify metric depth consistency
        indices = get_eval_indices(num_frames, args.eval_every)
        print("\nVerifying metric depth...")
        verify_metric_depth(eval_dir, indices)


if __name__ == "__main__":
    main()
