"""
Render RGB and depth frames from a SplaTAM checkpoint, plus save GT frames.
Saves side-by-side comparison plots and per-frame metrics.

Usage:
  python viz_scripts/render_frames.py configs/replica/replica_room0.py [--eval_every 25] [--save_frames]
"""
import argparse
import sys
import os
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

import numpy as np
import torch

from datasets.gradslam_datasets import load_dataset_config, ReplicaDataset
from utils.eval_helpers import eval


def get_eval_indices(num_frames: int, eval_every: int) -> list[int]:
    indices = []
    for time_idx in range(num_frames):
        if time_idx == 0 or (time_idx + 1) % eval_every == 0:
            indices.append(time_idx)
    return indices


def save_rendered_frame_poses(dataset, num_frames: int, eval_every: int, eval_dir: str):
    """Save poses for frames rendered as gs_XXXX.png in eval directory."""
    indices = get_eval_indices(num_frames, eval_every)
    pose_entries = []

    for frame_idx in indices:
        _, _, _, pose = dataset[frame_idx]
        c2w = pose.detach().cpu().numpy()
        w2c = np.linalg.inv(c2w)
        pose_entries.append(
            {
                "frame_idx": int(frame_idx),
                "image_name": f"gs_{frame_idx:04d}.png",
                "c2w": c2w.tolist(),
                "w2c": w2c.tolist(),
            }
        )

    json_path = os.path.join(eval_dir, "rendered_frame_poses.json")
    with open(json_path, "w") as f:
        import json
        json.dump({"eval_every": eval_every, "frames": pose_entries}, f, indent=2)

    c2w_txt_path = os.path.join(eval_dir, "rendered_frame_c2w.txt")
    with open(c2w_txt_path, "w") as f:
        for entry in pose_entries:
            flat = np.array(entry["c2w"], dtype=np.float64).reshape(-1)
            vals = " ".join(f"{v:.18e}" for v in flat)
            f.write(f"{entry['frame_idx']} {vals}\n")

    w2c_txt_path = os.path.join(eval_dir, "rendered_frame_w2c.txt")
    with open(w2c_txt_path, "w") as f:
        for entry in pose_entries:
            flat = np.array(entry["w2c"], dtype=np.float64).reshape(-1)
            vals = " ".join(f"{v:.18e}" for v in flat)
            f.write(f"{entry['frame_idx']} {vals}\n")

    return json_path, c2w_txt_path, w2c_txt_path, len(pose_entries)


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
        print(f"  rendered_rgb/ — rendered RGB frames")
        print(f"  rendered_depth/ — rendered depth maps")
        print(f"  rgb/          — GT RGB frames")
        print(f"  depth/        — GT depth maps")
        poses_json, poses_c2w_txt, poses_w2c_txt, n_poses = save_rendered_frame_poses(
            dataset=dataset,
            num_frames=num_frames,
            eval_every=args.eval_every,
            eval_dir=eval_dir,
        )
        print(f"  rendered_frame_poses.json — {n_poses} rendered frame poses")
        print(f"  rendered_frame_c2w.txt    — frame_idx + 4x4 c2w")
        print(f"  rendered_frame_w2c.txt    — frame_idx + 4x4 w2c")
        print(f"Pose files saved:\n  {poses_json}\n  {poses_c2w_txt}\n  {poses_w2c_txt}")


if __name__ == "__main__":
    main()
