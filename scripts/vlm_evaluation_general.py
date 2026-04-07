"""
Minimal VLM evaluation on a single frame.

Loads frame data from eval/frames_index.csv and intrinsics from eval/camera_info.json.
Renders an RGB image from the 3DGS map at the frame's camera pose, queries the VLM,
and saves the result.

Usage:
  python scripts/vlm_evaluation_general.py \
    --experiment_dir ./experiments/IsaacSim/office0_0 \
    --image_idx 42 --model gpt-4o [--use_rendered]
"""
import argparse
import base64
import csv
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import httpx

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    params2rendervar,
    transform_to_frame,
    transformed_params2rendervar,
)


TASK_QUESTION = """
Determine the spatial relationship between the computer monitor and the cardboard box.

Return only valid JSON (no markdown, no extra text) with exactly this schema:
{
    "spatial_relationship": "in front of" | "behind" | "to the left of" | "to the right of" | "above" | "below" | "uncertain",
    "confidence": float (0.0 to 1.0),
    "reasoning": string
}
"""


# ---------------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------------

def c2w_to_translation(c2w: np.ndarray) -> tuple[float, float, float]:
    return float(c2w[0, 3]), float(c2w[1, 3]), float(c2w[2, 3])


def c2w_to_rpy_deg(c2w: np.ndarray) -> tuple[float, float, float]:
    """Extract roll, pitch, yaw (degrees) from a 4x4 c2w matrix."""
    R = c2w[:3, :3]
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

# Rotation helpers (already have math imported)
def rotation_x(angle_deg):
    """Rotation around first-camera X axis (= world Y, lateral tilt)."""
    a = math.radians(angle_deg)
    return np.array([[1,0,0],[0,math.cos(a),-math.sin(a)],[0,math.sin(a),math.cos(a)]])

def rotation_y(angle_deg):
    """Rotation around first-camera Y axis (= world Z, yaw/heading)."""
    a = math.radians(angle_deg)
    return np.array([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]])

def rotation_z(angle_deg):
    """Rotation around first-camera Z axis (= world X, roll)."""
    a = math.radians(angle_deg)
    return np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]])

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_frames_index(eval_dir: str) -> list[dict]:
    csv_path = os.path.join(eval_dir, "frames_index.csv")
    if not os.path.isfile(csv_path):
        print(f"Error: frames_index.csv not found at {csv_path}")
        print("Run viz_scripts/render_frames.py --save_frames first.")
        sys.exit(1)

    frames = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            floats = [float(v) for v in row["estimated_pose"].strip().split()]
            frames.append({
                "index": int(row["index"]),
                "gt_rgb_path": os.path.join(eval_dir, row["gt_rgb_path"]),
                "gt_depth_path": os.path.join(eval_dir, row["gt_depth_path"]),
                "gs_rgb_path": os.path.join(eval_dir, row["gs_rgb_path"]),
                "gs_depth_path": os.path.join(eval_dir, row["gs_depth_path"]),
                "camera_pose": np.array(floats, dtype=np.float64).reshape(4, 4),
            })
    return frames


def load_camera_info(eval_dir: str) -> dict:
    path = os.path.join(eval_dir, "camera_info.json")
    if not os.path.isfile(path):
        print(f"Error: camera_info.json not found at {path}")
        print("Run viz_scripts/render_frames.py --save_frames first.")
        sys.exit(1)
    with open(path, "r") as f:
        return json.load(f)


def get_frame(frames: list[dict], image_idx: int) -> dict | None:
    for f in frames:
        if f["index"] == image_idx:
            return f
    return None


# ---------------------------------------------------------------------------
# 3DGS rendering
# ---------------------------------------------------------------------------

def load_checkpoint(experiment_dir: str) -> dict:
    params_path = os.path.join(experiment_dir, "params.npz")
    if not os.path.isfile(params_path):
        print(f"Error: checkpoint not found at {params_path}")
        sys.exit(1)
    arr = dict(np.load(params_path, allow_pickle=True))
    return {k: torch.tensor(v).cuda().float() for k, v in arr.items()}


def render_frame(params: dict, frame_idx: int, camera_info: dict) -> np.ndarray:
    """Render an RGB image from the 3DGS map at the given frame's estimated pose.

    Returns HxWx3 uint8 BGR image.
    """
    h = camera_info["image_height"]
    w = camera_info["image_width"]
    k = np.array([
        [camera_info["fx"], 0.0, camera_info["cx"]],
        [0.0, camera_info["fy"], camera_info["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    identity_w2c = np.eye(4, dtype=np.float32)
    cam = setup_camera(w, h, k, identity_w2c)

    with torch.no_grad():
        transformed_gaussians = transform_to_frame(
            params, frame_idx, gaussians_grad=False, camera_grad=False,
        )
        rendervar = transformed_params2rendervar(params, transformed_gaussians)
        im, _, _ = Renderer(raster_settings=cam)(**rendervar)

    rgb = torch.clamp(im, 0, 1).detach().cpu().permute(1, 2, 0).numpy()
    return cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def render_from_pose(params: dict, c2w: np.ndarray, camera_info: dict) -> np.ndarray:
    """Render an RGB image from an arbitrary c2w pose (in first-camera frame).

    Returns HxWx3 uint8 BGR image.
    """
    h = camera_info["image_height"]
    w = camera_info["image_width"]
    k = np.array([
        [camera_info["fx"], 0.0, camera_info["cx"]],
        [0.0, camera_info["fy"], camera_info["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    w2c = np.linalg.inv(c2w).astype(np.float32)
    cam = setup_camera(w, h, k, w2c)

    with torch.no_grad():
        rendervar = params2rendervar(params)
        im, _, _ = Renderer(raster_settings=cam)(**rendervar)

    rgb = torch.clamp(im, 0, 1).detach().cpu().permute(1, 2, 0).numpy()
    return cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# VLM
# ---------------------------------------------------------------------------

def encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_prompt(rgb_path: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are given an RGB image from an indoor robot scene.\n"
                        "Use only evidence visible in the image.\n\n"
                        f"TASK:\n{TASK_QUESTION}"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encode_image_b64(rgb_path)}",
                        "detail": "high",
                    },
                },
            ],
        }
    ]


def call_vlm(messages: list, api_key: str, model: str) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(3):
        try:
            resp = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60.0,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            return json.loads(content)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: parse error (attempt {attempt+1}): {e}")
            if attempt == 2:
                return {"raw_response": content, "parse_error": str(e)}
        except Exception as e:
            print(f"  Warning: API error (attempt {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return {"api_error": str(e)}

    return {"error": "max retries exceeded"}




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Minimal VLM evaluation on RGB images")
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--image_idx", type=int, required=True, help="Frame index to evaluate")
    parser.add_argument("--use_rendered", action="store_true", help="Use rendered RGB instead of GT")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: provide --api_key or set OPENAI_API_KEY")
        sys.exit(1)

    eval_dir = os.path.join(args.experiment_dir, "eval")
    frames = load_frames_index(eval_dir)
    camera_info = load_camera_info(eval_dir)

    frame = get_frame(frames, args.image_idx)
    if frame is None:
        print(f"Error: frame {args.image_idx} not found in frames_index.csv")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(args.experiment_dir, str(args.image_idx))
    os.makedirs(output_dir, exist_ok=True)

    render_dir = os.path.join(output_dir, "novel_view")
    os.makedirs(render_dir, exist_ok=True)

    # Load 3DGS checkpoint and render from the frame's pose
    print(f"Loading 3DGS checkpoint...")
    params = load_checkpoint(args.experiment_dir)

    idx = frame["index"]
    c2w = frame["camera_pose"]
    tx, ty, tz = c2w_to_translation(c2w)
    roll, pitch, yaw = c2w_to_rpy_deg(c2w)
    rgb_path = frame["gs_rgb_path"] if args.use_rendered else frame["gt_rgb_path"]

    print(f"Experiment:  {args.experiment_dir}")
    print(f"Frame:       {idx}")
    print(f"Source:      {'rendered' if args.use_rendered else 'ground truth'} RGB")
    print(f"Model:       {args.model}")
    print(f"Pose (c2w):  t=[{tx:.3f}, {ty:.3f}, {tz:.3f}]  rpy=[{roll:.1f}, {pitch:.1f}, {yaw:.1f}] deg")
    print()

    # Render from this frame's pose using the 3DGS map
    rendered_path = os.path.join(render_dir, f"rendered_{idx:04d}.png")
    rendered_img = render_frame(params, idx, camera_info)
    cv2.imwrite(rendered_path, rendered_img)
    print(f"Rendered: {rendered_path}")

    # Render novel view: +1m on world Z axis (= +1m on first-camera-frame Y axis)
    novel_c2w = c2w.copy()
    
    # Translation offsets (applied in world frame)
    # negative X -> forward, positive X -> backward (in first-camera frame, X is right = world Y)
    # negative Y -> up, positive Y -> down (in first-camera frame, Y is up = world Z)
    # negative Z -> left, positive Z -> right (in first-camera frame, Z is forward = world X)
    novel_c2w[0, 3] += 0.0  # first-camera X axis = world Y (right)
    novel_c2w[1, 3] += 0.0  # first-camera Y axis = world Z (up)
    novel_c2w[2, 3] += 0.0  # first-camera Z axis = world X (forward)
    
    # Rotation offsets (applied in camera's local frame)
    # Positive rotation around first-camera X axis tilts up, negative tilts down
    # Positive rotation around first-camera Y axis pans right, negative pans left
    # Positive rotation around first-camera Z axis rolls clockwise, negative rolls counterclockwise
    novel_c2w[:3, :3] = novel_c2w[:3, :3] @ rotation_x(0.0)  # tilt up/down
    novel_c2w[:3, :3] = novel_c2w[:3, :3] @ rotation_y(0.0)  # pan left/right
    novel_c2w[:3, :3] = novel_c2w[:3, :3] @ rotation_z(0.0) # roll
    
    novel_tx, novel_ty, novel_tz = c2w_to_translation(novel_c2w)
    novel_roll, novel_pitch, novel_yaw = c2w_to_rpy_deg(novel_c2w)

    novel_path = os.path.join(render_dir, f"novel_{idx:04d}_z+1m.png")
    novel_img = render_from_pose(params, novel_c2w, camera_info)
    cv2.imwrite(novel_path, novel_img)
    print(f"Novel view (+1m Z): {novel_path}")
    print(f"  Pose (c2w): t=[{novel_tx:.3f}, {novel_ty:.3f}, {novel_tz:.3f}]  rpy=[{novel_roll:.1f}, {novel_pitch:.1f}, {novel_yaw:.1f}] deg")

    # Query VLM
    messages = build_prompt(rgb_path)
    response = call_vlm(messages, api_key=api_key, model=args.model)
    print(f"Response: {json.dumps(response, indent=2)}")

    # Save result
    output = {
        "experiment_dir": args.experiment_dir,
        "frame_index": idx,
        "use_rendered": args.use_rendered,
        "model": args.model,
        "task_question": TASK_QUESTION.strip(),
        "camera_info": camera_info,
        "rgb_path": rgb_path,
        "rendered_path": rendered_path,
        "gt_depth_path": frame["gt_depth_path"],
        "gs_depth_path": frame["gs_depth_path"],
        "camera_pose_c2w": c2w.tolist(),
        "translation": [tx, ty, tz],
        "rotation_rpy_deg": [roll, pitch, yaw],
        "novel_view": {
            "path": novel_path,
            "camera_pose_c2w": novel_c2w.tolist(),
            "translation": [novel_tx, novel_ty, novel_tz],
            "rotation_rpy_deg": [novel_roll, novel_pitch, novel_yaw],
            "offset_world_z_m": 1.0,
        },
        "response": response,
    }

    out_path = os.path.join(output_dir, "vlm_evaluation.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
