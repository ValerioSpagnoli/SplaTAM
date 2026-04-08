"""
Minimal VLM evaluation on a single frame.

Loads frame data from eval/frames_index.csv and intrinsics from eval/camera_info.json.
Renders an RGB image from the 3DGS map at the frame's camera pose, queries the VLM,
and saves the result.

Usage:
  python scripts/vlm_evaluation_general.py \
    --experiment_dir ./experiments/IsaacSim/office0_0 \
    --image_idx 42 --model gpt-4o [--use_rendered]

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
    params2depthplussilhouette,
    transform_to_frame,
    transformed_params2rendervar,
    transformed_params2depthplussilhouette,
)


OBJECT_1 = "Computer monitor"
OBJECT_2 = "Orange frame picture on one of the desks"
MAX_DETECTION_ROUNDS = 10
MAX_SPATIAL_ROUNDS = 5
TRANSLATION_STEP_M = 0.5
ROTATION_STEP_DEG = 45.0

DETECTION_PLANNING_PROMPT = f"""
You are controlling a robot camera in an indoor scene. You need to find both of these objects:
1. {OBJECT_1}
2. {OBJECT_2}

{{POSE_INFO}}

This is the image from the robot's current (initial) camera pose. Analyze the scene carefully.

Your task is to understand the scene layout and reason about where each object is likely to be. Do NOT plan specific camera movements — just describe what you see and where the objects might be.

Consider:
  - What is the overall scene layout?
  - Which objects (if any) are already visible in this image?
  - For missing objects: based on their nature and the scene layout, where are they most likely located?
  - Are there any visual clues that hint at where to look?

Return only valid JSON (no markdown, no extra text) with exactly this schema:
{{
    "object_1_visible": boolean,
    "object_2_visible": boolean,
    "both_visible": boolean,
    "scene_description": string,
    "object_1_location_hint": string,
    "object_2_location_hint": string,
    "suggested_search_direction": string,
    "reasoning": string
}}
"""

DETECTION_EXECUTION_PROMPT = f"""
Look at this image from an indoor robot scene. Do you see both of these objects?
1. {OBJECT_1}
2. {OBJECT_2}

{{POSE_INFO}}

Available camera movements:
  Translation ({TRANSLATION_STEP_M} m step): "up", "down", "left", "right", "forward", "backward"
  Rotation ({ROTATION_STEP_DEG} deg step):    "tilt_up", "tilt_down", "pan_left", "pan_right"

You have a maximum of {MAX_DETECTION_ROUNDS} attempts total. Do NOT repeat a movement that already failed.

"from_initial" field:
  - false → movement is applied ON TOP of the current camera pose (incremental).
  - true  → camera is RESET to the starting pose BEFORE applying the movement.

Scene analysis from initial observation:
{{SCENE_ANALYSIS}}

{{HISTORY}}

Use the scene analysis to guide your search. Adapt based on what you see in the current image.

Return only valid JSON (no markdown, no extra text) with exactly this schema:
{{
    "object_1_visible": boolean,
    "object_2_visible": boolean,
    "both_visible": boolean,
    "reasoning": string,
    "request_new_view": boolean,
    "from_initial": boolean,
    "requested_translation": "" | "up" | "down" | "left" | "right" | "forward" | "backward",
    "requested_rotation": "" | "tilt_up" | "tilt_down" | "pan_left" | "pan_right",
    "new_view_reasoning": string
}}
"""

SPATIAL_PROMPT = f"""
You can see both objects in this image:
1. {OBJECT_1}
2. {OBJECT_2}

{{POSE_INFO}}

Determine the spatial relationship of the first object relative to the second object along all three axes:
  - left/right (lateral)
  - in front of/behind (depth)
  - above/below (vertical)

If you are confident, return your answer. If the relationship is uncertain or ambiguous from this viewpoint, you may request a new camera view to disambiguate.

Available camera movements (you can combine one translation and one rotation):
  Translation ({TRANSLATION_STEP_M} m step): "up", "down", "left", "right", "forward", "backward"
  Rotation ({ROTATION_STEP_DEG} deg step):    "tilt_up", "tilt_down", "pan_left", "pan_right"

You have a maximum of {MAX_SPATIAL_ROUNDS} attempts to determine the spatial relationship.

IMPORTANT — "from_initial" field:
  - "from_initial": false → the requested movement is applied ON TOP of the current camera pose (incremental).
  - "from_initial": true  → the camera is RESET to the starting pose BEFORE applying the requested movement. Use this when the current viewpoint has drifted too far and you want to try a different angle from the original position.

{{HISTORY}}

Return only valid JSON (no markdown, no extra text) with exactly this schema:
{{
    "lateral": "to the left of" | "to the right of" | "aligned" | "uncertain",
    "depth": "in front of" | "behind" | "aligned" | "uncertain",
    "vertical": "above" | "below" | "aligned" | "uncertain",
    "confidence": float (0.0 to 1.0),
    "reasoning": string,
    "request_new_view": boolean,
    "from_initial": boolean,
    "requested_translation": "" | "up" | "down" | "left" | "right" | "forward" | "backward",
    "requested_rotation": "" | "tilt_up" | "tilt_down" | "pan_left" | "pan_right",
    "new_view_reasoning": string
}}
"""

# Maps VLM action strings to c2w offsets in first-camera frame
TRANSLATION_MAP = {
    "up":       (0.0, -TRANSLATION_STEP_M, 0.0),   # -Y = world up
    "down":     (0.0, +TRANSLATION_STEP_M, 0.0),   # +Y = world down
    "left":     (-TRANSLATION_STEP_M, 0.0, 0.0),   # -X = world left
    "right":    (+TRANSLATION_STEP_M, 0.0, 0.0),   # +X = world right
    "forward":  (0.0, 0.0, +TRANSLATION_STEP_M),   # +Z = world forward
    "backward": (0.0, 0.0, -TRANSLATION_STEP_M),   # -Z = world backward
}

ROTATION_MAP = {
    "tilt_up":   ("x", +ROTATION_STEP_DEG),
    "tilt_down": ("x", -ROTATION_STEP_DEG),
    "pan_left":  ("y", -ROTATION_STEP_DEG),
    "pan_right": ("y", +ROTATION_STEP_DEG),
}


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

def build_pose_info(initial_c2w: np.ndarray, current_c2w: np.ndarray) -> str:
    """Build a description of the robot's initial and current pose for the VLM."""
    ix, iy, iz = c2w_to_translation(initial_c2w)
    ir, ip, iyaw = c2w_to_rpy_deg(initial_c2w)
    cx, cy, cz = c2w_to_translation(current_c2w)
    cr, cp, cyaw = c2w_to_rpy_deg(current_c2w)

    lines = [
        "Robot camera pose information:",
        f"  Initial pose: position=({ix:.3f}, {iy:.3f}, {iz:.3f})  orientation=(roll={ir:.1f}°, pitch={ip:.1f}°, yaw={iyaw:.1f}°)",
        f"  Current pose: position=({cx:.3f}, {cy:.3f}, {cz:.3f})  orientation=(roll={cr:.1f}°, pitch={cp:.1f}°, yaw={cyaw:.1f}°)",
        "  Orientation meaning: yaw=0° means facing straight ahead. Negative yaw = looking left, positive yaw = looking right.",
        "  Pitch: negative = looking down, positive = looking up.",
        "  Movements are relative to the camera: 'left'/'right' translate sideways, 'forward'/'backward' along the viewing direction, 'up'/'down' vertically.",
    ]
    return "\n".join(lines)


def build_history_string(rounds_list: list[dict], phase: str = "detection") -> str:
    """Build a summary of previous rounds for the VLM to avoid repeating moves."""
    if not rounds_list:
        return ""
    lines = ["Previous attempts (do NOT repeat these):"]
    for r in rounds_list:
        resp = r.get("response", {})
        if not isinstance(resp, dict):
            continue
        rnd = r["round"]
        trans = resp.get("requested_translation", "")
        rot = resp.get("requested_rotation", "")
        from_init = resp.get("from_initial", False)
        move = []
        if trans:
            move.append(f"translate={trans}")
        if rot:
            move.append(f"rotate={rot}")
        move_str = ", ".join(move) if move else "none"
        base = "from_initial" if from_init else "incremental"
        if phase == "detection":
            obj1 = "visible" if resp.get("object_1_visible") else "not visible"
            obj2 = "visible" if resp.get("object_2_visible") else "not visible"
            lines.append(f"  Round {rnd}: {move_str} ({base}) → obj1={obj1}, obj2={obj2}")
        else:
            conf = resp.get("confidence", "?")
            lines.append(f"  Round {rnd}: {move_str} ({base}) → confidence={conf}")
    return "\n".join(lines)


def apply_view_request(c2w: np.ndarray, translation: str, rotation: str) -> np.ndarray:
    """Apply a VLM-requested camera movement to a c2w pose."""
    new_c2w = c2w.copy()

    if translation in TRANSLATION_MAP:
        dx, dy, dz = TRANSLATION_MAP[translation]
        new_c2w[0, 3] += dx
        new_c2w[1, 3] += dy
        new_c2w[2, 3] += dz

    if rotation in ROTATION_MAP:
        axis, angle = ROTATION_MAP[rotation]
        if axis == "x":
            new_c2w[:3, :3] = new_c2w[:3, :3] @ rotation_x(angle)
        elif axis == "y":
            new_c2w[:3, :3] = new_c2w[:3, :3] @ rotation_y(angle)
        elif axis == "z":
            new_c2w[:3, :3] = new_c2w[:3, :3] @ rotation_z(angle)

    return new_c2w


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


def render_from_pose(params: dict, c2w: np.ndarray, camera_info: dict) -> tuple[np.ndarray, np.ndarray]:
    """Render RGB and depth from an arbitrary c2w pose (in first-camera frame).

    Returns:
        rgb_bgr: HxWx3 uint8 BGR image.
        depth: HxW float32 metric depth in meters.
    """
    h = camera_info["image_height"]
    w = camera_info["image_width"]
    k = np.array([
        [camera_info["fx"], 0.0, camera_info["cx"]],
        [0.0, camera_info["fy"], camera_info["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    w2c = np.linalg.inv(c2w).astype(np.float32)
    w2c_torch = torch.tensor(w2c).cuda().float()
    cam = setup_camera(w, h, k, w2c)

    with torch.no_grad():
        rendervar = params2rendervar(params)
        im, _, _ = Renderer(raster_settings=cam)(**rendervar)

        depth_rendervar = params2depthplussilhouette(params, w2c_torch)
        depth_sil, _, _ = Renderer(raster_settings=cam)(**depth_rendervar)
        depth = depth_sil[0].detach().cpu().numpy().astype(np.float32)

    rgb = torch.clamp(im, 0, 1).detach().cpu().permute(1, 2, 0).numpy()
    return cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), depth


def depth_to_colormap(depth: np.ndarray, vmin: float = 0.0, vmax: float = 6.0) -> np.ndarray:
    """Convert metric depth to a JET colormap BGR image for VLM input."""
    normalized = np.clip((depth - vmin) / (vmax - vmin), 0, 1)
    return cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)


# ---------------------------------------------------------------------------
# VLM
# ---------------------------------------------------------------------------

def encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_prompt(rgb_path: str, task_prompt: str, depth_path: str | None = None) -> list[dict]:
    text = "You are given an RGB image from an indoor robot scene.\n"
    if depth_path is not None:
        text += "You are also given a depth map (colorized, blue=near, red=far) aligned with the RGB image. Use it as geometric context.\n"
    text += f"Use only evidence visible in the provided inputs.\n\nTASK:\n{task_prompt}"

    content = [
        {"type": "text", "text": text},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encode_image_b64(rgb_path)}",
                "detail": "high",
            },
        },
    ]

    if depth_path is not None:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encode_image_b64(depth_path)}",
                "detail": "high",
            },
        })

    return [{"role": "user", "content": content}]


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
    parser.add_argument("--use_depth", action="store_true", help="Include depth map in VLM prompt")
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

    print(f"{'='*70}")
    print(f"  VLM Spatial Evaluation")
    print(f"{'='*70}")
    print(f"  Experiment:    {args.experiment_dir}")
    print(f"  Frame:         {idx}")
    print(f"  Source:        {'rendered' if args.use_rendered else 'ground truth'} RGB")
    print(f"  Depth:         {'enabled' if args.use_depth else 'disabled'}")
    print(f"  Model:         {args.model}")
    print(f"  Objects:       1. {OBJECT_1}")
    print(f"                 2. {OBJECT_2}")
    print(f"  Initial pose:  t=[{tx:.3f}, {ty:.3f}, {tz:.3f}]  rpy=[{roll:.1f}, {pitch:.1f}, {yaw:.1f}] deg")
    print(f"  Max detection: {MAX_DETECTION_ROUNDS} rounds")
    print(f"  Max spatial:   {MAX_SPATIAL_ROUNDS} rounds")
    print(f"{'='*70}")

    # Helper to render and get depth for a given pose
    def render_and_get_paths(pose, tag):
        rgb_file = os.path.join(render_dir, f"{tag}.png")
        rgb_img, depth_arr = render_from_pose(params, pose, camera_info)
        cv2.imwrite(rgb_file, rgb_img)
        depth_file = None
        if args.use_depth:
            depth_file = os.path.join(render_dir, f"{tag}_depth.png")
            cv2.imwrite(depth_file, depth_to_colormap(depth_arr))
        return rgb_file, depth_file

    def print_pose(c2w_mat, indent="  "):
        ptx, pty, ptz = c2w_to_translation(c2w_mat)
        pr, pp, py_ = c2w_to_rpy_deg(c2w_mat)
        print(f"{indent}Pose: t=[{ptx:.3f}, {pty:.3f}, {ptz:.3f}]  rpy=[{pr:.1f}, {pp:.1f}, {py_:.1f}] deg")

    def print_vlm_response(resp, indent="  "):
        for line in json.dumps(resp, indent=2).splitlines():
            print(f"{indent}{line}")

    # Resolve initial image and depth
    rendered_path = os.path.join(render_dir, f"rendered_{idx:04d}.png")
    rendered_img = render_frame(params, idx, camera_info)
    cv2.imwrite(rendered_path, rendered_img)

    initial_depth_path = None
    if args.use_depth:
        initial_depth = frame["gs_depth_path"] if args.use_rendered else frame["gt_depth_path"]
        if os.path.isfile(initial_depth):
            initial_depth_path = initial_depth
        else:
            print(f"  Warning: depth not found: {initial_depth}, proceeding without")

    # ===================================================================
    # Phase 1: Object Detection
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 1: OBJECT DETECTION")
    print(f"{'='*70}")

    detection_rounds = []
    current_c2w = c2w.copy()
    current_rgb = rgb_path
    current_depth = initial_depth_path
    both_visible = False

    # --- Planning stage ---
    print(f"\n  {'─'*50}")
    print(f"  Detection Planning")
    print(f"  {'─'*50}")
    print_pose(current_c2w, indent="    ")
    print(f"    Image: {current_rgb}")

    pose_info = build_pose_info(c2w, current_c2w)
    planning_prompt = DETECTION_PLANNING_PROMPT.replace("{POSE_INFO}", pose_info)
    planning_messages = build_prompt(current_rgb, planning_prompt, depth_path=current_depth)
    planning_response = call_vlm(planning_messages, api_key=api_key, model=args.model)

    print(f"    Planning Response:")
    print_vlm_response(planning_response, indent="      ")

    # Extract scene analysis to pass into each execution round
    if isinstance(planning_response, dict):
        scene_analysis_parts = []
        if planning_response.get("scene_description"):
            scene_analysis_parts.append(f"Scene: {planning_response['scene_description']}")
        if planning_response.get("object_1_location_hint"):
            scene_analysis_parts.append(f"{OBJECT_1} hint: {planning_response['object_1_location_hint']}")
        if planning_response.get("object_2_location_hint"):
            scene_analysis_parts.append(f"{OBJECT_2} hint: {planning_response['object_2_location_hint']}")
        if planning_response.get("suggested_search_direction"):
            scene_analysis_parts.append(f"Suggested direction: {planning_response['suggested_search_direction']}")
        scene_analysis_str = "\n".join(scene_analysis_parts) if scene_analysis_parts else "No scene analysis available."
        # Check if both are already visible from the planning image
        if planning_response.get("both_visible", False):
            both_visible = True
            print(f"    >> Both objects already visible! Skipping detection loop.")
            response = planning_response
    else:
        scene_analysis_str = "No scene analysis available — explore freely."

    print(f"\n    Scene analysis:")
    for line in scene_analysis_str.splitlines():
        print(f"      {line}")

    # --- Execution loop ---
    if not both_visible:
        for det_round in range(1, MAX_DETECTION_ROUNDS + 1):
            print(f"\n  {'─'*50}")
            print(f"  Detection Round {det_round}/{MAX_DETECTION_ROUNDS}")
            print(f"  {'─'*50}")
            print_pose(current_c2w, indent="    ")
            print(f"    Image: {current_rgb}")
            if current_depth:
                print(f"    Depth: {current_depth}")

            pose_info = build_pose_info(c2w, current_c2w)
            history = build_history_string(detection_rounds, phase="detection")
            if history:
                print(f"    History passed to VLM:")
                for line in history.splitlines():
                    print(f"      {line}")
            prompt = DETECTION_EXECUTION_PROMPT.replace("{POSE_INFO}", pose_info).replace("{SCENE_ANALYSIS}", scene_analysis_str).replace("{HISTORY}", history)
            messages = build_prompt(current_rgb, prompt, depth_path=current_depth)
            response = call_vlm(messages, api_key=api_key, model=args.model)

            print(f"    VLM Response:")
            print_vlm_response(response, indent="      ")

            rtx, rty, rtz = c2w_to_translation(current_c2w)
            rroll, rpitch, ryaw = c2w_to_rpy_deg(current_c2w)
            detection_rounds.append({
                "round": det_round,
                "rgb_path": current_rgb,
                "depth_path": current_depth,
                "camera_pose_c2w": current_c2w.tolist(),
                "translation": [rtx, rty, rtz],
                "rotation_rpy_deg": [rroll, rpitch, ryaw],
                "response": response,
            })

            if not isinstance(response, dict):
                print(f"    Result: Invalid response, stopping.")
                break

            obj1 = response.get("object_1_visible", False)
            obj2 = response.get("object_2_visible", False)
            both_visible = response.get("both_visible", False)
            print(f"    Visibility: obj1={'YES' if obj1 else 'NO'}  obj2={'YES' if obj2 else 'NO'}")

            if both_visible:
                print(f"    >> Both objects detected! Moving to Phase 2.")
                break

            if not response.get("request_new_view", False):
                print(f"    >> VLM did not request a new view, stopping.")
                break

            req_trans = response.get("requested_translation", "").strip()
            req_rot = response.get("requested_rotation", "").strip()
            if not req_trans and not req_rot:
                print(f"    >> No movement specified, stopping.")
                break

            from_initial = response.get("from_initial", False)
            base_label = "initial" if from_initial else "current"
            print(f"    >> Requesting new view (from {base_label}): translate=[{req_trans or 'none'}] rotate=[{req_rot or 'none'}]")

            base_c2w = c2w.copy() if from_initial else current_c2w
            current_c2w = apply_view_request(base_c2w, req_trans, req_rot)
            current_rgb, current_depth = render_and_get_paths(
                current_c2w, f"detect_{idx:04d}_round{det_round + 1}")

    if not both_visible:
        obj1_vis = response.get("object_1_visible", False) if isinstance(response, dict) else False
        obj2_vis = response.get("object_2_visible", False) if isinstance(response, dict) else False
        missing = []
        if not obj1_vis:
            missing.append(OBJECT_1)
        if not obj2_vis:
            missing.append(OBJECT_2)

        print(f"\n  {'='*50}")
        print(f"  RESULT: DETECTION FAILED")
        print(f"  Rounds used: {len(detection_rounds)}/{MAX_DETECTION_ROUNDS}")
        print(f"  Missing: {', '.join(missing)}")
        print(f"  {'='*50}")

        output = {
            "experiment_dir": args.experiment_dir,
            "frame_index": idx,
            "use_rendered": args.use_rendered,
            "use_depth": args.use_depth,
            "model": args.model,
            "object_1": OBJECT_1,
            "object_2": OBJECT_2,
            "status": "detection_failed",
            "missing_objects": missing,
            "camera_info": camera_info,
            "detection_planning": planning_response,
            "detection_rounds": detection_rounds,
        }

        out_path = os.path.join(output_dir, "vlm_evaluation.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to: {out_path}")
        return

    # ===================================================================
    # Phase 2: Spatial Relationship
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 2: SPATIAL RELATIONSHIP")
    print(f"{'='*70}")

    spatial_rounds = []
    for sp_round in range(1, MAX_SPATIAL_ROUNDS + 1):
        print(f"\n  {'─'*50}")
        print(f"  Spatial Round {sp_round}/{MAX_SPATIAL_ROUNDS}")
        print(f"  {'─'*50}")
        print_pose(current_c2w, indent="    ")
        print(f"    Image: {current_rgb}")
        if current_depth:
            print(f"    Depth: {current_depth}")

        pose_info = build_pose_info(c2w, current_c2w)
        history = build_history_string(spatial_rounds, phase="spatial")
        if history:
            print(f"    History passed to VLM:")
            for line in history.splitlines():
                print(f"      {line}")
        prompt = SPATIAL_PROMPT.replace("{POSE_INFO}", pose_info).replace("{HISTORY}", history)
        messages = build_prompt(current_rgb, prompt, depth_path=current_depth)
        response = call_vlm(messages, api_key=api_key, model=args.model)

        print(f"    VLM Response:")
        print_vlm_response(response, indent="      ")

        rtx, rty, rtz = c2w_to_translation(current_c2w)
        rroll, rpitch, ryaw = c2w_to_rpy_deg(current_c2w)
        spatial_rounds.append({
            "round": sp_round,
            "rgb_path": current_rgb,
            "depth_path": current_depth,
            "camera_pose_c2w": current_c2w.tolist(),
            "translation": [rtx, rty, rtz],
            "rotation_rpy_deg": [rroll, rpitch, ryaw],
            "response": response,
        })

        if not isinstance(response, dict):
            print(f"    Result: Invalid response, stopping.")
            break

        if not response.get("request_new_view", False):
            lat = response.get("lateral", "?")
            dep = response.get("depth", "?")
            vert = response.get("vertical", "?")
            conf = response.get("confidence", "?")
            print(f"    >> Final answer: lateral={lat}  depth={dep}  vertical={vert}  confidence={conf}")
            break

        req_trans = response.get("requested_translation", "").strip()
        req_rot = response.get("requested_rotation", "").strip()
        if not req_trans and not req_rot:
            print(f"    >> No movement specified, stopping.")
            break

        from_initial = response.get("from_initial", False)
        base_label = "initial" if from_initial else "current"
        print(f"    >> Requesting new view (from {base_label}): translate=[{req_trans or 'none'}] rotate=[{req_rot or 'none'}]")

        base_c2w = c2w.copy() if from_initial else current_c2w
        current_c2w = apply_view_request(base_c2w, req_trans, req_rot)
        current_rgb, current_depth = render_and_get_paths(
            current_c2w, f"spatial_{idx:04d}_round{sp_round + 1}")

    # ===================================================================
    # Summary
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Status:           completed")
    print(f"  Detection rounds: {len(detection_rounds)}")
    print(f"  Spatial rounds:   {len(spatial_rounds)}")
    if isinstance(response, dict) and "lateral" in response:
        print(f"  Final answer:")
        print(f"    Lateral:    {response.get('lateral', '?')}")
        print(f"    Depth:      {response.get('depth', '?')}")
        print(f"    Vertical:   {response.get('vertical', '?')}")
        print(f"    Confidence: {response.get('confidence', '?')}")
    print(f"{'='*70}")

    # Save result
    output = {
        "experiment_dir": args.experiment_dir,
        "frame_index": idx,
        "use_rendered": args.use_rendered,
        "use_depth": args.use_depth,
        "model": args.model,
        "object_1": OBJECT_1,
        "object_2": OBJECT_2,
        "status": "completed",
        "camera_info": camera_info,
        "detection_planning": planning_response,
        "detection_rounds": detection_rounds,
        "spatial_rounds": spatial_rounds,
    }

    out_path = os.path.join(output_dir, "vlm_evaluation.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
