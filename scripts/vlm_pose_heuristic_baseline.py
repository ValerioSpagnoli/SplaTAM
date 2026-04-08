"""
Heuristic viewpoint-selection baseline for the spatial relation task.

This script compares three methods on the same frame/task:
1) control: iterative VLM search from scripts/vlm_evaluation_general.py
2) clip: sample candidate poses, score with CLIP image-text similarity, pick best
3) vlm: sample candidate poses, score each with one-shot VLM visibility query, pick best

After selecting a viewpoint (clip/vlm methods), the script runs:
- one-shot detection check at the selected view
- full spatial reasoning loop from that selected view

Usage example:
  pixi run python scripts/vlm_pose_heuristic_baseline.py \
    --experiment_dir ./experiments/IsaacSim/office0_0 \
    --image_idx 42 \
    --methods control,clip,vlm \
    --azimuth_step_deg 30 \
    --elevation_values_deg -15,0,15,30 \
    --distance_values_m 0.5,1.0,2.0 \
    --num_runs 3 \
    --model gpt-4o


---- Methods ----  
control:
    Runs your original iterative VLM search from experiment 2 (movement requests over rounds).
    Branch is here: vlm_pose_heuristic_baseline.py:755
clip:
    Samples candidate poses, renders each, scores with CLIP text-image similarity, picks best.
    Then does detection confirmation and spatial phase.
    Branch is here: vlm_pose_heuristic_baseline.py:783
vlm:
    Samples candidate poses, asks VLM one-shot visibility per candidate, scores/ranks, picks best.
    Then continues with spatial phase.
    Branch is here: vlm_pose_heuristic_baseline.py:798
    
---- Azimuth step ----
Defined in vlm_pose_heuristic_baseline.py:658, used in vlm_pose_heuristic_baseline.py:111.
It creates azimuth angles from 0_deg to 360_deg with that step size, e.g. 30_deg gives 12 bins: 0,30,...,330.

---- Elevation values ---
Defined in vlm_pose_heuristic_baseline.py:659, used in vlm_pose_heuristic_baseline.py:157.
It is an explicit list of pitch-like vertical sampling angles
The position offset uses:
dy = -radius_m * math.sin(el) in vlm_pose_herustic_baseline.py:131. E.g. with values -15,0,15,30:
With this axis convention, negative elevation moves the camera up, positive elevation moves it down.

---- Distance values ---
Defined in vlm_pose_heuristic_baseline.py:660, used in vlm_pose_heuristic_baseline.py:156.
It is the radius list around the base pose, in meters. Larger values sample farther viewpoints.

---- Candidate generation ----
Candidate generation is a full cartesian product in vlm_pose_heuristic_baseline.py:156
Total candidates = (#azimuth_bins) * (#elevation_values) * (#distance_values)
E.g.
* azimuth_step_deg=30 gives 12 azimuth bins
* elevation_values_deg=-15,0,15,30 gives 4 elevation values
* distance_values_m=0.5,1.0,2.0 gives 3 distance values
-> total candidates = 12 * 4 * 3 = 144
"""

import argparse
import importlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

from scripts.vlm_evaluation_general import (
    OBJECT_1,
    OBJECT_2,
    MAX_SPATIAL_ROUNDS,
    build_pose_info,
    build_prompt,
    call_vlm,
    apply_view_request,
    render_from_pose,
    depth_to_colormap,
    load_checkpoint,
    load_frames_index,
    load_camera_info,
    get_frame,
    c2w_to_translation,
    c2w_to_rpy_deg,
    run_single_experiment,
    SPATIAL_PROMPT,
)


DETECTION_ONE_SHOT_PROMPT = f"""
Look at this image from an indoor robot scene. Determine if you can see these objects:
1. {OBJECT_1}
2. {OBJECT_2}

{{POSE_INFO}}

Return only valid JSON (no markdown, no extra text) with exactly this schema:
{{
    "object_1_visible": boolean,
    "object_2_visible": boolean,
    "both_visible": boolean,
    "confidence": float,
    "reasoning": string
}}
"""


@dataclass
class CandidatePose:
    candidate_id: int
    azimuth_deg: float
    elevation_deg: float
    radius_m: float
    c2w: np.ndarray


def parse_float_csv(text: str) -> list[float]:
    values = []
    for p in text.split(","):
        p = p.strip()
        if not p:
            continue
        values.append(float(p))
    return values


def build_azimuth_values(step_deg: float) -> list[float]:
    if step_deg <= 0:
        raise ValueError("azimuth_step_deg must be > 0")
    vals = np.arange(0.0, 360.0, step_deg, dtype=np.float64)
    return [float(v) for v in vals.tolist()]


def make_candidate_pose(base_c2w: np.ndarray, azimuth_deg: float, elevation_deg: float, radius_m: float) -> np.ndarray:
    """Create a candidate pose around base pose in first-camera frame.

    Convention reused from vlm_evaluation_general.py:
    - +X: right, -X: left
    - +Z: forward, -Z: backward
    - -Y: up, +Y: down
    """
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)

    dx = radius_m * math.cos(el) * math.sin(az)
    dz = radius_m * math.cos(el) * math.cos(az)
    dy = -radius_m * math.sin(el)

    out = base_c2w.copy()
    out[0, 3] += dx
    out[1, 3] += dy
    out[2, 3] += dz

    # Rotate heading and tilt to match sampled angle in local camera frame.
    from scripts.vlm_evaluation_general import rotation_x, rotation_y

    out[:3, :3] = out[:3, :3] @ rotation_y(azimuth_deg) @ rotation_x(elevation_deg)
    return out


def generate_candidates(
    base_c2w: np.ndarray,
    azimuth_step_deg: float,
    elevation_values_deg: list[float],
    distance_values_m: list[float],
    max_candidates: int,
) -> list[CandidatePose]:
    azimuth_values = build_azimuth_values(azimuth_step_deg)
    candidates: list[CandidatePose] = []
    cid = 0

    for radius in distance_values_m:
        for elevation in elevation_values_deg:
            for azimuth in azimuth_values:
                c2w = make_candidate_pose(base_c2w, azimuth, elevation, radius)
                candidates.append(
                    CandidatePose(
                        candidate_id=cid,
                        azimuth_deg=azimuth,
                        elevation_deg=elevation,
                        radius_m=radius,
                        c2w=c2w,
                    )
                )
                cid += 1

    if max_candidates > 0 and len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]

    return candidates


def render_candidate_views(
    params: dict,
    camera_info: dict,
    candidates: list[CandidatePose],
    out_dir: str,
    use_depth: bool,
) -> list[dict]:
    os.makedirs(out_dir, exist_ok=True)
    rendered = []

    for cand in candidates:
        t0 = time.time()
        rgb_bgr, depth = render_from_pose(params, cand.c2w, camera_info)
        rgb_path = os.path.join(out_dir, f"cand_{cand.candidate_id:04d}.png")
        cv2.imwrite(rgb_path, rgb_bgr)

        depth_path = None
        if use_depth:
            depth_path = os.path.join(out_dir, f"cand_{cand.candidate_id:04d}_depth.png")
            cv2.imwrite(depth_path, depth_to_colormap(depth))

        rendered.append(
            {
                "candidate": cand,
                "rgb_path": rgb_path,
                "depth_path": depth_path,
                "render_time_sec": time.time() - t0,
            }
        )

    return rendered


class ClipScorer:
    def __init__(self, model_name: str, pretrained: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.pretrained = pretrained

        try:
            open_clip = importlib.import_module("open_clip")
            from PIL import Image
        except ImportError as e:
            raise RuntimeError(
                "open_clip is required for --methods clip. Install open-clip-torch."
            ) from e

        self.open_clip = open_clip
        self.Image = Image
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def score(self, image_path: str, text_query: str) -> float:
        image = self.Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        text_tokens = self.tokenizer([text_query]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tokens)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            sim = (image_features @ text_features.T).squeeze().item()

        return float(sim)


def detection_one_shot(
    initial_c2w: np.ndarray,
    current_c2w: np.ndarray,
    rgb_path: str,
    depth_path: str | None,
    api_key: str,
    model: str,
) -> dict:
    pose_info = build_pose_info(initial_c2w, current_c2w)
    prompt = DETECTION_ONE_SHOT_PROMPT.replace("{POSE_INFO}", pose_info)
    messages = build_prompt(rgb_path, prompt, depth_path=depth_path)
    response = call_vlm(messages, api_key=api_key, model=model)
    return response


def score_candidates_with_clip(rendered_candidates: list[dict], text_query: str, clip_scorer: ClipScorer) -> tuple[list[dict], dict]:
    scored = []
    t0 = time.time()

    for item in rendered_candidates:
        score = clip_scorer.score(item["rgb_path"], text_query)
        scored.append(
            {
                "candidate_id": item["candidate"].candidate_id,
                "score": score,
                "rgb_path": item["rgb_path"],
                "depth_path": item["depth_path"],
                "azimuth_deg": item["candidate"].azimuth_deg,
                "elevation_deg": item["candidate"].elevation_deg,
                "radius_m": item["candidate"].radius_m,
                "camera_pose_c2w": item["candidate"].c2w.tolist(),
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    stats = {
        "method": "clip",
        "num_candidates": len(scored),
        "clip_calls": len(scored),
        "vlm_calls": 0,
        "scoring_time_sec": time.time() - t0,
    }
    return scored, stats


def score_candidates_with_vlm(
    rendered_candidates: list[dict],
    initial_c2w: np.ndarray,
    api_key: str,
    model: str,
) -> tuple[list[dict], dict]:
    scored = []
    t0 = time.time()

    for item in rendered_candidates:
        cand = item["candidate"]
        response = detection_one_shot(
            initial_c2w=initial_c2w,
            current_c2w=cand.c2w,
            rgb_path=item["rgb_path"],
            depth_path=item["depth_path"],
            api_key=api_key,
            model=model,
        )
        obj1 = bool(response.get("object_1_visible", False)) if isinstance(response, dict) else False
        obj2 = bool(response.get("object_2_visible", False)) if isinstance(response, dict) else False
        both = bool(response.get("both_visible", False)) if isinstance(response, dict) else False
        conf = float(response.get("confidence", 0.0)) if isinstance(response, dict) else 0.0

        # Primary signal: both visible. Secondary: single-object visibility and confidence.
        score = (2.0 if both else 0.0) + (0.5 if obj1 else 0.0) + (0.5 if obj2 else 0.0) + 0.05 * conf

        scored.append(
            {
                "candidate_id": cand.candidate_id,
                "score": score,
                "vlm_detection": response,
                "rgb_path": item["rgb_path"],
                "depth_path": item["depth_path"],
                "azimuth_deg": cand.azimuth_deg,
                "elevation_deg": cand.elevation_deg,
                "radius_m": cand.radius_m,
                "camera_pose_c2w": cand.c2w.tolist(),
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    stats = {
        "method": "vlm",
        "num_candidates": len(scored),
        "clip_calls": 0,
        "vlm_calls": len(scored),
        "scoring_time_sec": time.time() - t0,
    }
    return scored, stats


def run_spatial_phase(
    initial_c2w: np.ndarray,
    start_c2w: np.ndarray,
    start_rgb_path: str,
    start_depth_path: str | None,
    params: dict,
    camera_info: dict,
    render_dir: str,
    run_id: int,
    frame_idx: int,
    use_depth: bool,
    api_key: str,
    model: str,
) -> tuple[dict, list[dict], int]:
    current_c2w = start_c2w.copy()
    current_rgb = start_rgb_path
    current_depth = start_depth_path
    rounds = []
    vlm_calls = 0

    for sp_round in range(1, MAX_SPATIAL_ROUNDS + 1):
        pose_info = build_pose_info(initial_c2w, current_c2w)

        history_lines = []
        if rounds:
            history_lines.append("Previous attempts:")
            for r in rounds:
                rr = r["response"] if isinstance(r.get("response"), dict) else {}
                history_lines.append(
                    f"  Round {r['round']}: conf={rr.get('confidence', '?')} "
                    f"lat={rr.get('lateral', '?')} depth={rr.get('depth', '?')} vert={rr.get('vertical', '?')}"
                )
        history = "\n".join(history_lines)

        prompt = SPATIAL_PROMPT.replace("{POSE_INFO}", pose_info).replace("{HISTORY}", history)
        messages = build_prompt(current_rgb, prompt, depth_path=current_depth)
        response = call_vlm(messages, api_key=api_key, model=model)
        vlm_calls += 1

        tx, ty, tz = c2w_to_translation(current_c2w)
        rr, rp, ry = c2w_to_rpy_deg(current_c2w)
        rounds.append(
            {
                "round": sp_round,
                "rgb_path": current_rgb,
                "depth_path": current_depth,
                "camera_pose_c2w": current_c2w.tolist(),
                "translation": [tx, ty, tz],
                "rotation_rpy_deg": [rr, rp, ry],
                "response": response,
            }
        )

        if not isinstance(response, dict):
            break

        if not response.get("request_new_view", False):
            return response, rounds, vlm_calls

        req_trans = response.get("requested_translation", "").strip()
        req_rot = response.get("requested_rotation", "").strip()
        if not req_trans and not req_rot:
            break

        from_initial = bool(response.get("from_initial", False))
        base_c2w = initial_c2w.copy() if from_initial else current_c2w
        current_c2w = apply_view_request(base_c2w, req_trans, req_rot)

        rgb_bgr, depth = render_from_pose(params, current_c2w, camera_info)
        current_rgb = os.path.join(render_dir, f"run{run_id}_spatial_{frame_idx:04d}_round{sp_round + 1}.png")
        cv2.imwrite(current_rgb, rgb_bgr)

        current_depth = None
        if use_depth:
            current_depth = os.path.join(render_dir, f"run{run_id}_spatial_{frame_idx:04d}_round{sp_round + 1}_depth.png")
            cv2.imwrite(current_depth, depth_to_colormap(depth))

    return {
        "lateral": "?",
        "depth": "?",
        "vertical": "?",
        "confidence": 0.0,
        "reasoning": "Spatial phase ended without final answer",
    }, rounds, vlm_calls


def run_heuristic_method(
    method_name: str,
    run_id: int,
    frame: dict,
    camera_info: dict,
    base_c2w: np.ndarray,
    params: dict,
    rendered_candidates: list[dict],
    args,
    run_dir: str,
    clip_scorer: ClipScorer | None,
) -> dict:
    start_time = time.time()

    if method_name == "clip":
        assert clip_scorer is not None
        scored, score_stats = score_candidates_with_clip(
            rendered_candidates,
            text_query=args.clip_query,
            clip_scorer=clip_scorer,
        )
    elif method_name == "vlm":
        scored, score_stats = score_candidates_with_vlm(
            rendered_candidates,
            initial_c2w=base_c2w,
            api_key=args.api_key,
            model=args.model,
        )
    else:
        raise ValueError(f"Unknown heuristic method: {method_name}")

    if not scored:
        return {
            "method": method_name,
            "status": "detection_failed",
            "reason": "No candidates available",
            "cost": {
                "num_candidates": 0,
                "num_renders": 0,
                "vlm_calls": 0,
                "clip_calls": 0,
                "wall_time_sec": time.time() - start_time,
            },
        }

    top_k = scored[: max(1, args.top_k)]
    best = top_k[0]

    # Detection check for selected pose.
    selected_c2w = np.array(best["camera_pose_c2w"], dtype=np.float64)
    selected_rgb = best["rgb_path"]
    selected_depth = best.get("depth_path")

    if method_name == "vlm" and isinstance(best.get("vlm_detection"), dict):
        det_resp = best["vlm_detection"]
        det_vlm_calls = 0
    else:
        det_resp = detection_one_shot(
            initial_c2w=base_c2w,
            current_c2w=selected_c2w,
            rgb_path=selected_rgb,
            depth_path=selected_depth,
            api_key=args.api_key,
            model=args.model,
        )
        det_vlm_calls = 1

    both_visible = bool(det_resp.get("both_visible", False)) if isinstance(det_resp, dict) else False

    spatial_answer = None
    spatial_rounds = []
    spatial_vlm_calls = 0
    status = "detection_failed"
    if both_visible and not args.skip_spatial:
        spatial_answer, spatial_rounds, spatial_vlm_calls = run_spatial_phase(
            initial_c2w=base_c2w,
            start_c2w=selected_c2w,
            start_rgb_path=selected_rgb,
            start_depth_path=selected_depth,
            params=params,
            camera_info=camera_info,
            render_dir=os.path.join(run_dir, f"{method_name}_spatial_views"),
            run_id=run_id,
            frame_idx=frame["index"],
            use_depth=args.use_depth,
            api_key=args.api_key,
            model=args.model,
        )
        status = "completed"
    elif both_visible and args.skip_spatial:
        spatial_answer = {
            "lateral": "skipped",
            "depth": "skipped",
            "vertical": "skipped",
            "confidence": 0.0,
        }
        status = "completed"

    result = {
        "method": method_name,
        "status": status,
        "object_1": OBJECT_1,
        "object_2": OBJECT_2,
        "frame_index": frame["index"],
        "selected_candidate": best,
        "top_k_candidates": top_k,
        "detection_response": det_resp,
        "both_visible_after_selection": both_visible,
        "spatial_rounds_count": len(spatial_rounds),
        "spatial_rounds": spatial_rounds,
        "final_spatial_answer": spatial_answer,
        "score_stats": score_stats,
    }

    vlm_scoring_calls = score_stats["vlm_calls"]
    clip_scoring_calls = score_stats["clip_calls"]
    result["cost"] = {
        "num_candidates": score_stats["num_candidates"],
        "num_renders": score_stats["num_candidates"],
        "vlm_calls": vlm_scoring_calls + det_vlm_calls + spatial_vlm_calls,
        "clip_calls": clip_scoring_calls,
        "scoring_time_sec": score_stats["scoring_time_sec"],
        "wall_time_sec": time.time() - start_time,
    }
    return result


def estimate_control_cost(control_result: dict) -> dict:
    det_rounds = int(control_result.get("detection_rounds_count", 0))
    sp_rounds = int(control_result.get("spatial_rounds_count", 0))
    # Approximation: planning + detection rounds + spatial rounds.
    vlm_calls = 1 + det_rounds + sp_rounds
    return {
        "num_candidates": 0,
        "num_renders": det_rounds + sp_rounds,
        "vlm_calls": vlm_calls,
        "clip_calls": 0,
    }


def aggregate_method_results(all_runs: list[dict], methods: list[str]) -> dict:
    by_method: dict[str, list[dict]] = {m: [] for m in methods}
    for run in all_runs:
        for m in methods:
            if m in run:
                by_method[m].append(run[m])

    out = {"methods": {}}
    for m, vals in by_method.items():
        total = len(vals)
        completed = [v for v in vals if v.get("status") == "completed"]
        detection_failed = [v for v in vals if v.get("status") == "detection_failed"]

        total_vlm_calls = sum(v.get("cost", {}).get("vlm_calls", 0) for v in vals)
        total_clip_calls = sum(v.get("cost", {}).get("clip_calls", 0) for v in vals)
        total_renders = sum(v.get("cost", {}).get("num_renders", 0) for v in vals)
        total_time = sum(v.get("cost", {}).get("wall_time_sec", 0.0) for v in vals)

        out["methods"][m] = {
            "num_runs": total,
            "num_completed": len(completed),
            "num_detection_failed": len(detection_failed),
            "completion_rate": (len(completed) / total) if total > 0 else 0.0,
            "avg_vlm_calls": (total_vlm_calls / total) if total > 0 else 0.0,
            "avg_clip_calls": (total_clip_calls / total) if total > 0 else 0.0,
            "avg_renders": (total_renders / total) if total > 0 else 0.0,
            "avg_wall_time_sec": (total_time / total) if total > 0 else 0.0,
        }

    return out


def parse_methods(methods_str: str) -> list[str]:
    methods = [m.strip().lower() for m in methods_str.split(",") if m.strip()]
    valid = {"control", "clip", "vlm"}
    bad = [m for m in methods if m not in valid]
    if bad:
        raise ValueError(f"Invalid methods: {bad}. Valid methods are: control, clip, vlm")
    # Remove duplicates while preserving order.
    uniq = []
    for m in methods:
        if m not in uniq:
            uniq.append(m)
    return uniq


def normalize_cli_args(argv: list[str]) -> list[str]:
    """Normalize argv so optional flags can accept negative-leading CSV values.

    Example:
      --elevation_values_deg -15,0,15,30
    becomes:
      --elevation_values_deg=-15,0,15,30
    """
    value_flags = {
        "--elevation_values_deg",
        "--distance_values_m",
    }

    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in value_flags and i + 1 < len(argv):
            nxt = argv[i + 1]
            if nxt.startswith("-") and ("," in nxt or nxt.replace(".", "", 1).lstrip("-").isdigit()):
                out.append(f"{tok}={nxt}")
                i += 2
                continue
        out.append(tok)
        i += 1
    return out


def main():
    parser = argparse.ArgumentParser(description="Heuristic viewpoint selection baseline")
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--image_idx", type=int, required=True)
    parser.add_argument("--use_rendered", action="store_true", help="Use rendered RGB instead of GT for control init")
    parser.add_argument("--use_depth", action="store_true", help="Include depth maps in VLM prompts")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--methods", type=str, default="control,clip,vlm")

    parser.add_argument("--azimuth_step_deg", type=float, default=30.0)
    parser.add_argument("--elevation_values_deg", type=str, default="-15,0,15,30")
    parser.add_argument("--distance_values_m", type=str, default="0.5,1.0,2.0")
    parser.add_argument("--max_candidates", type=int, default=0, help="0 means keep full grid")
    parser.add_argument("--top_k", type=int, default=5)

    parser.add_argument("--clip_query", type=str, default=f"{OBJECT_1}. {OBJECT_2}.")
    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="openai")

    parser.add_argument("--skip_spatial", action="store_true", help="Only evaluate detection success after viewpoint selection")

    args = parser.parse_args(normalize_cli_args(sys.argv[1:]))

    methods = parse_methods(args.methods)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    args.api_key = api_key

    # This script always needs VLM for detection confirmation and spatial phase.
    if not args.api_key:
        print("Error: provide --api_key or set OPENAI_API_KEY")
        sys.exit(1)

    elevation_values = parse_float_csv(args.elevation_values_deg)
    distance_values = parse_float_csv(args.distance_values_m)
    if not elevation_values:
        print("Error: elevation_values_deg is empty")
        sys.exit(1)
    if not distance_values:
        print("Error: distance_values_m is empty")
        sys.exit(1)

    eval_dir = os.path.join(args.experiment_dir, "eval")
    frames = load_frames_index(eval_dir)
    camera_info = load_camera_info(eval_dir)

    frame = get_frame(frames, args.image_idx)
    if frame is None:
        print(f"Error: frame {args.image_idx} not found in frames_index.csv")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(args.experiment_dir, str(args.image_idx), "heuristic_pose_eval")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading 3DGS checkpoint...")
    params = load_checkpoint(args.experiment_dir)

    base_c2w = frame["camera_pose"]
    tx, ty, tz = c2w_to_translation(base_c2w)
    rr, rp, ry = c2w_to_rpy_deg(base_c2w)

    print(f"{'='*70}")
    print("  Heuristic Pose Baseline")
    print(f"{'='*70}")
    print(f"  Experiment:    {args.experiment_dir}")
    print(f"  Frame:         {frame['index']}")
    print(f"  Methods:       {', '.join(methods)}")
    print(f"  Num runs:      {args.num_runs}")
    print(f"  Initial pose:  t=[{tx:.3f}, {ty:.3f}, {tz:.3f}]  rpy=[{rr:.1f}, {rp:.1f}, {ry:.1f}] deg")
    print(f"  Sampling:      az_step={args.azimuth_step_deg} deg  elev={elevation_values}  dist={distance_values}")
    print(f"  Max cand:      {args.max_candidates if args.max_candidates > 0 else 'none'}")
    print(f"  Skip spatial:  {args.skip_spatial}")
    print(f"{'='*70}")

    clip_scorer = None
    if "clip" in methods:
        print("Initializing CLIP scorer...")
        clip_scorer = ClipScorer(args.clip_model, args.clip_pretrained)

    all_runs = []
    for run_id in range(1, args.num_runs + 1):
        print(f"\n{'#'*70}")
        print(f"  RUN {run_id}/{args.num_runs}")
        print(f"{'#'*70}")

        run_dir = os.path.join(output_dir, f"run_{run_id:02d}")
        os.makedirs(run_dir, exist_ok=True)

        candidates = generate_candidates(
            base_c2w=base_c2w,
            azimuth_step_deg=args.azimuth_step_deg,
            elevation_values_deg=elevation_values,
            distance_values_m=distance_values,
            max_candidates=args.max_candidates,
        )
        print(f"Generated {len(candidates)} candidate poses")

        rendered_candidates = render_candidate_views(
            params=params,
            camera_info=camera_info,
            candidates=candidates,
            out_dir=os.path.join(run_dir, "candidates"),
            use_depth=args.use_depth,
        )

        rgb_path = frame["gs_rgb_path"] if args.use_rendered else frame["gt_rgb_path"]
        initial_depth_path = None
        if args.use_depth:
            initial_depth_path = frame["gs_depth_path"] if args.use_rendered else frame["gt_depth_path"]
            if not os.path.isfile(initial_depth_path):
                initial_depth_path = None

        run_result = {
            "run_id": run_id,
            "frame_index": frame["index"],
            "num_candidates": len(candidates),
        }

        if "control" in methods:
            print("Running control method (iterative VLM search)...")
            control_t0 = time.time()
            control_render_dir = os.path.join(run_dir, "control_views")
            os.makedirs(control_render_dir, exist_ok=True)

            control_result = run_single_experiment(
                run_id=run_id,
                params=params,
                frame=frame,
                camera_info=camera_info,
                rgb_path=rgb_path,
                initial_depth_path=initial_depth_path,
                c2w=base_c2w,
                render_dir=control_render_dir,
                output_dir=run_dir,
                api_key=args.api_key,
                model=args.model,
                use_depth=args.use_depth,
                use_rendered=args.use_rendered,
                experiment_dir=args.experiment_dir,
            )
            control_cost = estimate_control_cost(control_result)
            control_cost["wall_time_sec"] = time.time() - control_t0
            control_result["method"] = "control"
            control_result["cost"] = control_cost
            run_result["control"] = control_result

        if "clip" in methods:
            print("Running heuristic method: CLIP sample-then-score...")
            run_result["clip"] = run_heuristic_method(
                method_name="clip",
                run_id=run_id,
                frame=frame,
                camera_info=camera_info,
                base_c2w=base_c2w,
                params=params,
                rendered_candidates=rendered_candidates,
                args=args,
                run_dir=run_dir,
                clip_scorer=clip_scorer,
            )

        if "vlm" in methods:
            print("Running heuristic method: VLM sample-then-score...")
            run_result["vlm"] = run_heuristic_method(
                method_name="vlm",
                run_id=run_id,
                frame=frame,
                camera_info=camera_info,
                base_c2w=base_c2w,
                params=params,
                rendered_candidates=rendered_candidates,
                args=args,
                run_dir=run_dir,
                clip_scorer=None,
            )

        run_out = os.path.join(run_dir, "results.json")
        with open(run_out, "w") as f:
            json.dump(run_result, f, indent=2)
        print(f"Run {run_id} saved to: {run_out}")

        all_runs.append(run_result)

    aggregate = {
        "experiment_dir": args.experiment_dir,
        "frame_index": frame["index"],
        "objects": [OBJECT_1, OBJECT_2],
        "methods": methods,
        "num_runs": args.num_runs,
        "sampling": {
            "azimuth_step_deg": args.azimuth_step_deg,
            "elevation_values_deg": elevation_values,
            "distance_values_m": distance_values,
            "max_candidates": args.max_candidates,
        },
        "clip": {
            "query": args.clip_query,
            "model": args.clip_model,
            "pretrained": args.clip_pretrained,
        },
        "results": all_runs,
    }
    aggregate["summary"] = aggregate_method_results(all_runs, methods)

    agg_path = os.path.join(output_dir, "aggregate.json")
    with open(agg_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\nAggregate saved to: {agg_path}")


if __name__ == "__main__":
    main()
