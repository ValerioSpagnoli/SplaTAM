"""
VLM Hallucination Evaluation on 3DGS Renders - Single Frame Version.

Evaluates a single frame through GPT-4o with 3 conditions:
  A) Baseline: rendered image, no quality info
  B) Uncertainty-annotated: rendered image + quality score warning
  C) GT ceiling: ground truth image, no quality info

Usage:
  python scripts/vlm_evaluation_single_frame.py \
    --scene room0 \
    --eval_dir experiments/Replica/room0_0/eval \
    --frame_idx 183 \
    --api_key sk-... \
    [--model gpt-4o-mini]
"""
import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)


# ─── Image helpers ────────────────────────────────────────────────────────────

def encode_image_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_metrics(eval_dir: str) -> dict:
    """Load per-frame metrics from eval directory."""
    metrics = {}
    for name in ["psnr", "rmse", "l1", "ssim", "lpips"]:
        path = os.path.join(eval_dir, f"{name}.txt")
        if os.path.exists(path):
            metrics[name] = np.loadtxt(path)
    return metrics


def get_frame_quality(eval_dir: str, frame_idx: int, eval_every: int = 25) -> dict:
    """
    Get quality metrics for a specific frame index.
    Returns dict with psnr, quality_score, quality_bucket.
    """
    metrics = load_metrics(eval_dir)
    
    if not metrics:
        print("Warning: No metrics found, using default quality score")
        return {
            "psnr": 20.0,
            "quality_score": 0.2,
            "quality_bucket": "unknown",
        }
    
    psnr = metrics.get("psnr", [])
    
    # Find which metric index corresponds to this frame
    rendered_dir = os.path.join(eval_dir, "rendered_rgb")
    frame_indices = []
    if os.path.exists(rendered_dir):
        for f in sorted(os.listdir(rendered_dir)):
            if f.startswith("gs_") and f.endswith(".png"):
                idx = int(f.replace("gs_", "").replace(".png", ""))
                frame_indices.append(idx)
    
    if not frame_indices:
        # Fallback: assume sequential eval_every spacing
        num_frames = len(psnr)
        frame_indices = list(range(0, num_frames * eval_every, eval_every))
    
    # Find the metric index for this frame
    if frame_idx not in frame_indices:
        print(f"Warning: Frame {frame_idx} not in evaluated frames")
        print(f"Available frames: {frame_indices[:10]}... (showing first 10)")
        return {
            "psnr": 20.0,
            "quality_score": 0.2,
            "quality_bucket": "unknown",
        }
    
    metric_idx = frame_indices.index(frame_idx)
    frame_psnr = float(psnr[metric_idx])
    
    # Map PSNR to quality score (same as batch script)
    quality_score = float(np.clip((frame_psnr - 15) / 25, 0, 1))
    
    # Determine quality bucket based on tercile distribution
    sorted_psnr = np.sort(psnr)
    n = len(sorted_psnr)
    low_threshold = sorted_psnr[n // 3]
    high_threshold = sorted_psnr[2 * n // 3]
    
    if frame_psnr <= low_threshold:
        quality_bucket = "low"
    elif frame_psnr >= high_threshold:
        quality_bucket = "high"
    else:
        quality_bucket = "mid"
    
    return {
        "psnr": frame_psnr,
        "quality_score": quality_score,
        "quality_bucket": quality_bucket,
    }


# ─── VLM prompts ──────────────────────────────────────────────────────────────

TASK_QUESTIONS = """Answer the following questions about this image:

1. **Room type:** What type of room is this? (e.g., bedroom, kitchen, office, bathroom, living room, hallway)
2. **Objects:** List all objects you can clearly identify in this image. Only list objects you are confident about.
3. **Confidence:** Rate your overall confidence in your answers: high, medium, or low.
4. **Navigation decision:** You are a robot exploring this environment. Based on this view, should you:
   (a) Search this area for useful objects
   (b) Move on and explore elsewhere
   (c) Request a better observation of this area (e.g., move closer or get a different angle)

Respond in JSON format:
{
  "room_type": "...",
  "objects": ["obj1", "obj2", ...],
  "confidence": "high/medium/low",
  "navigation_decision": "a/b/c",
  "reasoning": "brief explanation"
}"""


def build_prompt_condition_a(image_path: str) -> list:
    """Condition A: Baseline — rendered image, no quality info."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": TASK_QUESTIONS,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encode_image_b64(image_path)}",
                        "detail": "high",
                    },
                },
            ],
        }
    ]


def build_prompt_condition_b(image_path: str, quality_score: float) -> list:
    """Condition B: Uncertainty-annotated — rendered image + quality warning."""
    uncertainty_note = (
        f"IMPORTANT CONTEXT: This image was rendered from an online 3D Gaussian Splatting "
        f"reconstruction. The estimated reconstruction quality for this viewpoint is "
        f"{quality_score:.2f}/1.00. "
    )
    if quality_score < 0.4:
        uncertainty_note += (
            "This is a LOW quality render — the region was poorly observed during mapping. "
            "The image likely contains significant artifacts, missing geometry, or incorrect colors. "
            "Be cautious about identifying objects in this view."
        )
    elif quality_score < 0.7:
        uncertainty_note += (
            "This is a MEDIUM quality render — the region was partially observed during mapping. "
            "Some areas may contain artifacts or incomplete reconstruction."
        )
    else:
        uncertainty_note += (
            "This is a HIGH quality render — the region was well-observed during mapping. "
            "The rendering should be fairly accurate."
        )

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": uncertainty_note + "\n\n" + TASK_QUESTIONS,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encode_image_b64(image_path)}",
                        "detail": "high",
                    },
                },
            ],
        }
    ]


def build_prompt_condition_c(image_path: str) -> list:
    """Condition C: GT ceiling — ground truth image, no quality info."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": TASK_QUESTIONS,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encode_image_b64(image_path)}",
                        "detail": "high",
                    },
                },
            ],
        }
    ]


# ─── VLM API call ─────────────────────────────────────────────────────────────

def call_vlm(messages: list, api_key: str, model: str = "gpt-4o") -> dict:
    """Call OpenAI API and return parsed JSON response."""
    import httpx

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.0,  # deterministic for reproducibility
    }

    for attempt in range(3):
        try:
            response = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60.0,
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # Parse JSON from response (handle markdown code blocks)
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                content = content.rsplit("```", 1)[0]
            return json.loads(content)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: Could not parse JSON response (attempt {attempt+1}): {e}")
            if attempt == 2:
                return {"raw_response": content, "parse_error": str(e)}
        except Exception as e:
            print(f"  Warning: API error (attempt {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return {"api_error": str(e)}

    return {"error": "max retries exceeded"}


# ─── Main evaluation ──────────────────────────────────────────────────────────

def evaluate_single_frame(
    scene: str,
    eval_dir: str,
    frame_idx: int,
    api_key: str,
    model: str,
    eval_every: int,
    output_dir: str,
):
    """Evaluate a single frame with 3 conditions."""
    os.makedirs(output_dir, exist_ok=True)

    rendered_dir = os.path.join(eval_dir, "rendered_rgb")
    gt_dir = os.path.join(eval_dir, "rgb")

    # Find image files
    rendered_path = os.path.join(rendered_dir, f"gs_{frame_idx:04d}.png")
    gt_path = os.path.join(gt_dir, f"gt_{frame_idx:04d}.png")

    if not os.path.exists(rendered_path):
        print(f"Error: Rendered image not found: {rendered_path}")
        sys.exit(1)
    if not os.path.exists(gt_path):
        print(f"Error: GT image not found: {gt_path}")
        sys.exit(1)

    # Get quality metrics for this frame
    quality_info = get_frame_quality(eval_dir, frame_idx, eval_every)
    
    print("=" * 70)
    print(f"EVALUATING FRAME {frame_idx}")
    print("=" * 70)
    print(f"Scene: {scene}")
    print(f"PSNR: {quality_info['psnr']:.2f}")
    print(f"Quality score: {quality_info['quality_score']:.2f}")
    print(f"Quality bucket: {quality_info['quality_bucket']}")
    print(f"Rendered: {rendered_path}")
    print(f"GT: {gt_path}")
    print()

    result = {
        "scene": scene,
        "frame_idx": frame_idx,
        "psnr": quality_info["psnr"],
        "quality_score": quality_info["quality_score"],
        "quality_bucket": quality_info["quality_bucket"],
        "rendered_path": rendered_path,
        "gt_path": gt_path,
    }

    # Condition A: Baseline
    print("[1/3] Condition A (baseline) - rendered image, no quality info")
    messages_a = build_prompt_condition_a(rendered_path)
    result["condition_a"] = call_vlm(messages_a, api_key, model)
    print(f"  → {json.dumps(result['condition_a'], indent=2)}")
    print()

    # Condition B: Uncertainty-annotated
    print("[2/3] Condition B (uncertainty-annotated) - rendered image + quality warning")
    messages_b = build_prompt_condition_b(rendered_path, quality_info["quality_score"])
    result["condition_b"] = call_vlm(messages_b, api_key, model)
    print(f"  → {json.dumps(result['condition_b'], indent=2)}")
    print()

    # Condition C: GT ceiling
    print("[3/3] Condition C (GT ceiling) - ground truth image")
    messages_c = build_prompt_condition_c(gt_path)
    result["condition_c"] = call_vlm(messages_c, api_key, model)
    print(f"  → {json.dumps(result['condition_c'], indent=2)}")
    print()

    # Save result
    output_file = os.path.join(output_dir, f"frame_{frame_idx:04d}_result.json")
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print_summary(result)
    print()
    print(f"Results saved to: {output_file}")


def print_summary(result: dict):
    """Print comparison summary of the 3 conditions."""
    print(f"Frame {result['frame_idx']} | PSNR: {result['psnr']:.2f} | Quality: {result['quality_score']:.2f} ({result['quality_bucket']})")
    print()
    
    for condition, label in [
        ("condition_a", "A (baseline)"),
        ("condition_b", "B (uncertainty)"),
        ("condition_c", "C (GT ceiling)"),
    ]:
        resp = result.get(condition, {})
        if "error" in resp or "api_error" in resp or "parse_error" in resp:
            print(f"{label:20s} ERROR: {resp}")
            continue
        
        room = resp.get("room_type", "?")
        objs = resp.get("objects", [])
        conf = resp.get("confidence", "?")
        nav = resp.get("navigation_decision", "?")
        
        print(f"{label:20s} room={room:15s} | objects={len(objs):2d} | conf={conf:6s} | nav={nav} | {objs[:3]}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VLM hallucination evaluation on single 3DGS frame"
    )
    parser.add_argument("--scene", type=str, required=True, help="Scene name (e.g., room0)")
    parser.add_argument("--eval_dir", type=str, required=True, help="Path to eval directory with rendered/GT frames")
    parser.add_argument("--frame_idx", type=int, required=True, help="Frame index to evaluate")
    parser.add_argument("--eval_every", type=int, default=25, help="eval_every used during rendering (for metrics indexing)")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name (gpt-4o, gpt-4o-mini)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: Provide --api_key or set OPENAI_API_KEY environment variable")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(args.eval_dir, "..", "vlm_eval_single")
    output_dir = os.path.normpath(output_dir)

    evaluate_single_frame(
        scene=args.scene,
        eval_dir=args.eval_dir,
        frame_idx=args.frame_idx,
        api_key=api_key,
        model=args.model,
        eval_every=args.eval_every,
        output_dir=output_dir,
    )
