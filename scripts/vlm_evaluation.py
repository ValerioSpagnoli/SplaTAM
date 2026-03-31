"""
VLM Hallucination Evaluation on 3DGS Renders.

Runs 3 conditions through GPT-4o:
  A) Baseline: rendered image, no quality info
  B) Uncertainty-annotated: rendered image + quality score warning
  C) GT ceiling: ground truth image, no quality info

Usage:
  python scripts/vlm_evaluation.py \
    --scene room0 \
    --eval_dir experiments/Replica/room0_0/eval \
    --num_views 40 \
    --api_key sk-... \
    [--model gpt-4o-mini]  # use mini for testing, gpt-4o for real experiment
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


# ─── View selection ───────────────────────────────────────────────────────────

def select_views(eval_dir: str, num_views: int, eval_every: int) -> list[dict]:
    """
    Select a mix of good and bad quality views based on PSNR.
    Returns list of dicts with frame_idx, psnr, quality_bucket.
    """
    metrics = load_metrics(eval_dir)
    psnr = metrics["psnr"]
    num_evaluated = len(psnr)

    # Map metric index back to frame index
    # eval saves one value per evaluated frame (every eval_every-th frame)
    frame_indices = []
    rendered_dir = os.path.join(eval_dir, "rendered_rgb")
    if os.path.exists(rendered_dir):
        for f in sorted(os.listdir(rendered_dir)):
            if f.startswith("gs_") and f.endswith(".png"):
                idx = int(f.replace("gs_", "").replace(".png", ""))
                frame_indices.append(idx)

    if not frame_indices:
        # Fallback: reconstruct from eval_every
        frame_indices = list(range(0, num_evaluated * eval_every, eval_every))
        if len(frame_indices) > num_evaluated:
            frame_indices = frame_indices[:num_evaluated]

    assert len(frame_indices) == len(psnr), \
        f"Mismatch: {len(frame_indices)} frames vs {len(psnr)} PSNR values"

    # Sort by PSNR and split into terciles
    sorted_idx = np.argsort(psnr)
    n = len(sorted_idx)
    low_tercile = sorted_idx[:n // 3]
    mid_tercile = sorted_idx[n // 3: 2 * n // 3]
    high_tercile = sorted_idx[2 * n // 3:]

    # Sample: ~40% low, ~20% mid, ~40% high
    n_low = max(1, int(num_views * 0.4))
    n_mid = max(1, int(num_views * 0.2))
    n_high = num_views - n_low - n_mid

    rng = np.random.default_rng(42)
    selected = []

    for bucket, indices, count in [
        ("low", low_tercile, n_low),
        ("mid", mid_tercile, n_mid),
        ("high", high_tercile, n_high),
    ]:
        chosen = rng.choice(indices, size=min(count, len(indices)), replace=False)
        for i in chosen:
            selected.append({
                "metric_idx": int(i),
                "frame_idx": frame_indices[int(i)],
                "psnr": float(psnr[i]),
                "quality_score": float(np.clip((psnr[i] - 15) / 25, 0, 1)),  # map PSNR to 0-1
                "quality_bucket": bucket,
            })

    selected.sort(key=lambda x: x["frame_idx"])
    print(f"Selected {len(selected)} views: "
          f"{sum(1 for v in selected if v['quality_bucket']=='low')} low, "
          f"{sum(1 for v in selected if v['quality_bucket']=='mid')} mid, "
          f"{sum(1 for v in selected if v['quality_bucket']=='high')} high")
    return selected


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
        "max_tokens": 500,
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


# ─── Main evaluation loop ────────────────────────────────────────────────────

def run_evaluation(
    eval_dir: str,
    scene: str,
    num_views: int,
    api_key: str,
    model: str,
    eval_every: int,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)

    rendered_dir = os.path.join(eval_dir, "rendered_rgb")
    gt_dir = os.path.join(eval_dir, "rgb")

    if not os.path.exists(rendered_dir) or not os.path.exists(gt_dir):
        print(f"Error: Need rendered and GT frames. Run render_frames.py with --save_frames first.")
        print(f"  Missing: {rendered_dir}" if not os.path.exists(rendered_dir) else "")
        print(f"  Missing: {gt_dir}" if not os.path.exists(gt_dir) else "")
        sys.exit(1)

    # Select views
    views = select_views(eval_dir, num_views, eval_every)

    results = []
    total = len(views) * 3
    done = 0

    for view in views:
        frame_idx = view["frame_idx"]
        rendered_path = os.path.join(rendered_dir, f"gs_{frame_idx:04d}.png")
        gt_path = os.path.join(gt_dir, f"gt_{frame_idx:04d}.png")

        if not os.path.exists(rendered_path):
            print(f"  Skipping frame {frame_idx}: rendered image not found")
            continue
        if not os.path.exists(gt_path):
            print(f"  Skipping frame {frame_idx}: GT image not found")
            continue

        entry = {
            "scene": scene,
            "frame_idx": frame_idx,
            "psnr": view["psnr"],
            "quality_score": view["quality_score"],
            "quality_bucket": view["quality_bucket"],
        }

        # Condition A: Baseline
        done += 1
        print(f"[{done}/{total}] Frame {frame_idx} | Condition A (baseline) | PSNR={view['psnr']:.1f}")
        messages_a = build_prompt_condition_a(rendered_path)
        entry["condition_a"] = call_vlm(messages_a, api_key, model)

        # Condition B: Uncertainty-annotated
        done += 1
        print(f"[{done}/{total}] Frame {frame_idx} | Condition B (uncertainty) | quality={view['quality_score']:.2f}")
        messages_b = build_prompt_condition_b(rendered_path, view["quality_score"])
        entry["condition_b"] = call_vlm(messages_b, api_key, model)

        # Condition C: GT ceiling
        done += 1
        print(f"[{done}/{total}] Frame {frame_idx} | Condition C (GT)")
        messages_c = build_prompt_condition_c(gt_path)
        entry["condition_c"] = call_vlm(messages_c, api_key, model)

        results.append(entry)

        # Save intermediate results
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    # ─── Aggregate statistics ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    summary = compute_summary(results)

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump({"views": results, "summary": summary}, f, indent=2)

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}/")
    print(f"  results.json  — full per-view responses")
    print(f"  summary.json  — aggregate statistics")


def compute_summary(results: list) -> dict:
    """Compute aggregate statistics from VLM responses."""
    summary = {}

    for condition in ["condition_a", "condition_b", "condition_c"]:
        cond_label = {"condition_a": "A (baseline)", "condition_b": "B (uncertainty)", "condition_c": "C (GT)"}[condition]

        confidence_counts = {"high": 0, "medium": 0, "low": 0}
        nav_counts = {"a": 0, "b": 0, "c": 0}
        obj_counts = []
        valid = 0

        # Per quality bucket
        bucket_confidence = {"low": [], "mid": [], "high": []}
        bucket_nav = {"low": [], "mid": [], "high": []}
        bucket_obj_counts = {"low": [], "mid": [], "high": []}

        for r in results:
            resp = r.get(condition, {})
            bucket = r["quality_bucket"]

            if "error" in resp or "api_error" in resp or "parse_error" in resp:
                continue
            valid += 1

            conf = resp.get("confidence", "").lower()
            if conf in confidence_counts:
                confidence_counts[conf] += 1
                bucket_confidence[bucket].append(conf)

            nav = resp.get("navigation_decision", "").lower().strip()
            if nav in nav_counts:
                nav_counts[nav] += 1
                bucket_nav[bucket].append(nav)

            objs = resp.get("objects", [])
            obj_counts.append(len(objs))
            bucket_obj_counts[bucket].append(len(objs))

        cond_summary = {
            "valid_responses": valid,
            "confidence_distribution": confidence_counts,
            "navigation_decisions": nav_counts,
            "avg_objects_listed": float(np.mean(obj_counts)) if obj_counts else 0,
        }

        # Per-bucket breakdown
        for bucket in ["low", "mid", "high"]:
            bc = bucket_confidence[bucket]
            bn = bucket_nav[bucket]
            bo = bucket_obj_counts[bucket]
            cond_summary[f"bucket_{bucket}"] = {
                "n": len(bc),
                "confidence": {k: bc.count(k) for k in ["high", "medium", "low"]} if bc else {},
                "navigation": {k: bn.count(k) for k in ["a", "b", "c"]} if bn else {},
                "avg_objects": float(np.mean(bo)) if bo else 0,
            }

        summary[cond_label] = cond_summary

        # Print
        print(f"\n{cond_label} ({valid} valid responses):")
        print(f"  Confidence: {confidence_counts}")
        print(f"  Navigation: a={nav_counts['a']} b={nav_counts['b']} c={nav_counts['c']}")
        print(f"  Avg objects: {cond_summary['avg_objects_listed']:.1f}")
        for bucket in ["low", "mid", "high"]:
            bs = cond_summary[f"bucket_{bucket}"]
            if bs["n"] > 0:
                print(f"    [{bucket} quality] n={bs['n']}, conf={bs['confidence']}, nav={bs['navigation']}, objs={bs['avg_objects']:.1f}")

    return summary


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM hallucination evaluation on 3DGS renders")
    parser.add_argument("--scene", type=str, required=True, help="Scene name (e.g., room0)")
    parser.add_argument("--eval_dir", type=str, required=True, help="Path to eval directory with rendered/GT frames")
    parser.add_argument("--num_views", type=int, default=40, help="Number of views to evaluate")
    parser.add_argument("--eval_every", type=int, default=25, help="eval_every used during rendering (for frame index mapping)")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name (gpt-4o, gpt-4o-mini)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: Provide --api_key or set OPENAI_API_KEY environment variable")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(args.eval_dir, "..", "vlm_eval")
    output_dir = os.path.normpath(output_dir)

    run_evaluation(
        eval_dir=args.eval_dir,
        scene=args.scene,
        num_views=args.num_views,
        api_key=api_key,
        model=args.model,
        eval_every=args.eval_every,
        output_dir=output_dir,
    )
