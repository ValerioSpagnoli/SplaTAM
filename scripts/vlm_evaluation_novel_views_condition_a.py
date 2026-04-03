"""
VLM evaluation on novel-view renders using only base prompt (Condition A).

This script evaluates one or more rendered novel-view RGB images with the same
baseline prompt used in the other VLM evaluation scripts.

Usage:
  python scripts/vlm_evaluation_novel_views_condition_a.py \
    --novel_views_dir experiments/IsaacSim/office0_0/novel_view \
    --scene office0 \
    --api_key sk-... \
    [--model gpt-4o] \
    [--image_glob "*rgb*.png"]
"""

import argparse
import base64
import json
import os
import sys
import time
from glob import glob


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


def encode_image_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_prompt_condition_a(image_path: str) -> list:
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


def call_vlm(messages: list, api_key: str, model: str = "gpt-4o") -> dict:
    import httpx

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.0,
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
            content = result["choices"][0]["message"]["content"].strip()

            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                content = content.rsplit("```", 1)[0]

            return json.loads(content)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: JSON parse failed (attempt {attempt+1}): {e}")
            if attempt == 2:
                return {"raw_response": content, "parse_error": str(e)}
        except Exception as e:
            print(f"  Warning: API error (attempt {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return {"api_error": str(e)}

    return {"error": "max retries exceeded"}


def list_novel_images(novel_views_dir: str, image_glob: str):
    pattern = os.path.join(novel_views_dir, image_glob)
    paths = sorted(glob(pattern))
    return [p for p in paths if os.path.isfile(p)]


def evaluate_novel_views(
    scene: str,
    novel_views_dir: str,
    image_glob: str,
    api_key: str,
    model: str,
    output_dir: str,
    max_images: int,
):
    os.makedirs(output_dir, exist_ok=True)

    image_paths = list_novel_images(novel_views_dir, image_glob)
    if not image_paths:
        print(f"Error: No images found in {novel_views_dir} matching glob '{image_glob}'")
        sys.exit(1)

    if max_images > 0:
        image_paths = image_paths[:max_images]

    print("=" * 70)
    print("VLM EVALUATION ON NOVEL VIEWS (CONDITION A ONLY)")
    print("=" * 70)
    print(f"Scene: {scene}")
    print(f"Novel views dir: {novel_views_dir}")
    print(f"Image glob: {image_glob}")
    print(f"Images to evaluate: {len(image_paths)}")
    print(f"Model: {model}")
    print()

    aggregate = {
        "scene": scene,
        "novel_views_dir": novel_views_dir,
        "image_glob": image_glob,
        "model": model,
        "num_images": len(image_paths),
        "results": [],
    }

    for idx, image_path in enumerate(image_paths):
        image_name = os.path.basename(image_path)
        print(f"[{idx+1}/{len(image_paths)}] Evaluating {image_name}")

        messages = build_prompt_condition_a(image_path)
        response = call_vlm(messages, api_key=api_key, model=model)

        entry = {
            "image_path": image_path,
            "condition": "A",
            "response": response,
        }
        aggregate["results"].append(entry)

        per_image_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_condition_a.json")
        with open(per_image_path, "w") as f:
            json.dump(entry, f, indent=2)

        print(f"  Saved: {per_image_path}")

    aggregate_path = os.path.join(output_dir, "novel_views_condition_a_summary.json")
    with open(aggregate_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Per-image results in: {output_dir}")
    print(f"Aggregate summary: {aggregate_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run VLM baseline prompt (Condition A) on novel-view RGB renders"
    )
    parser.add_argument("--scene", type=str, required=True, help="Scene name")
    parser.add_argument("--novel_views_dir", type=str, required=True, help="Directory containing novel-view images")
    parser.add_argument("--image_glob", type=str, default="*rgb*.png", help="Glob for selecting RGB images in novel_views_dir")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name (gpt-4o, gpt-4o-mini)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for json results")
    parser.add_argument("--max_images", type=int, default=0, help="Optional cap on number of images (0 = all)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: Provide --api_key or set OPENAI_API_KEY environment variable")
        sys.exit(1)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(args.novel_views_dir, "..", "vlm_eval_novel_condition_a")
        output_dir = os.path.normpath(output_dir)

    evaluate_novel_views(
        scene=args.scene,
        novel_views_dir=args.novel_views_dir,
        image_glob=args.image_glob,
        api_key=api_key,
        model=args.model,
        output_dir=output_dir,
        max_images=args.max_images,
    )
