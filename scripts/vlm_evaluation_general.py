import argparse
import base64
import json
import os
import sys
import time
from glob import glob


TASK_QUESTIONS = """
Determine the spatial relationship between the computer monitor and the cardboard box.

Use the depth input as metric information (meters) and prioritize depth evidence over ambiguous RGB cues.
If the relationship cannot be determined reliably, return "uncertain" and request a new rendered view.
Assume any requested new view will be rendered using 3DGS.

Return only valid JSON (no markdown, no extra text) with exactly this schema:
{
    "spatial_relationship": "in front of" | "behind" | "to the left of" | "to the right of" | "above" | "below" | "uncertain",
    "confidence": float (0.0 to 1.0),
    "request_new_view": boolean,
    "new_view_pose": string (if request_new_view is true, format: "x,y,z,roll,pitch,yaw"; otherwise ""),
    "new_view_reasoning": string (if request_new_view is true, explain why that view disambiguates the relation; otherwise "")
}
"""


def encode_image_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def build_prompt(rgb_path: str, depth_path: str | None = None) -> list:
    prompt_text = (
        "You are given observations from an indoor robot scene.\n\n"
        "Input 1 (RGB): A perspective color image.\n"
    )
    if depth_path is not None:
        prompt_text += (
            "Input 2 (Depth): A depth visualization aligned with the RGB image. "
            "Use it only as geometric context.\n"
        )
    prompt_text += (
        "Use only evidence visible in the provided inputs.\n\n"
        "TASK:\n"
        f"{TASK_QUESTIONS}"
    )

    content = [
        {
            "type": "text",
            "text": prompt_text,
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encode_image_b64(rgb_path)}",
                "detail": "high",
            },
        },
    ]

    if depth_path is not None:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encode_image_b64(depth_path)}",
                    "detail": "high",
                },
            }
        )

    return [
        {
            "role": "user",
            "content": content,
        }
    ]

def call_vlm(messages: list, api_key: str, model: str = "gpt-4o", expect_json: bool = False) -> dict:
    import httpx

    def _strip_code_fences(text: str) -> str:
        cleaned = text.strip()
        if not cleaned.startswith("```"):
            return cleaned

        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

        # Handle accidental leading "json" token outside the code fence line.
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip("\n :")
        return cleaned

    def _parse_json_from_response(content: str) -> tuple[dict | list | None, str | None]:
        cleaned = _strip_code_fences(content)

        # Fast path: already valid JSON.
        try:
            return json.loads(cleaned), None
        except json.JSONDecodeError:
            pass

        # Fallback: find the first valid JSON object/array embedded in prose.
        decoder = json.JSONDecoder()
        for i, ch in enumerate(cleaned):
            if ch not in "[{":
                continue
            try:
                parsed, _ = decoder.raw_decode(cleaned[i:])
                return parsed, None
            except json.JSONDecodeError:
                continue

        return None, "No valid JSON object/array found in model output"

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
    if expect_json:
        payload["response_format"] = {"type": "json_object"}

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

            parsed, parse_err = _parse_json_from_response(content)
            if parse_err is None:
                return parsed if isinstance(parsed, dict) else {"parsed_response": parsed}

            # In general mode, plain text answers are valid and should not be treated as errors.
            if not expect_json:
                return {
                    "text_response": content,
                    "response_format": "plain_text",
                }

            raise json.JSONDecodeError(parse_err, content, 0)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: JSON parse failed (attempt {attempt+1}): {e}")
            if attempt == 2:
                return {
                    "raw_response": content,
                    "raw_response_lines": content.splitlines(),
                    "parse_error": str(e),
                }
        except Exception as e:
            print(f"  Warning: API error (attempt {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return {"api_error": str(e)}

    return {"error": "max retries exceeded"}

def list_rgb_images(rgb_dir: str, image_glob: str):
    pattern = os.path.join(rgb_dir, image_glob)
    paths = sorted(glob(pattern))
    return [p for p in paths if os.path.isfile(p)]

def find_matching_depth(rgb_path: str, depth_dir: str) -> str | None:
    depth_candidate = os.path.join(depth_dir, os.path.basename(rgb_path))
    if os.path.exists(depth_candidate):
        return depth_candidate
    return None

def select_images(image_paths: list[str], image_idx: str, max_images: int) -> list[str]:
    if image_idx.strip() == "-1":
        selected = image_paths
    else:
        selected = []
        tokens = [t.strip() for t in image_idx.split(",") if t.strip()]
        for tok in tokens:
            if not tok.lstrip("-").isdigit():
                print(f"Warning: ignoring invalid image_idx token '{tok}'")
                continue
            idx = int(tok)
            if 0 <= idx < len(image_paths):
                selected.append(image_paths[idx])
                continue

            found = None
            target = f"_{idx:04d}.png"
            for p in image_paths:
                if p.endswith(target):
                    found = p
                    break
            if found is not None:
                selected.append(found)
            else:
                print(f"Warning: no image matched index '{tok}'")

    if max_images > 0:
        selected = selected[:max_images]

    # Remove duplicates while preserving order.
    deduped = []
    seen = set()
    for p in selected:
        if p not in seen:
            deduped.append(p)
            seen.add(p)
    return deduped


def evaluate(
    experiment_dir: str,
    image_idx: str,
    use_rendered: bool,
    use_depth: bool,
    expect_json: bool,
    image_glob: str,
    api_key: str,
    model: str,
    output_dir: str,
    max_images: int,
):
    os.makedirs(output_dir, exist_ok=True)

    eval_dir = os.path.join(experiment_dir, "eval")
    rgb_dir = os.path.join(eval_dir, "rendered_rgb" if use_rendered else "rgb")
    depth_dir = os.path.join(eval_dir, "depth_metric")

    if not os.path.isdir(rgb_dir):
        print(f"Error: RGB directory not found: {rgb_dir}")
        sys.exit(1)

    image_paths = list_rgb_images(rgb_dir, image_glob)
    if not image_paths:
        print(f"Error: No images found in {rgb_dir} matching glob '{image_glob}'")
        sys.exit(1)

    image_paths = select_images(image_paths, image_idx=image_idx, max_images=max_images)
    if not image_paths:
        print("Error: no images selected after applying --image_idx/--max_images")
        sys.exit(1)

    print("=" * 70)
    print("VLM Evaluation")
    print("=" * 70)
    print(f"Experiment dir:     {experiment_dir}")
    print(f"RGB dir:            {rgb_dir}")
    print(f"Depth enabled:      {use_depth}")
    if use_depth:
        print(f"Depth dir:          {depth_dir}")
        if not os.path.isdir(depth_dir):
            print("Warning: depth directory not found, prompts will use only RGB")
    print(f"Image glob:         {image_glob}")
    print(f"Images to evaluate: {len(image_paths)}")
    print(f"Model:              {model}")
    print(f"Expect JSON:        {expect_json}")
    print()

    aggregate = {
        "experiment_dir": experiment_dir,
        "rgb_dir": rgb_dir,
        "depth_dir": depth_dir if use_depth else None,
        "use_rendered": use_rendered,
        "use_depth": use_depth,
        "expect_json": expect_json,
        "image_idx": image_idx,
        "image_glob": image_glob,
        "model": model,
        "num_images": len(image_paths),
        "results": [],
    }

    for idx, image_path in enumerate(image_paths):
        image_name = os.path.basename(image_path)
        print(f"[{idx+1}/{len(image_paths)}] Evaluating {image_name}")

        depth_path = None
        if use_depth and os.path.isdir(depth_dir):
            depth_path = find_matching_depth(image_path, depth_dir)
            if depth_path is None:
                print(f"  Warning: matching depth not found for {image_name}, using RGB only")

        messages = build_prompt(image_path, depth_path=depth_path)
        response = call_vlm(messages, api_key=api_key, model=model, expect_json=expect_json)

        entry = {
            "image_path": image_path,
            "depth_path": depth_path,
            "response": response,
        }
        aggregate["results"].append(entry)

        per_image_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.json")
        with open(per_image_path, "w") as f:
            json.dump(entry, f, indent=2)

        if isinstance(response, dict) and "parse_error" in response and "raw_response" in response:
            raw_text_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_raw_response.txt")
            with open(raw_text_path, "w") as f:
                f.write(response["raw_response"])

        print(f"  Saved: {per_image_path}")

    aggregate_path = os.path.join(output_dir, "vlm_evaluation_summary.json")
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
        description="Run VLM baseline prompt (Condition A) on experiment RGB/depth images"
    )
    parser.add_argument("--experiment_dir", type=str, required=True, help="Directory of the experiment")
    parser.add_argument("--image_idx", type=str, default="-1", help="Image index (by sorted order) or frame id; use '-1' for all")
    parser.add_argument("--max_images", type=int, default=-1, help="Maximum number of images to evaluate (-1 for all)")
    parser.add_argument("--use_rendered", action=argparse.BooleanOptionalAction, default=False, help="Use rendered images from eval/rendered_* instead of eval/*")
    parser.add_argument("--use_depth", action=argparse.BooleanOptionalAction, default=False, help="Include depth image in the prompt")
    parser.add_argument("--expect_json", action=argparse.BooleanOptionalAction, default=False, help="Require JSON output from model")
    parser.add_argument("--image_glob", type=str, default="*.png", help="Glob for images inside selected RGB directory")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name (gpt-4o, gpt-4o-mini)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for json results")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: Provide --api_key or set OPENAI_API_KEY environment variable")
        sys.exit(1)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(args.experiment_dir, "vlm_evaluation_results")
        output_dir = os.path.normpath(output_dir)

    evaluate(
        experiment_dir=args.experiment_dir,
        image_idx=args.image_idx,
        use_rendered=args.use_rendered,
        use_depth=args.use_depth,
        expect_json=args.expect_json,
        image_glob=args.image_glob,
        max_images=args.max_images,
        model=args.model,
        output_dir=output_dir,
        api_key=api_key,
    )
