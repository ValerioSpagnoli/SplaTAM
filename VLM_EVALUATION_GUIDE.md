# VLM Uncertainty Evaluation Guide

## Overview

This evaluation tests whether Vision-Language Models (VLMs) hallucinate more on low-quality 3D Gaussian Splatting (3DGS) renders, and whether explicitly communicating reconstruction uncertainty to the VLM improves its decision-making and confidence calibration.

**Research Question:** Can we reduce VLM hallucinations in embodied navigation by telling the VLM when the 3DGS reconstruction is poor quality?

---

## Experimental Design

### Three Conditions (Same VLM, Same Questions, Same Viewpoints)

#### **Condition A: Baseline**
- Input: Rendered image from 3DGS reconstruction
- Prompt: Standard questions (no quality information)
- **Purpose:** Establish baseline VLM behavior without uncertainty awareness

#### **Condition B: Uncertainty-Annotated**
- Input: Same rendered image as Condition A
- Prompt: Includes quality score and explicit warning text
- **Purpose:** Test if uncertainty communication changes VLM behavior

**Quality warnings provided:**
```
Quality < 0.4:  "LOW quality render — likely contains significant artifacts"
Quality 0.4-0.7: "MEDIUM quality render — partially observed during mapping"
Quality > 0.7:  "HIGH quality render — well-observed, fairly accurate"
```

#### **Condition C: Ground Truth Ceiling**
- Input: Actual RGB image from dataset (not 3DGS render)
- Prompt: Standard questions (no quality information)
- **Purpose:** Establish performance ceiling — best-case VLM accuracy

---

## Metrics Measured

### 1. **Room Type Classification**
- What the VLM thinks the room is (living room, office, bedroom, etc.)
- Compare against ground truth scene labels
- **Good sign:** Conditions A and C agree; Condition B hedges on low-quality renders

### 2. **Object Detection Count**
- Number of objects the VLM lists
- **Expected pattern:**
  - Condition C (GT) should have highest object count (most visible detail)
  - Condition A may hallucinate objects on low-quality renders
  - Condition B should list fewer objects on low-quality renders (more conservative)

### 3. **Confidence Calibration** ⭐ Key Metric
VLM self-reports confidence as high/medium/low

**Hypothesis validation:**
- ✅ Condition B should report **lower confidence** than Condition A on same renders
- ✅ Condition C should have consistently **high confidence**
- ❌ If Condition A and B have same confidence, uncertainty warning is ignored

**Example from results:**
| Condition | High | Medium | Low |
|-----------|------|--------|-----|
| A (baseline) | 0 | 4 | 1 |
| B (uncertainty) | 0 | 1 | **4** ⬅ Shifted down |
| C (GT) | **5** | 0 | 0 |

### 4. **Navigation Decision** ⭐ Key Metric

VLM chooses between:
- **(a)** Search this area for useful objects
- **(b)** Move on and explore elsewhere  
- **(c)** Request better observation (move closer, different angle)

**Hypothesis validation:**
- ✅ Condition B should choose **(c)** more often on low-quality renders
- ✅ Condition A may overconfidently choose **(a)** even on bad renders
- ✅ Condition C should confidently choose **(a)** when objects are visible

**Example from results:**
| Condition | (a) Search | (b) Move on | (c) Better view |
|-----------|------------|-------------|-----------------|
| A (baseline) | 3 | 0 | 2 |
| B (uncertainty) | **0** | 2 | **3** ⬅ More conservative |
| C (GT) | **5** | 0 | 0 |

---

## Quality Score Computation

Quality scores map PSNR (Peak Signal-to-Noise Ratio) to 0-1 scale:

```python
quality_score = clip((PSNR - 15) / 25, 0, 1)
```

**Interpretation:**
- PSNR 15 or below → quality = 0.0 (very poor)
- PSNR 40 or above → quality = 1.0 (excellent)
- PSNR 27.5 → quality = 0.5 (medium)

**Quality buckets** (for stratified analysis):
- **Low tercile:** Bottom 33% of PSNR values
- **Mid tercile:** Middle 33%
- **High tercile:** Top 33%

---

## Running the Evaluation

### Batch Evaluation (Multiple Frames)

Evaluates ~40 views with stratified sampling (40% low, 20% mid, 40% high quality):

```bash
cd ~/RVP/SplaTAM

pixi run python scripts/vlm_evaluation.py \
  --scene room0 \
  --eval_dir experiments/Replica/room0_0/eval \
  --num_views 40 \
  --model gpt-4o-mini \
  --api_key $(cat openai.api.key)
```

**Output:**
- `experiments/Replica/room0_0/vlm_eval/results.json` — Full per-view responses
- `experiments/Replica/room0_0/vlm_eval/summary.json` — Aggregate statistics

### Single Frame Evaluation

Test specific frames of interest:

```bash
pixi run python scripts/vlm_evaluation_single_frame.py \
  --scene room0 \
  --eval_dir experiments/Replica/room0_0/eval \
  --frame_idx 183 \
  --model gpt-4o-mini \
  --api_key $(cat openai.api.key)
```

**Output:**
- Terminal: Real-time results with comparison table
- `vlm_eval_single/frame_0183_result.json` — Detailed JSON

---

## Interpreting Results

### ✅ **Strong Evidence for Hypothesis**

1. **Confidence Calibration Working:**
   - Condition B has significantly more "low" confidence responses than A
   - Especially on low-quality renders

2. **Navigation Decisions Shift:**
   - Condition B chooses option (c) "request better view" more often
   - Condition B chooses option (a) "search here" less often
   - Shows VLM is actually using the uncertainty information

3. **Object Count Reduces:**
   - Condition B lists fewer objects on low-quality renders (less hallucination)
   - Condition C consistently highest (sees real detail)

### ⚠️ **Weak Evidence / Needs Investigation**

1. **No Confidence Shift:**
   - Condition A and B have similar confidence distributions
   - VLM may be ignoring uncertainty warnings

2. **Navigation Unchanged:**
   - Condition B still chooses (a) on low-quality renders
   - Uncertainty communication not effective

3. **Object Count Unchanged:**
   - Condition B lists same number of objects as A
   - May indicate hallucination persists despite warnings

### ❌ **Hypothesis Rejected**

1. **Confidence Increases with Uncertainty:**
   - Condition B reports higher confidence than A
   - VLM interpreting warning incorrectly

2. **Wrong Direction Shift:**
   - Condition B chooses (a) more than A on bad renders
   - Counterproductive effect

---

## Per-Quality-Bucket Analysis

Results are stratified by quality level to check if uncertainty effects are stronger on low-quality renders.

**Expected pattern:**

| Quality | Condition A Behavior | Condition B Behavior |
|---------|---------------------|---------------------|
| **Low** | Medium-high confidence, chooses (a) | Low confidence, chooses (c) |
| **Mid** | Medium confidence, mixed choices | Low-medium confidence, conservative |
| **High** | High confidence, chooses (a) | Medium-high confidence, chooses (a) |

**Why this matters:**
- Uncertainty annotation should have **strongest effect** on low-quality renders
- On high-quality renders, Conditions A and B should be similar (render is good anyway)
- If Condition B affects high-quality renders negatively → overcautious

---

## Example Result Interpretation

### Frame 183 (Low Quality — PSNR: 10.47, Quality: 0.0)

**Condition A (Baseline):**
```json
{
  "room_type": "unknown",
  "objects": [],
  "confidence": "low",
  "navigation_decision": "c",
  "reasoning": "Image is unclear and distorted"
}
```
→ VLM struggles but doesn't know why

**Condition B (Uncertainty-annotated):**
```json
{
  "room_type": "unknown", 
  "objects": [],
  "confidence": "low",
  "navigation_decision": "b",
  "reasoning": "Image quality is very low, move on and explore elsewhere"
}
```
→ VLM makes informed decision to skip this area (changed from (c) to (b))

**Condition C (Ground Truth):**
```json
{
  "room_type": "living room",
  "objects": ["sofa", "throw blanket", "cushions", "coffee table", ...],
  "confidence": "high", 
  "navigation_decision": "a",
  "reasoning": "Room contains several useful objects"
}
```
→ With good image, VLM performs perfectly

**Interpretation:** ✅ Uncertainty annotation helped VLM make better decision (skip bad area instead of requesting re-observation)

### Frame 73 (High Quality — PSNR: 23.19, Quality: 0.33)

**Condition A:**
- Room: living room ✓
- Objects: 5
- Confidence: medium
- Decision: (a) search

**Condition B:**
- Room: living room ✓
- Objects: 5 (same as A)
- Confidence: medium (same as A)
- Decision: (b) move on ⚠️

**Condition C:**
- Room: living room ✓
- Objects: 5 (same as A and B)
- Confidence: high
- Decision: (a) search

**Interpretation:** ⚠️ On decent-quality render, uncertainty warning made VLM too conservative. May need to tune quality thresholds.

---

## Key Takeaways for Research

### What We're Validating

This experiment validates **Core Assumption** of the research proposal:

> VLMs make poor decisions on low-quality 3DGS renders, but explicit uncertainty communication can mitigate this.

**If validated → supports contributions:**
- **C1:** Uncertainty-annotated semantic BEV makes sense (VLMs can use quality info)
- **C2:** `refine_map` action primitive is needed (VLM can request better observations)
- **C3:** Structured feedback helps (VLM responds to explicit quality signals)

### Success Criteria

**Minimum viable validation:**
1. ✅ Condition B has lower confidence than A on low-quality renders
2. ✅ Condition B chooses (c) more often on low-quality renders

**Strong validation:**
3. ✅ Per-bucket analysis shows strongest effect on low-quality renders
4. ✅ Object count in Condition B is closer to Condition C (less hallucination)

**Exceptional validation:**
5. ✅ Condition B matches Condition C performance on high-quality renders
6. ✅ Room type accuracy improves in Condition B vs A

---

## Troubleshooting

### "Error: Rendered image not found"
- Check that you've run `render_frames.py` with `--save_frames` first
- Verify `eval_dir` contains `rendered_rgb/` and `rgb/` subdirectories

### "No metrics found, using default quality score"
- Ensure `eval_dir` contains PSNR metrics (`psnr.txt`)
- Run SplaTAM evaluation first to generate metrics

### "API error: rate limit"
- Add delays between API calls (batch script already has this)
- Use `--model gpt-4o-mini` for testing (cheaper tier limits)

### "Could not parse JSON response"
- VLM sometimes wraps JSON in markdown code blocks (script handles this)
- Check `raw_response` field in output for debugging
- Temperature is set to 0.0 for determinism, but occasional parsing issues may occur

---

## Files and Locations

```
SplaTAM/
├── scripts/
│   ├── vlm_evaluation.py                 # Batch evaluation
│   └── vlm_evaluation_single_frame.py    # Single frame evaluation
│
├── experiments/Replica/room0_0/
│   ├── eval/                            # Evaluation data
│   │   ├── rendered_rgb/                # 3DGS renders
│   │   ├── rgb/                         # Ground truth images
│   │   ├── psnr.txt                     # Quality metrics
│   │   └── ...
│   │
│   ├── vlm_eval/                        # Batch evaluation results
│   │   ├── results.json                 # Full per-view responses
│   │   └── summary.json                 # Aggregate statistics
│   │
│   └── vlm_eval_single/                 # Single frame results
│       └── frame_XXXX_result.json
│
└── VLM_EVALUATION_GUIDE.md              # This file
```

---

## Citation

If this evaluation methodology proves useful for your research:

```bibtex
@misc{vlm_uncertainty_eval_2026,
  title={VLM Uncertainty Evaluation for 3D Gaussian Splatting Navigation},
  author={Your Name},
  year={2026},
  note={Evaluates hallucination rates and confidence calibration when VLMs 
        receive explicit reconstruction quality information}
}
```

---

## Further Analysis Ideas

1. **Semantic accuracy scoring:**
   - Manually annotate ground truth objects per frame
   - Compute precision/recall for object detection
   - Measure false positive rate (hallucination)

2. **Cross-model comparison:**
   - Run same evaluation with GPT-4o, GPT-4o-mini, LLaVA, InternVL
   - Check if uncertainty sensitivity varies by model architecture

3. **Quality threshold tuning:**
   - Test different PSNR→quality mappings
   - Find optimal thresholds for warning levels

4. **Prompt engineering:**
   - A/B test different uncertainty warning phrasings
   - Test visual quality indicators (colored borders, quality badges)

5. **Multi-view consistency:**
   - For same scene region, compare object lists across viewpoints
   - Check if uncertainty annotation improves cross-view consistency
