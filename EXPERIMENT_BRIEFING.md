# VLM Uncertainty Experiment — Context Briefing for Claude Code

## Goal

Run a preliminary experiment to test whether VLMs hallucinate more on low-quality 3DGS renders, and whether explicitly telling the VLM about reconstruction uncertainty changes its behavior.

## Hypothesis

VLMs make worse semantic decisions (object identification, room classification) when given under-reconstructed 3DGS renders. Explicitly communicating quality information (textual uncertainty cue) reduces hallucination rate and improves VLM confidence calibration.

## Experimental Protocol

### 3 Conditions (same VLM, same questions, same views):

- **Condition A (Baseline):** Rendered view from 3DGS → VLM prompt asks "What objects/room type?" No quality info.
- **Condition B (Uncertainty-annotated):** Same rendered view → prompt includes "This region has low reconstruction confidence (quality: 0.3/1.0). The render may contain artifacts."
- **Condition C (Ground truth ceiling):** Actual GT RGB image from dataset for the same viewpoint → same questions.

### What to Measure:
1. **Hallucination rate:** False objects, wrong room types vs GT annotations.
2. **VLM confidence calibration:** Ask VLM to output confidence (high/medium/low). Check if Condition B makes it downgrade confidence on bad renders.
3. **Decision change rate:** Frame as navigation decision — "should you (a) search here, (b) explore elsewhere, (c) request better observation?" Check if B shifts answers toward (c).

### View Selection Strategy:
- ~30-50 view pairs across 2 scenes
- For each viewpoint: render from 3DGS (quality varies) + GT RGB from dataset
- Need views from well-reconstructed regions AND under-reconstructed regions (single-viewpoint coverage, map periphery, fast traversal areas)

## Environment

- **Machine:** Ubuntu, RTX 5050 8GB VRAM, CUDA 13.0
- **PyTorch:** 2.9.0+cu130
- **Python:** 3.10.12
- **SplaTAM repo:** ~/RVP/SplaTAM/
- **Replica dataset:** ~/RVP/SplaTAM/data/Replica/ (downloading room0 and office0)
  - Format: Nice-SLAM Replica (RGB frames as .jpg, depth as .png, traj.txt with GT poses)
  - Camera: 1200x680, fx=fy=600.0, cx=599.5, cy=339.5
- **Package manager:** pixi (v0.63.2), no conda
- **No SplaTAM dependencies installed yet** — needs pixi environment setup

## SplaTAM Config

Config at `~/RVP/SplaTAM/configs/replica/replica_eval.py`:
- Scenes: room0-room2, office0-office4
- Data path: `./data/Replica`
- Dataset config: `./configs/data/replica.yaml`
- Key params: map_every=1, keyframe_every=5, mapping_window_size=24, tracking_iters=40, mapping_iters=60
- Gaussian distribution: isotropic
- Image size: 680x1200

## SplaTAM Dependencies (from requirements.txt)

```
tqdm==4.65.0
Pillow
opencv-python
imageio
matplotlib
kornia
natsort
pyyaml
wandb
lpips
open3d==0.16.0
torchmetrics
cyclonedds
pytorch-msssim
plyfile==0.8.1
git+https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth.git@cb65e4b
```

Note: open3d==0.16.0 may conflict with CUDA 13.0/PyTorch 2.9. The custom gaussian rasterizer needs to be built from source. User wants pixi for environment management.

## Steps to Complete

1. **Set up pixi environment** for SplaTAM with all dependencies (handle CUDA 13.0 compatibility)
2. **Run SplaTAM** on Replica room0 and office0 → get 3DGS reconstructions + checkpoints
3. **Identify good vs bad regions** using per-Gaussian opacity/covariance metrics after reconstruction
4. **Render paired views:** high-quality renders, low-quality renders, extract corresponding GT RGB frames
5. **Build VLM evaluation prompts** for 3 conditions × N views
6. **Run VLM** (GPT-4o API or local InternVL/LLaVA) and collect responses
7. **Score** hallucinations against GT annotations

## Broader Project Context

This experiment validates the core assumption of a research proposal on uncertainty-aware VLM navigation with online 3DGS map refinement. The proposal's key contributions are:

- C1: Uncertainty-annotated semantic BEV with height-aware quality encoding (visual BEV + text height profiles for low-quality cells)
- C2: `refine_map` action primitive — exploration-time (not detection-time) map refinement triggered by reconstruction uncertainty
- C3: Structured planner-result feedback to VLM

The proposal builds on 3DGSNav (arXiv 2602.12159) and GSMem (arXiv 2603.19137). Both already use opacity-based quality metrics and active re-observation, but neither communicates reconstruction uncertainty to the VLM as a global spatial signal, nor offers proactive exploration-time refinement.

## Files

- Revised proposal: available as proposal_v2.md / proposal_v2.pdf
- 3DGSNav paper: ~/RVP/SplaTAM/ (or wherever user stores papers) — arXiv 2602.12159
- GSMem paper: arXiv 2603.19137
