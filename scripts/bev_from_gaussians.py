"""
Bird's Eye View (BEV) rendering from a 3D Gaussian Splatting map.

Generates two BEV images from a SplaTAM checkpoint:
  1. Occupancy Grid BEV: shows free space vs occupied space from a top-down view
  2. Quality BEV: encodes reconstruction quality (opacity accumulation) of the gaussians

Inspired by 3DGSNav (Zheng et al., 2026), which uses opacity-based top-down
rendering to construct exploration maps and BEV representations for VLM reasoning.

Usage:
  python scripts/bev_from_gaussians.py \
    --params_path experiments/Replica/room0_0/params.npz \
    --output_dir experiments/Replica/room0_0/bev \
    [--resolution 512] \
    [--floor_height_percentile 10] \
    [--ceil_height_percentile 90] \
    [--occupancy_threshold 0.5] \
    [--margin 0.5]
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera


def load_params(params_path: str) -> dict:
    """Load gaussian parameters from npz and move to GPU."""
    raw = dict(np.load(params_path, allow_pickle=True))
    params = {k: torch.tensor(v).cuda().float() for k, v in raw.items()}
    return params


def compute_scene_bounds(means3D: torch.Tensor, margin: float = 0.5):
    """Compute axis-aligned bounding box of the gaussian centers."""
    mins = means3D.min(dim=0).values
    maxs = means3D.max(dim=0).values
    mins -= margin
    maxs += margin
    return mins, maxs


def axis_to_index(axis_name: str) -> int:
    mapping = {"x": 0, "y": 1, "z": 2}
    if axis_name not in mapping:
        raise ValueError(f"Invalid axis '{axis_name}', expected one of x/y/z")
    return mapping[axis_name]


def infer_up_axis(params_path: str, means3D: torch.Tensor) -> str:
    """Infer the up axis from dataset hint or scene geometry."""
    path_l = params_path.lower()
    if "replica" in path_l:
        return "y"
    if "isaac" in path_l:
        return "z"

    # Fallback heuristic: vertical axis often has the smallest spread indoors.
    spreads = (means3D.max(dim=0).values - means3D.min(dim=0).values).detach().cpu().numpy()
    return ["x", "y", "z"][int(np.argmin(spreads))]


def get_horizontal_axes(up_axis_idx: int):
    horiz = [i for i in [0, 1, 2] if i != up_axis_idx]
    return horiz[0], horiz[1]


def filter_gaussians_by_height(
    params: dict,
    z_min: float,
    z_max: float,
    up_axis_idx: int = 1,
) -> dict:
    """Filter gaussians to keep only those within a height range.

    This removes floor and ceiling gaussians following the 3DGSNav approach
    (Appendix C) to get a clean occupancy map.
    """
    means = params['means3D']
    axis_vals = means[:, up_axis_idx]
    mask = (axis_vals >= z_min) & (axis_vals <= z_max)
    n_gaussians = means.shape[0]

    filtered = {}
    for k, v in params.items():
        # Only filter tensors that are indexed per-gaussian along dim 0.
        # Some checkpoint entries are scalars or metadata tensors.
        if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == n_gaussians:
            filtered[k] = v[mask]
        else:
            filtered[k] = v
    return filtered


def setup_topdown_camera(
    scene_min: torch.Tensor,
    scene_max: torch.Tensor,
    resolution: int,
    up_axis_idx: int = 1,
    height_offset: float = 5.0,
    near: float = 0.01,
    far: float = 100.0,
):
    """Create a top-down orthographic-like camera looking straight down.

    Following the 3DGSNav approach (Section 3.1.3, Appendix C), we set up a
    virtual camera from above to render the scene in a top-down view.
    The camera looks down along the Y axis (negative Y direction in camera frame
    corresponds to looking down in world frame for Replica scenes).
    """
    # Build top-down camera from configurable world up-axis.
    h0_idx, h1_idx = get_horizontal_axes(up_axis_idx)

    x_min, x_max = scene_min[h0_idx].item(), scene_max[h0_idx].item()
    z_min, z_max = scene_min[h1_idx].item(), scene_max[h1_idx].item()
    up_min, up_max = scene_min[up_axis_idx].item(), scene_max[up_axis_idx].item()

    x_center = (x_min + x_max) / 2.0
    z_center = (z_min + z_max) / 2.0
    x_extent = x_max - x_min
    z_extent = z_max - z_min

    # Camera placed above the scene center along the configured up axis.
    up_top = up_max + height_offset

    # Camera basis in world coordinates:
    # - cam X aligns with first horizontal axis
    # - cam Y is negative second horizontal axis so image Y increases downward
    # - cam Z looks down along negative up axis
    e1 = torch.zeros(3, dtype=torch.float32, device="cuda")
    e2 = torch.zeros(3, dtype=torch.float32, device="cuda")
    eup = torch.zeros(3, dtype=torch.float32, device="cuda")
    e1[h0_idx] = 1.0
    e2[h1_idx] = 1.0
    eup[up_axis_idx] = 1.0

    r0 = e1
    r1 = -e2
    r2 = -eup
    R = torch.stack([r0, r1, r2], dim=0)

    cam_pos = torch.zeros(3, dtype=torch.float32, device="cuda")
    cam_pos[h0_idx] = x_center
    cam_pos[h1_idx] = z_center
    cam_pos[up_axis_idx] = up_top

    w2c = torch.eye(4, dtype=torch.float32, device="cuda")
    w2c[:3, :3] = R
    w2c[:3, 3] = -R @ cam_pos

    # Compute focal length to fit the scene into the image
    # Use a perspective camera with large focal length to approximate orthographic
    max_extent = max(x_extent, z_extent)

    # Set focal length so that the scene fills the image
    # tan(fov/2) = (max_extent/2) / focal => focal = resolution / (max_extent / distance)
    distance = max(height_offset, 1e-3)
    fx = resolution * distance / max_extent
    fy = fx
    cx = resolution / 2.0
    cy = resolution / 2.0

    # Adjust resolution to match aspect ratio
    w = int(resolution * (x_extent / max_extent))
    h = int(resolution * (z_extent / max_extent))
    # Ensure even dimensions
    w = max(w, 2)
    h = max(h, 2)

    # Recompute focal lengths for actual dimensions
    fx = w * distance / x_extent
    fy = h * distance / z_extent
    cx = w / 2.0
    cy = h / 2.0

    cam_center = torch.inverse(w2c)[:3, 3]
    w2c_t = w2c.unsqueeze(0).transpose(1, 2)

    opengl_proj = torch.tensor([
        [2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
        [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
        [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=torch.float32).cuda().unsqueeze(0).transpose(1, 2)

    full_proj = w2c_t.bmm(opengl_proj)

    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c_t,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False,
    )

    return cam, w2c, w, h


def prepare_rendervar(params: dict) -> dict:
    """Prepare render variables for the gaussian rasterizer."""
    if params['log_scales'].shape[1] == 1:
        log_scales = torch.tile(params['log_scales'], (1, 3))
    else:
        log_scales = params['log_scales']

    return {
        'means3D': params['means3D'],
        'colors_precomp': params['rgb_colors'],
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=False, device="cuda"),
    }


def prepare_opacity_rendervar(params: dict) -> dict:
    """Prepare render variables that encode opacity as color for quality visualization.

    Following 3DGSNav (Section 3.1.2), opacity values indicate observation
    completeness. Low opacity regions correspond to poorly observed or
    unobserved areas.
    """
    if params['log_scales'].shape[1] == 1:
        log_scales = torch.tile(params['log_scales'], (1, 3))
    else:
        log_scales = params['log_scales']

    opacities = torch.sigmoid(params['logit_opacities'])

    # Encode opacity as color: R=opacity, G=1 (silhouette), B=0
    opacity_color = torch.zeros_like(params['rgb_colors'])
    opacity_color[:, 0] = opacities.squeeze(-1)      # R: opacity value
    opacity_color[:, 1] = 1.0                          # G: silhouette (always 1)
    opacity_color[:, 2] = opacities.squeeze(-1) ** 2   # B: squared opacity (quality emphasis)

    return {
        'means3D': params['means3D'],
        'colors_precomp': opacity_color,
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': opacities,
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=False, device="cuda"),
    }


def render_bev(cam, rendervar):
    """Render using the gaussian rasterizer and return color, depth."""
    renderer = Renderer(raster_settings=cam)
    color, radii, depth = renderer(
        means3D=rendervar['means3D'],
        means2D=rendervar['means2D'],
        opacities=rendervar['opacities'],
        colors_precomp=rendervar['colors_precomp'],
        scales=rendervar['scales'],
        rotations=rendervar['rotations'],
    )
    return color, depth


def create_occupancy_grid(
    color_bev: torch.Tensor,
    depth_bev: torch.Tensor,
    opacity_threshold: float = 0.3,
    min_component_area: int = 30,
    morph_kernel: int = 3,
) -> np.ndarray:
    """Create an occupancy grid from the BEV rendering.

    Following 3DGSNav (Appendix C): regions with opacity below threshold
    are regarded as obstacles/occupied, while traversable areas have high opacity.

    Returns an RGB image where:
      - White: free space (high opacity, well-observed traversable area)
      - Black: occupied space (obstacles, walls, furniture)
      - Gray: unobserved/uncertain regions (low opacity)
    """
    # Use the silhouette channel (G channel from opacity rendering)
    # or compute accumulated opacity from the color rendering
    silhouette = color_bev[1].cpu().numpy()  # G channel = silhouette
    depth = depth_bev.cpu().numpy()

    # The rasterizer can return depth as [1, H, W] while silhouette is [H, W].
    # Normalize both to 2D maps so boolean masks line up.
    silhouette = np.squeeze(silhouette)
    depth = np.squeeze(depth)

    if silhouette.ndim != 2:
        raise ValueError(f"Expected 2D silhouette map, got shape {silhouette.shape}")
    if depth.ndim == 3:
        depth = depth[0]
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth map, got shape {depth.shape}")
    if depth.shape != silhouette.shape:
        raise ValueError(
            f"Depth/silhouette shape mismatch: depth={depth.shape}, silhouette={silhouette.shape}"
        )

    h, w = silhouette.shape
    occupancy = np.zeros((h, w, 3), dtype=np.uint8)

    # Unobserved: silhouette near 0 (no gaussians projected here)
    unobserved_mask = silhouette < 0.05

    # Free space from opacity confidence, occupied as complement among observed.
    observed_mask = ~unobserved_mask

    if observed_mask.any():
        free_mask = observed_mask & (silhouette >= opacity_threshold)
        occupied_mask = observed_mask & ~free_mask

        # Suppress tiny islands that typically come from sparse/noisy splats.
        if min_component_area > 0:
            try:
                import cv2

                free_u8 = (free_mask.astype(np.uint8) * 255)
                occ_u8 = (occupied_mask.astype(np.uint8) * 255)

                if morph_kernel > 1:
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
                    free_u8 = cv2.morphologyEx(free_u8, cv2.MORPH_OPEN, k)
                    free_u8 = cv2.morphologyEx(free_u8, cv2.MORPH_CLOSE, k)
                    occ_u8 = cv2.morphologyEx(occ_u8, cv2.MORPH_OPEN, k)

                def drop_small(mask_u8):
                    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
                    out = np.zeros_like(mask_u8)
                    for cid in range(1, n):
                        if stats[cid, cv2.CC_STAT_AREA] >= min_component_area:
                            out[labels == cid] = 255
                    return out

                free_mask = drop_small(free_u8) > 0
                occupied_mask = drop_small(occ_u8) > 0
            except Exception:
                pass

        occupancy[free_mask] = [255, 255, 255]       # White = free
        occupancy[occupied_mask] = [40, 40, 40]      # Dark gray = occupied
        occupancy[unobserved_mask] = [128, 128, 128]  # Gray = unobserved
    else:
        occupancy[:] = [128, 128, 128]

    return occupancy


def create_quality_map(
    opacity_color_bev: torch.Tensor,
    depth_bev: torch.Tensor,
) -> np.ndarray:
    """Create a quality heatmap from BEV opacity rendering.

    Following 3DGSNav (Section 3.1.2), opacity values indicate observation
    completeness. This generates a heatmap where:
      - Red/warm: high quality reconstruction (high accumulated opacity)
      - Blue/cool: low quality reconstruction (low accumulated opacity)
      - Black: unobserved regions (no gaussians)

    The quality map helps VLMs understand which parts of the scene are
    well-reconstructed vs uncertain/poorly observed.
    """
    # R channel = accumulated opacity, G channel = silhouette
    opacity_acc = opacity_color_bev[0].cpu().numpy()
    silhouette = opacity_color_bev[1].cpu().numpy()

    h, w = opacity_acc.shape
    quality = np.zeros((h, w, 3), dtype=np.uint8)

    # Only color observed regions
    observed = silhouette > 0.05

    if observed.any():
        # Normalize opacity to [0, 1] range within observed regions
        obs_opacity = opacity_acc[observed]
        op_min, op_max = obs_opacity.min(), obs_opacity.max()
        if op_max > op_min:
            normalized = (opacity_acc - op_min) / (op_max - op_min)
        else:
            normalized = np.ones_like(opacity_acc)
        normalized = np.clip(normalized, 0, 1)

        # Apply colormap: blue (low quality) -> green (medium) -> red (high quality)
        # Using a simple HSV-like mapping
        for i in range(h):
            for j in range(w):
                if not observed[i, j]:
                    quality[i, j] = [0, 0, 0]  # Black = unobserved
                    continue
                v = normalized[i, j]
                # Blue -> Cyan -> Green -> Yellow -> Red
                if v < 0.25:
                    t = v / 0.25
                    quality[i, j] = [int(255 * (1 - t)), 0, 255]  # Blue -> Cyan (R decreases)
                    quality[i, j] = [0, int(255 * t), 255]
                elif v < 0.5:
                    t = (v - 0.25) / 0.25
                    quality[i, j] = [0, 255, int(255 * (1 - t))]  # Cyan -> Green
                elif v < 0.75:
                    t = (v - 0.5) / 0.25
                    quality[i, j] = [int(255 * t), 255, 0]  # Green -> Yellow
                else:
                    t = (v - 0.75) / 0.25
                    quality[i, j] = [255, int(255 * (1 - t)), 0]  # Yellow -> Red

    return quality


def create_quality_map_fast(
    opacity_color_bev: torch.Tensor,
) -> np.ndarray:
    """Vectorized version of quality map creation."""
    opacity_acc = opacity_color_bev[0].cpu().numpy()
    silhouette = opacity_color_bev[1].cpu().numpy()

    h, w = opacity_acc.shape
    observed = silhouette > 0.05

    # Normalize
    if observed.any():
        obs_vals = opacity_acc[observed]
        op_min, op_max = obs_vals.min(), obs_vals.max()
        if op_max > op_min:
            normalized = np.clip((opacity_acc - op_min) / (op_max - op_min), 0, 1)
        else:
            normalized = np.ones_like(opacity_acc)
    else:
        return np.zeros((h, w, 3), dtype=np.uint8)

    # Vectorized colormap (jet-like: blue -> cyan -> green -> yellow -> red)
    r = np.zeros((h, w), dtype=np.uint8)
    g = np.zeros((h, w), dtype=np.uint8)
    b = np.zeros((h, w), dtype=np.uint8)

    v = normalized

    # Segment 1: [0, 0.25) -> Blue to Cyan
    m = observed & (v < 0.25)
    t = v[m] / 0.25
    r[m] = 0
    g[m] = (255 * t).astype(np.uint8)
    b[m] = 255

    # Segment 2: [0.25, 0.5) -> Cyan to Green
    m = observed & (v >= 0.25) & (v < 0.5)
    t = (v[m] - 0.25) / 0.25
    r[m] = 0
    g[m] = 255
    b[m] = (255 * (1 - t)).astype(np.uint8)

    # Segment 3: [0.5, 0.75) -> Green to Yellow
    m = observed & (v >= 0.5) & (v < 0.75)
    t = (v[m] - 0.5) / 0.25
    r[m] = (255 * t).astype(np.uint8)
    g[m] = 255
    b[m] = 0

    # Segment 4: [0.75, 1.0] -> Yellow to Red
    m = observed & (v >= 0.75)
    t = (v[m] - 0.75) / 0.25
    r[m] = 255
    g[m] = (255 * (1 - t)).astype(np.uint8)
    b[m] = 0

    quality = np.stack([r, g, b], axis=-1)
    # Unobserved stays black (0, 0, 0)
    return quality


def infer_default_traj_path(params_path: str) -> str:
    """Infer dataset trajectory file from an experiments/<group>/<run>/params*.npz path."""
    p = os.path.normpath(params_path)
    parts = p.split(os.sep)
    try:
        exp_idx = parts.index("experiments")
    except ValueError:
        return ""

    if exp_idx + 2 >= len(parts):
        return ""

    group = parts[exp_idx + 1]
    run_name = parts[exp_idx + 2]
    # Typical run name is <scene>_<seed>, e.g., office0_0.
    scene = run_name.rsplit("_", 1)[0] if "_" in run_name else run_name
    candidate = os.path.join("data", group, scene, "traj.txt")
    return candidate if os.path.isfile(candidate) else ""


def load_positions_from_traj_txt(traj_path: str, target_len: int = 0) -> np.ndarray:
    """Load c2w trajectory and convert to frame-0 relative positions."""
    with open(traj_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    if not lines:
        return np.zeros((0, 3), dtype=np.float32)

    poses = np.array([np.fromstring(ln, sep=" ") for ln in lines], dtype=np.float64)
    poses = poses.reshape(-1, 4, 4)
    c2w0_inv = np.linalg.inv(poses[0])
    rel_c2w = np.einsum("ij,njk->nik", c2w0_inv, poses)
    positions = rel_c2w[:, :3, 3].astype(np.float32)

    if target_len > 0 and len(positions) != target_len:
        idx = np.linspace(0, len(positions) - 1, target_len).round().astype(np.int64)
        positions = positions[idx]
    return positions


def load_positions_from_params(params: dict) -> np.ndarray:
    """Load camera positions from params cam_{rot,trans} tensors."""
    if 'cam_trans' not in params or 'cam_unnorm_rots' not in params:
        return np.zeros((0, 3), dtype=np.float32)

    cam_trans = params['cam_trans']
    cam_rots = params['cam_unnorm_rots']
    num_frames = cam_trans.shape[-1]

    w2c_init = params.get('w2c', None)
    if w2c_init is not None and w2c_init.dim() == 1:
        w2c_init = w2c_init.reshape(4, 4)

    positions_world = []
    from utils.slam_external import build_rotation
    for t in range(num_frames):
        cam_rot = F.normalize(cam_rots[..., t])
        cam_tran = cam_trans[..., t]

        rel_w2c = torch.eye(4).cuda().float()
        rel_w2c[:3, :3] = build_rotation(cam_rot)
        rel_w2c[:3, 3] = cam_tran
        full_w2c = rel_w2c if w2c_init is None else (rel_w2c @ w2c_init)
        cam_pos = torch.inverse(full_w2c)[:3, 3]
        positions_world.append(cam_pos.cpu().numpy())

    return np.array(positions_world, dtype=np.float32)


def add_trajectory_overlay(
    bev_image: np.ndarray,
    params: dict,
    scene_min: torch.Tensor,
    scene_max: torch.Tensor,
    up_axis_idx: int = 1,
    smoothing_window: int = 9,
    trajectory_source: str = "auto",
    traj_path: str = "",
    params_path: str = "",
) -> np.ndarray:
    """Overlay camera trajectory on the BEV image.

    Following 3DGSNav (Section 3.2.3), the BEV depicts the historical trajectory
    in blue and the current position as a marker.
    """
    try:
        import cv2
    except ImportError:
        print("Warning: cv2 not available, skipping trajectory overlay")
        return bev_image

    h, w = bev_image.shape[:2]
    overlay = bev_image.copy()

    target_len = int(params['cam_trans'].shape[-1]) if 'cam_trans' in params else 0
    positions_params = load_positions_from_params(params)
    positions_world = positions_params

    if trajectory_source in ["gt", "auto"]:
        use_gt = trajectory_source == "gt"
        if trajectory_source == "auto" and len(positions_params) > 1:
            traj_spread = np.ptp(positions_params, axis=0)
            # If estimated trajectory is nearly degenerate, fallback to GT traj file.
            use_gt = float(np.linalg.norm(traj_spread)) < 0.1

        if use_gt:
            resolved_traj = traj_path if traj_path else infer_default_traj_path(params_path)
            if resolved_traj:
                try:
                    positions_world = load_positions_from_traj_txt(resolved_traj, target_len=target_len)
                except Exception as exc:
                    print(f"Warning: failed to load GT trajectory from {resolved_traj}: {exc}")

    if len(positions_world) == 0:
        return overlay

    if smoothing_window > 1 and smoothing_window % 2 == 1 and len(positions_world) >= smoothing_window:
        k = np.ones(smoothing_window, dtype=np.float32) / float(smoothing_window)
        pad = smoothing_window // 2
        for d in range(3):
            v = positions_world[:, d]
            v_pad = np.pad(v, (pad, pad), mode='edge')
            positions_world[:, d] = np.convolve(v_pad, k, mode='valid')

    # Project to BEV image coordinates using the configured up axis.
    h0_idx, h1_idx = get_horizontal_axes(up_axis_idx)
    x_min, x_max = scene_min[h0_idx].item(), scene_max[h0_idx].item()
    z_min, z_max = scene_min[h1_idx].item(), scene_max[h1_idx].item()

    def world_to_pixel(x_world, z_world):
        x_den = max(x_max - x_min, 1e-6)
        z_den = max(z_max - z_min, 1e-6)
        px = int((x_world - x_min) / x_den * (w - 1))
        # Flip vertical axis to match top-down camera convention in setup_topdown_camera.
        py = int((z_max - z_world) / z_den * (h - 1))
        return np.clip(px, 0, w - 1), np.clip(py, 0, h - 1)

    # Draw trajectory line in blue
    for i in range(1, len(positions_world)):
        p1 = world_to_pixel(positions_world[i - 1, h0_idx], positions_world[i - 1, h1_idx])
        p2 = world_to_pixel(positions_world[i, h0_idx], positions_world[i, h1_idx])
        cv2.line(overlay, p1, p2, (255, 100, 100), thickness=2)  # Blue (BGR)

    # Draw current position as a triangle (purple, following 3DGSNav)
    if len(positions_world) > 0:
        last_pos = positions_world[-1]
        px, py = world_to_pixel(last_pos[h0_idx], last_pos[h1_idx])
        triangle_size = max(5, min(w, h) // 40)
        pts = np.array([
            [px, py - triangle_size],
            [px - triangle_size // 2, py + triangle_size // 2],
            [px + triangle_size // 2, py + triangle_size // 2],
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (200, 0, 200))  # Purple

    return overlay


def main():
    parser = argparse.ArgumentParser(
        description="Generate BEV images from a 3DGS map"
    )
    parser.add_argument(
        "--params_path", type=str, required=True,
        help="Path to params.npz checkpoint",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: same dir as params.npz)",
    )
    parser.add_argument(
        "--resolution", type=int, default=512,
        help="BEV image resolution (longest side)",
    )
    parser.add_argument(
        "--floor_height_percentile", type=float, default=10,
        help="Percentile for floor height cutoff (filters floor gaussians)",
    )
    parser.add_argument(
        "--ceil_height_percentile", type=float, default=90,
        help="Percentile for ceiling height cutoff (filters ceiling gaussians)",
    )
    parser.add_argument(
        "--occupancy_threshold", type=float, default=0.3,
        help="Opacity threshold for occupancy (from 3DGSNav: tau=0.3)",
    )
    parser.add_argument(
        "--margin", type=float, default=0.5,
        help="Margin around scene bounds (meters)",
    )
    parser.add_argument(
        "--no_trajectory", action="store_true",
        help="Skip trajectory overlay on BEV images",
    )
    parser.add_argument(
        "--height_offset", type=float, default=5.0,
        help="Camera height above scene for top-down view",
    )
    parser.add_argument(
        "--min_component_area", type=int, default=30,
        help="Remove occupancy/free connected components smaller than this many pixels (0 disables).",
    )
    parser.add_argument(
        "--morph_kernel", type=int, default=3,
        help="Morphological kernel size for occupancy denoising (1 disables).",
    )
    parser.add_argument(
        "--trajectory_smoothing", type=int, default=9,
        help="Odd moving-average window for trajectory smoothing in BEV pixels (1 disables).",
    )
    parser.add_argument(
        "--trajectory_source", type=str, default="auto", choices=["auto", "params", "gt"],
        help="Trajectory source: params (estimated), gt (from traj.txt), or auto fallback.",
    )
    parser.add_argument(
        "--traj_path", type=str, default="",
        help="Optional path to traj.txt (c2w per line) used when trajectory_source=gt or auto fallback.",
    )
    parser.add_argument(
        "--up_axis", type=str, default="auto", choices=["auto", "x", "y", "z"],
        help="World up axis. Use auto to infer from dataset/path (Replica->y, Isaac->z).",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.params_path), "bev")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load gaussian map
    print(f"Loading gaussians from: {args.params_path}")
    params = load_params(args.params_path)
    n_gaussians = params['means3D'].shape[0]
    print(f"  Loaded {n_gaussians} gaussians")

    # Compute scene bounds
    means = params['means3D']
    scene_min, scene_max = compute_scene_bounds(means, margin=args.margin)
    print(f"  Scene bounds: min={scene_min.cpu().numpy()}, max={scene_max.cpu().numpy()}")

    # Determine up axis (Replica usually y-up; IsaacSim usually z-up)
    up_axis = infer_up_axis(args.params_path, means) if args.up_axis == "auto" else args.up_axis
    up_axis_idx = axis_to_index(up_axis)
    spreads = (means.max(dim=0).values - means.min(dim=0).values).detach().cpu().numpy()
    print(f"  Using up axis: {up_axis.upper()} (spreads X/Y/Z = {spreads[0]:.3f}/{spreads[1]:.3f}/{spreads[2]:.3f})")

    # Filter by height (remove floor/ceiling following 3DGSNav Appendix C)
    h_vals = means[:, up_axis_idx].cpu().numpy()
    h_floor = np.percentile(h_vals, args.floor_height_percentile)
    h_ceil = np.percentile(h_vals, args.ceil_height_percentile)
    print(f"  Height filter: keeping {up_axis.upper()} in [{h_floor:.3f}, {h_ceil:.3f}]")

    filtered_params = filter_gaussians_by_height(
        params, h_floor, h_ceil, up_axis_idx=up_axis_idx
    )
    n_filtered = filtered_params['means3D'].shape[0]
    print(f"  After filtering: {n_filtered} gaussians ({n_filtered/n_gaussians*100:.1f}%)")

    # Setup top-down camera
    print(f"  Setting up top-down camera (resolution={args.resolution})")
    cam, w2c, img_w, img_h = setup_topdown_camera(
        scene_min, scene_max,
        resolution=args.resolution,
        up_axis_idx=up_axis_idx,
        height_offset=args.height_offset,
    )
    print(f"  BEV image size: {img_w} x {img_h}")

    # ── 1. Render RGB BEV ──
    print("Rendering RGB BEV...")
    with torch.no_grad():
        rendervar = prepare_rendervar(filtered_params)
        rgb_bev, depth_bev = render_bev(cam, rendervar)

    rgb_np = (rgb_bev.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    # ── 2. Render Opacity/Quality BEV ──
    print("Rendering quality BEV...")
    with torch.no_grad():
        opacity_rendervar = prepare_opacity_rendervar(filtered_params)
        opacity_bev, opacity_depth_bev = render_bev(cam, opacity_rendervar)

    # ── 3. Create occupancy grid ──
    print("Creating occupancy grid...")
    occupancy_grid = create_occupancy_grid(
        opacity_bev, opacity_depth_bev,
        opacity_threshold=args.occupancy_threshold,
        min_component_area=args.min_component_area,
        morph_kernel=args.morph_kernel,
    )

    # ── 4. Create quality heatmap ──
    print("Creating quality heatmap...")
    quality_map = create_quality_map_fast(opacity_bev)

    # ── 5. Add trajectory overlays ──
    if not args.no_trajectory:
        print("Adding trajectory overlays...")
        rgb_with_traj = add_trajectory_overlay(
            rgb_np, params, scene_min, scene_max,
            up_axis_idx=up_axis_idx,
            smoothing_window=args.trajectory_smoothing,
            trajectory_source=args.trajectory_source,
            traj_path=args.traj_path,
            params_path=args.params_path,
        )
        occupancy_with_traj = add_trajectory_overlay(
            occupancy_grid, params, scene_min, scene_max,
            up_axis_idx=up_axis_idx,
            smoothing_window=args.trajectory_smoothing,
            trajectory_source=args.trajectory_source,
            traj_path=args.traj_path,
            params_path=args.params_path,
        )
        quality_with_traj = add_trajectory_overlay(
            quality_map, params, scene_min, scene_max,
            up_axis_idx=up_axis_idx,
            smoothing_window=args.trajectory_smoothing,
            trajectory_source=args.trajectory_source,
            traj_path=args.traj_path,
            params_path=args.params_path,
        )

    # ── 6. Save outputs ──
    try:
        import cv2
        use_cv2 = True
    except ImportError:
        from PIL import Image
        use_cv2 = False

    def save_image(img_rgb, path):
        if use_cv2:
            import cv2
            cv2.imwrite(path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        else:
            Image.fromarray(img_rgb).save(path)

    save_image(rgb_np, os.path.join(args.output_dir, "bev_rgb.png"))
    save_image(occupancy_grid, os.path.join(args.output_dir, "bev_occupancy.png"))
    save_image(quality_map, os.path.join(args.output_dir, "bev_quality.png"))

    if not args.no_trajectory:
        save_image(rgb_with_traj, os.path.join(args.output_dir, "bev_rgb_trajectory.png"))
        save_image(occupancy_with_traj, os.path.join(args.output_dir, "bev_occupancy_trajectory.png"))
        save_image(quality_with_traj, os.path.join(args.output_dir, "bev_quality_trajectory.png"))

    # Save depth map as numpy array for further analysis
    depth_np = depth_bev.cpu().numpy()
    np.save(os.path.join(args.output_dir, "bev_depth.npy"), depth_np)

    print(f"\nOutputs saved to: {args.output_dir}/")
    print(f"  bev_rgb.png              — Top-down RGB rendering")
    print(f"  bev_occupancy.png        — Occupancy grid (white=free, dark=occupied, gray=unobserved)")
    print(f"  bev_quality.png          — Quality heatmap (red=high, blue=low, black=unobserved)")
    print(f"  bev_depth.npy            — Raw depth values")
    if not args.no_trajectory:
        print(f"  *_trajectory.png         — Versions with camera trajectory overlay")


if __name__ == "__main__":
    main()
