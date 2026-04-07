import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Hardcoded from data/IsaacSim/office0/Cameras_Camera/camera_params/camera_params_00000.json
FX = 1662.769
FY = 1662.769
CX = 960.0
CY = 540.0


def load_depth(depth_path: str) -> tuple[np.ndarray, bool]:
    ext = os.path.splitext(depth_path)[1].lower()

    if ext == ".npy":
        depth = np.load(depth_path).astype(np.float32)
        return depth, True

    img = Image.open(depth_path)
    arr = np.array(img)

    # If depth is a 3-channel color map, values are not metric.
    if arr.ndim == 3:
        depth = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.float32)
        return depth, False

    depth = arr.astype(np.float32)

    # Heuristic for IsaacSim raw depth PNGs stored in millimeters.
    if np.nanmax(depth) > 1000.0:
        depth *= 0.001
        return depth, True

    return depth, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Depth-only click visualizer with fixed camera intrinsics.")
    parser.add_argument("--depth_path", type=str, required=True, help="Full path to depth file (.npy or image)")
    args = parser.parse_args()

    depth, metric_depth = load_depth(args.depth_path)
    depth_unit = "m" if metric_depth else "raw"

    print(f"Using fixed intrinsics: fx={FX:.3f}, fy={FY:.3f}, cx={CX:.3f}, cy={CY:.3f}")
    if not metric_depth:
        print("Warning: depth appears non-metric (likely colorized). XYZ/range are in relative units.")

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(depth, cmap="turbo")
    ax.set_title("Depth")
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Depth ({depth_unit})")

    click_text = fig.text(0.02, 0.01, "Click on the depth image to inspect point distance.", fontsize=10)

    def on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))
        h, w = depth.shape[:2]
        if x < 0 or x >= w or y < 0 or y >= h:
            return

        z = float(depth[y, x])
        if not np.isfinite(z) or z <= 0.0:
            msg = f"pixel=({x}, {y}) depth=invalid"
            print(msg)
            click_text.set_text(msg)
            fig.canvas.draw_idle()
            return

        X = (x - CX) * z / FX
        Y = (y - CY) * z / FY
        Z = z
        rng = float(np.sqrt(X * X + Y * Y + Z * Z))
        msg = (
            f"pixel=({x}, {y}) depth={z:.4f} {depth_unit} | "
            f"XYZ=({X:.4f}, {Y:.4f}, {Z:.4f}) {depth_unit} | "
            f"range={rng:.4f} {depth_unit}"
        )
        print(msg)
        click_text.set_text(msg)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)

    fig.suptitle(f"Depth Click Visualizer | {os.path.basename(args.depth_path)}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
