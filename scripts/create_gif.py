import argparse
import os
from PIL import Image


def create_gif(input_folder, output_filename, duration=200):
    """
    Creates a GIF from all PNG images in a folder.

    :param input_folder: Path to the folder containing PNGs
    :param output_filename: The name of the resulting GIF (e.g., 'animation.gif')
    :param duration: Time between frames in milliseconds
    """
    if not os.path.isdir(input_folder):
        print(f"Input folder does not exist: {input_folder}")
        return

    images = []

    # Get all png files and sort them alphabetically
    files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith('.png')])

    if not files:
        print("No PNG images found in the specified folder.")
        return

    print(f"Processing {len(files)} images...")

    for filename in files:
        file_path = os.path.join(input_folder, filename)
        img = Image.open(file_path)
        # Pillow works best if all images are in the same mode (e.g., RGBA or RGB)
        images.append(img.convert("RGBA"))

    # Save the GIF
    # duration is in milliseconds; loop=0 means it will loop forever
    images[0].save(
        output_filename,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        disposal=2  # Helps prevent ghosting/transparency artifacts
    )

    print(f"Success! GIF saved as: {output_filename}")


# --- Usage ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a GIF from PNG files in a folder")
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to the folder containing PNG images",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="animation.gif",
        help="Output GIF file path (default: animation.gif)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=200,
        help="Time between frames in milliseconds (default: 200)",
    )

    args = parser.parse_args()
    create_gif(
        input_folder=args.input_folder,
        output_filename=args.output_filename,
        duration=args.duration,
    )
