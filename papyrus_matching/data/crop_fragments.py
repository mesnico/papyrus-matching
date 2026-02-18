#!/usr/bin/env python3

"""
Batch crops all PNG images in a folder based on their opaque content.

This script uses OpenCV and NumPy to find the smallest bounding box
containing all non-transparent pixels (alpha > 0) and crops the
image to that box.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import sys

def crop_opaque_content_numpy(image_array: np.ndarray) -> np.ndarray | None:
    """
    Efficiently crops a NumPy array (representing an image) to the smallest
    bounding box containing all opaque content.

    Args:
        image_array: The input NumPy array. Expects a 4-channel BGRA format.

    Returns:
        A new, cropped NumPy array, or None if the input is entirely transparent.
    """
    # --- 1. Check for alpha channel ---
    if image_array.ndim < 3 or image_array.shape[2] != 4:
        print("    Warning: Image does not have an alpha channel. Skipping crop.", file=sys.stderr)
        return image_array

    # --- 2. Find bounding box using OpenCV ---
    
    # Extract the Alpha channel. In OpenCV (BGRA), this is the 4th channel (index 3).
    alpha_channel = (image_array[:, :, 3] > 250).astype(np.uint8)

    # Create a binary mask of all non-transparent pixels.
    # The original function used `> 250`, which is a very high threshold.
    # We use `> 0` to match the docstring "all opaque content".
    # This creates a mask where 0 = transparent, 255 = opaque.
    _, binary_mask = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)

    # Define a kernel (e.g., a 3x3 square)
    kernel = np.ones((3, 3), np.uint8)

    # Perform a morphological opening to remove small noise
    # This helps ignore tiny, stray pixels.
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find the coordinates of all opaque pixels
    coords = cv2.findNonZero(cleaned_mask)

    # If the image is fully transparent (or all pixels were noise), coords will be None
    if coords is None:
        return None

    # Get the bounding rectangle
    x, y, w, h = cv2.boundingRect(coords)

    # --- 3. Crop the NumPy array using slicing ---
    cropped_arr = image_array[y:y+h, x:x+w]

    return cropped_arr

def main():
    """
    Main function to parse arguments and process images.
    """
    parser = argparse.ArgumentParser(
        description="Batch crop PNG images to their opaque content.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="The folder containing the original PNG images."
    )
    # parser.add_argument(
    #     "output_folder",
    #     type=str,
    #     help="The folder where cropped images will be saved."
    # )
    
    args = parser.parse_args()

    args.output_folder = args.input_folder + "_cropped"

    # Convert string paths to Path objects
    input_dir = Path(args.input_folder)
    output_dir = Path(args.output_folder)

    # --- 1. Validate paths ---
    if not input_dir.is_dir():
        print(f"Error: Input folder does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input folder:  {input_dir.resolve()}")
    print(f"Output folder: {output_dir.resolve()}\n")

    # --- 2. Find and process images ---
    image_paths = list(input_dir.glob('*.png'))

    if not image_paths:
        print("No .png files found in the input folder.")
        return

    processed_count = 0
    skipped_count = 0
    
    for image_path in image_paths:
        print(f"Processing {image_path.name}...")
        
        # Read the image with its alpha channel (IMREAD_UNCHANGED)
        # OpenCV reads as BGRA
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        
        if image is None:
            print(f"    Warning: Could not read image. Skipping.")
            skipped_count += 1
            continue

        # --- 3. Crop the image ---
        cropped_image = crop_opaque_content_numpy(image)
        
        # --- 4. Save the result ---
        if cropped_image is None:
            print(f"    Image is fully transparent. Skipping.")
            skipped_count += 1
            continue
            
        output_path = output_dir / image_path.name
        
        try:
            cv2.imwrite(str(output_path), cropped_image)
            print(f"    Saved cropped image to {output_path}")
            processed_count += 1
        except Exception as e:
            print(f"    Error saving image: {e}")
            skipped_count += 1
            
    print("\n--- Batch processing complete ---")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped:                {skipped_count}")

if __name__ == "__main__":
    main()