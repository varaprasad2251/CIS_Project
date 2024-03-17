import os
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
import argparse

def process_and_save_image(image_path, output_dir, start_pos, patch_area):
    # Read the image as a NumPy array
    image = imageio.imread(image_path)
    
    # Calculate patch dimensions as a percentage of the shortest image dimension, keeping it square
    shorter_side = min(image.shape[:2])
    patch_side = int(shorter_side * patch_area)  # Convert patch_area percentage to actual pixels

    # Use start_pos to determine where to extract the patch
    start_y, start_x = start_pos

    # Calculate new positions for the patch based on the specified area
    position_y = start_y + patch_side
    position_x = start_x + patch_side

    # Extract and modify the patch
    patch = image[start_y:start_y + patch_side, start_x:start_x + patch_side].copy()
    if len(patch.shape) == 2:  # Grayscale image
        patch_gray = patch
    else:  # Color image
        patch_gray = np.mean(patch, axis=2).astype(np.uint8)
        patch_gray = np.stack([patch_gray, patch_gray, patch_gray], axis=-1)

    # Add the modified patch back to the image at the calculated new position
    image[position_y:position_y + patch_side, position_x:position_x + patch_side] = patch_gray

    # Construct new output directory path
    new_output_dir = os.path.join(output_dir, f"{start_y}_{start_x}_{patch_area}")
    os.makedirs(new_output_dir, exist_ok=True)

    # Save the modified image to the output directory
    output_path = os.path.join(new_output_dir, os.path.basename(image_path))
    imageio.imwrite(output_path, image)

# Define the input and output directories
# input_dir = 'input_raw_images'
# output_dir = 'raw_images_patch'

parser = argparse.ArgumentParser(description='Process images to apply adversarial patches.')
parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images.')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed images.')
parser.add_argument('--start_pos', type=int, nargs=2, required=True, help='Starting position (y, x) for patch extraction.')
parser.add_argument('--patch_area', type=float, required=True, help='Patch area as a percentage of the shorter image side (0.1 for 10%).')
args = parser.parse_args()
# Process each image in the input directory
for filename in os.listdir(args.input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(args.input_dir, filename)
        process_and_save_image(image_path, args.output_dir, tuple(args.start_pos), args.patch_area)
        print(f"Processed and saved {filename}")