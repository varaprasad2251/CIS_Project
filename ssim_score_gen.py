import os
import argparse
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread
from skimage import img_as_float
import pandas as pd
import numpy as np

# Function to calculate SSIM for each pair of images in folder1 and each subfolder of folder2
def calculate_ssim(folder1, folder2):
    file_names = []
    ssim_scores = []
    folder2_subfolders = []
    
    files1 = sorted(os.listdir(folder1))
    subfolders = [f.path for f in os.scandir(folder2) if f.is_dir()]
    
    for subfolder in subfolders:
        files2 = sorted(os.listdir(subfolder))
        
        for f1, f2 in zip(files1, files2):
            img1 = imread(os.path.join(folder1, f1), as_gray=True)
            img2 = imread(os.path.join(subfolder, f2), as_gray=True)
            
            # Convert images to float type
            img1 = img_as_float(img1)
            img2 = img_as_float(img2)
            
            # Ensure the images have the same dimension
            img1_resized = np.resize(img1, (min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])))
            img2_resized = np.resize(img2, (min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])))
            
            # Calculate SSIM
            score = ssim(img1_resized, img2_resized, data_range=img1_resized.max() - img1_resized.min())
            file_names.append(f1)
            ssim_scores.append(score)
            folder2_subfolders.append(os.path.basename(subfolder))
    
    df = pd.DataFrame({'Folder1': folder1, 'Folder2': folder2, 'Subfolder': folder2_subfolders, 'Image Name': file_names, 'SSIM Score': ssim_scores})
    return df

# Parse command line arguments
parser = argparse.ArgumentParser(description="Calculate SSIM between images in two folders.")
parser.add_argument("folder1", type=str, help="Path to the first folder")
parser.add_argument("folder2", type=str, help="Path to the second folder containing subfolders")
args = parser.parse_args()

# Calculate SSIM and display the table
ssim_df = calculate_ssim(args.folder1, args.folder2)
print(ssim_df)
