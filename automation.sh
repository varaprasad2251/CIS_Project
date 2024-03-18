#!/bin/bash

# Define the current working directory
current_working_directory=$(pwd)

# Assign input and output folder locations for raw images
input_folder_raw="$current_working_directory/input_raw_images"
output_folder_raw="$current_working_directory/output_images"

# Assign input and output folder locations for patch images
input_folder_patch="$current_working_directory/raw_images_patch"
output_folder_patch="$current_working_directory/output_images_patch"



coordinates=(20 30 60 90)

# Define patch area values
patch_area_values=(0.2 0.25 0.3 0.33 0.4)  # Example values for --patch_area

# Loop over Y coordinates
for y_coord in "${coordinates[@]}"; do
    # Loop over X coordinates
    for x_coord in "${coordinates[@]}"; do
        # Loop over patch area values
        for patch_area in "${patch_area_values[@]}"; do
            # Run Python script with current values
            python3 patch_gen.py --input_dir "$input_folder_raw" --output_dir "$input_folder_patch" --start_pos "$y_coord" "$x_coord" --patch_area "$patch_area"
        done
    done
done




# Check if input folder for raw images exists
if [ ! -d "$input_folder_raw" ]; then
    echo "Input folder '$input_folder_raw' does not exist."
    exit 1
fi

# Check if output folder for raw images exists, if not create it
if [ ! -d "$output_folder_raw" ]; then
    mkdir -p "$output_folder_raw" || { echo "Failed to create output folder '$output_folder_raw'"; exit 1; }
fi

# Check if input folder for patch images exists
if [ ! -d "$input_folder_patch" ]; then
    echo "Input folder '$input_folder_patch' does not exist."
    exit 1
fi

# Check if output folder for patch images exists, if not create it
if [ ! -d "$output_folder_patch" ]; then
    mkdir -p "$output_folder_patch" || { echo "Failed to create output folder '$output_folder_patch'"; exit 1; }
fi

# Define the error log files with full paths
error_log_raw="$current_working_directory/matlab_error_raw.log"
error_log_patch="$current_working_directory/matlab_error_patch.log"

# MATLAB command to execute the function for raw images and redirect stderr to the error log file
matlab -nodisplay -nosplash -r "try, addpath('$current_working_directory/DevelopCameraProcessingPipelineUsingDeepLearningExample'); ProcessImages('$input_folder_raw', '$output_folder_raw'), catch err, fprintf(2, 'Error: %s\n', err.message), exit(1), end, exit(0);" 2> "$error_log_raw"

# MATLAB command to execute the function for patch images and redirect stderr to the error log file
matlab -nodisplay -nosplash -r "try, addpath('$current_working_directory/DevelopCameraProcessingPipelineUsingDeepLearningExample'); ProcessImages('$input_folder_patch', '$output_folder_patch'), catch err, fprintf(2, 'Error: %s\n', err.message), exit(1), end, exit(0);" 2> "$error_log_patch"

# Check if MATLAB commands were successful
if [ -s "$error_log_raw" ] || [ -s "$error_log_patch" ]; then
    echo "MATLAB function execution failed. Check '$error_log_raw' and '$error_log_patch' for details."
else
    echo "MATLAB function execution successful"
fi


python3 ssim_score_gen.py "output_images" "output_images_patch"