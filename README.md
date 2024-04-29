# CIS_Project

<!-- Command to generate raw images with patch from raw images
```bash
python3 patch_gen.py --input_dir input_raw_images --output_dir raw_images_patch --start_pos 50 50 --patch_area 0.1
```


Command to generate ssim score between final images and final images with patch
```bash
python3 ssim_score_gen.py output_images output_images_patch
```

Go into project folder and then run 

```bash
./automation.sh
``` -->

# DeepISP Patch Optimization Script

This script is designed to optimize patches on raw images using the DeepISP model. It provides functionality to evaluate the impact of patches on Structural Similarity Index (SSIM) and visualize the results.

## Usage

```bash
python3 patch_optimization.py  [-w WEIGHTS_FILE] [-dataset DATASET_PATH] [-path MAIN_PATH] [-res RESULTS_FOLDER]
[-orig ORIGINAL_IMAGES_FOLDER] [-in INPUT_FILE_NAME] [-tmp TMP_PATH] [-pf PROCESS_FOLDER]
[-pos_x POSITION_X] [-pos_y POSITION_Y] [-size PATCH_SIZE]
```


## Arguments

- `-w`, `--weights_file`: Name of the best weight file to be used for optimization (only prefix while evaluating). Default is `'weights2_0191.h5'`.

- `-dataset`, `--dataset_path`: Complete path for the dataset containing input raw images. Default is `'./DeepISP/input_raw_images/'`.

- `-path`, `--main_path`: Main path where the result/experiment folders are stored. Default is `'./DeepISP/'`.

- `-res`, `--results_folder`: Folder to save patch results. Default is `'patch_results'`.

- `-orig`, `--original_images_folder`: Folder containing original images from the Zurich Dataset. Default is `'Zurich_Original'`.

- `-in`, `--input_file_name`: Input image file name without extension. Default is `'663'`.

- `-tmp`, `--tmp_path`: If output images need to be stored in results/tmp path. Default is `'false'`.

- `-pf`, `--process_folder`: To process a whole folder or a single image. Default is `'false'`.

- `-pos_x`, `--position_x`: X coordinate of the patch start position. Default is `None`.

- `-pos_y`, `--position_y`: Y coordinate of the patch start position. Default is `None`.

- `-size`, `--patch_size`: Percentage of the patch in the total image size. Default is `10`.

## Example

To optimize a patch for a single image with a specific patch size and have a specified start X position and Start Y position 

```bash
python patch_optimization.py -in 1115 -size 20 -pos_x 10 -pos_y 10
```



**in** is the input file number from DeepISP/input_raw_images folder. eg: 1115



# YOLO Object Detection Script

This script performs object detection using the YOLO (You Only Look Once) model. It utilizes a pretrained YOLOv8n detection model to detect objects in images.

## Usage

```bash
python yolo_object_detection.py [-in INPUT_FILE_NAME] [-path MAIN_PATH] [-res RESULTS_FOLDER]
[-orig ORIGINAL_IMAGES_FOLDER] [-tmp TMP_PATH] [-pf PROCESS_FOLDER]
```


## Arguments

- `-in`, `--input_file_name`: Input image file name without extension. Default is `'0'`.

- `-path`, `--main_path`: Main path where the result/experiment folders are stored. Default is `'./DeepISP/'`.

- `-res`, `--results_folder`: Folder to save detection results. Default is `'patch_results'`.

- `-orig`, `--original_images_folder`: Folder containing original images from the Zurich Dataset. Default is `'Zurich_Original'`.

- `-tmp`, `--tmp_path`: If images need to be fetched from results/tmp path. Default is `'false'`.

- `-pf`, `--process_folder`: To process a whole folder or single image. Default is `'false'`.

## Example

To perform object detection on a single image:

```bash
python yolo_object_detection.py -in 1115
```