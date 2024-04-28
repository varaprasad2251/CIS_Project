from ultralytics import YOLO
import os
import argparse
import json
import pandas as pd
parser = argparse.ArgumentParser()

parser.add_argument('-in' ,'--input_file_name', type = str, default = '0' , help = 'input image file name')
parser.add_argument('-path' ,'--main_path', type = str, default = './DeepISP/' , help = 'main path where the result/experiment folders are stored')
parser.add_argument('-res' ,'--results_folder', type = str, default = 'patch_results' , help = 'folder to save patch results')
parser.add_argument('-orig' ,'--orig_images_folder', type = str, default = 'Zurich_Original' , help = 'Folder containing original images from Zurich Dataset')
parser.add_argument('-tmp' ,'--tmp_path', type = str, default = 'false' , help = 'if images need to be fetched from results/tmp path')
parser.add_argument('-pf' ,'--process_folder', type = str, default = 'false' , help = 'To process whole folder or single image')

args = parser.parse_args()
input_file_name = args.input_file_name
tmp_path = args.tmp_path
process_folder = args.process_folder
current_path = args.main_path
orig_img_folder = args.orig_images_folder

# model.train(data='coco128.yaml', epochs=3)  # Uncomment to train the model

class_id_to_name = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush'
}

def compute_confidence_scores_for_folder(original_folder, pred_folder):
    """
    Computes confidence scores for all images in the original folder against predicted images and predicted images with patch derived from pred folder.
    Parameters:
        original_folder (str): Path to the folder containing the original images.
        raw_folder (str): Path to the folder containing the raw images.
    """
    # List all files in the original images folder
    original_files = [f for f in os.listdir(original_folder) if f.endswith('.png') or f.endswith('.jpg')]
    
    total_confidence_scores = []
    model = YOLO('yolov8n.pt')  # load a pretrained YOLOv8n detection model
    class_names = model.names
    for file_name in original_files:
        original_path = os.path.join(original_folder, file_name)
        base_name = os.path.splitext(file_name)[0]
        pred_original_path = os.path.join(pred_folder, "original_image_" + base_name + '.png')
        pred_patch_path = os.path.join(pred_folder, "patch_image_" + base_name + '.png')

        # Determine the correct path if it exists
        if not os.path.exists(pred_original_path):
            print(f"Skipping {file_name}: No corresponding raw file found.")
            continue
        imgs = [original_path, pred_original_path, pred_patch_path]
        results = model(imgs)
        # Extract confidence scores from the results
        confidence_scores = {}
        for result in results:
            # Assuming each 'result' is a dictionary with 'class' and 'confidence' keys
            confidence_score = {'class': result.boxes.cls.tolist(), 'confidence': result.boxes.conf.tolist()}
            confidence_scores[result.path] = confidence_score
        # confidence_score = {result['class']: result['confidence'] for result in results.pred}
        # confidence_scores.append({file_name: confidence_score})
        total_confidence_scores.append(confidence_scores)

    # Group by image with lists of classes and confidences
    grouped_data = {}
    for entry in total_confidence_scores:
        for image_path, details in entry.items():
            if image_path not in grouped_data:
                grouped_data[image_path] = {'Class': [], 'Confidence': []}
            grouped_data[image_path]['Class'].extend(details['class'])
            grouped_data[image_path]['Confidence'].extend(details['confidence'])

    # Flatten the grouped data
    flattened_grouped_data = []
    for image_path, details in grouped_data.items():
        flattened_grouped_data.append({
            'Image': image_path,
            'Class': details['Class'],
            'Confidence': details['Confidence']
        })

    # Convert to pandas DataFrame
    df_grouped = pd.DataFrame(flattened_grouped_data)
    df_grouped['Class'] = df_grouped['Class'].apply(lambda classes: [class_id_to_name.get(int(cl), 'Unknown') for cl in classes])
    # print(df_grouped)
    
    return total_confidence_scores, df_grouped

def main():
    if process_folder == 'true':
        original_folder = current_path + orig_img_folder + "/" 
        pred_folder = current_path + "patch_results" + "/"
        confidence_scores, df = compute_confidence_scores_for_folder(original_folder, pred_folder)
        # print(confidence_scores)
        # print(json.dumps(confidence_scores, indent=4))
        print(df)
        df.to_csv(os.getcwd() + "/"+ "YOLO/results/results.csv")
    else:
        if tmp_path == 'true':
                image_path = os.getcwd() + "/" + "DeepISP/patch_results/tmp/"
                save_dir = os.getcwd() + "/"+ "YOLO/results/tmp/"
        else:
                image_path = os.getcwd() + "/" + "DeepISP/patch_results/"
                save_dir = os.getcwd() + "/"+ "YOLO/results/"

        imgs = [image_path + name + "_" + input_file_name + ".png" for name in ["original_image", "patch_image"]]
        result_paths = [save_dir + name + "_" + input_file_name + ".png" for name in ["original_image", "patch_image"]]
        if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        results = model(imgs) # Predictions
        idx = 0
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            # print(result)
            result.show()  # display to screen
            result.save(filename=result_paths[idx])
            idx+=1

if __name__ == '__main__':
    main()