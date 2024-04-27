from ultralytics import YOLO
import os
import argparse
import json

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


    # results_df = pd.DataFrame.from_dict(final_data, orient='index', columns=['orig vs orig_pred', 'orig vs patched', 'orig_pred vs patched'])
    # print(results_df)
    return total_confidence_scores

def main():
    if process_folder == 'true':
        original_folder = current_path + orig_img_folder + "/" 
        pred_folder = current_path + "patch_results" + "/"
        confidence_scores = compute_confidence_scores_for_folder(original_folder, pred_folder)
        # print(confidence_scores)
        print(json.dumps(confidence_scores, indent=4))
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