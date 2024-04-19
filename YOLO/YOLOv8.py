from ultralytics import YOLO
import os
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('-in' ,'--input_file_name', type = str, default = '0' , help = 'input image file name')
args = parser.parse_args()
input_file_name = args.input_file_name

model = YOLO('yolov8n.pt')  # load a pretrained YOLOv8n detection model
# model.train(data='coco128.yaml', epochs=3)  # Uncomment to train the model

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
    result.show()  # display to screen
    result.save(filename=result_paths[idx])
    idx+=1