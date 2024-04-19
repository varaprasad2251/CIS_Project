import numpy as np
from load_data import extract_bayer_channels
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.applications.vgg16 import VGG16
import argparse
import os
from network import network
import imageio.v2 as imageio
from load_data import load_testing_inp
import time


PATCH_HEIGHT, PATCH_WIDTH = 224, 224

parser = argparse.ArgumentParser()

parser.add_argument('-w' ,'--weights_file', type = str, default = 'weights2_0191.h5' , help = 'best weight file name (only prefix while evaluating)')
parser.add_argument('-dataset' ,'--dataset_path', type = str, default = './DeepISP/input_raw_images/' , help = 'complete path for the dataset')
parser.add_argument('-path' ,'--main_path', type = str, default = './DeepISP/' , help = 'main path where the result/experiment folders are stored')
parser.add_argument('-res' ,'--results_folder', type = str, default = 'patch_results' , help = 'folder to save patch results')
parser.add_argument('-in' ,'--input_file_name', type = str, default = '0' , help = 'input image file name')

args = parser.parse_args()
weights_file = args.weights_file
dataset_dir = args.dataset_path
current_path = args.main_path
res_folder = args.results_folder
input_file_name = args.input_file_name

def clip_eps(tensor, eps):
	# clip the values of the tensor to a given range and return it
	return tf.clip_by_value(tensor, clip_value_min=-eps,
		clip_value_max=eps)

# Load and preprocess the image
def load_image(image_path):
    I = np.asarray(imageio.imread(image_path))
    # Bayer channel extraction
    I = extract_bayer_channels(I)
    if I.shape[0] != PATCH_HEIGHT or I.shape[1] != PATCH_WIDTH:
        raise ValueError("Extracted patch does not match specified dimensions: ({}, {}).".format(PATCH_WIDTH, PATCH_HEIGHT))
    raw_img = np.zeros((1, PATCH_HEIGHT, PATCH_WIDTH, 4))
    raw_img[0, :] = I
    return raw_img

def save_image(patched_image, file_path):
    patched_image = np.uint8(patched_image * 255.0)  # Convert from [0, 1] to [0, 255]
    imageio.imwrite(file_path, patched_image)

# To Load the DeepISP model with pre-trained weights
def load_model():
    in_shape = (224,224,4)
    base_vgg = VGG16(weights = 'imagenet', include_top = False, input_shape = (448,448,3))
    vgg = Model(inputs = base_vgg.input, outputs = base_vgg.get_layer('block4_pool').output)
    for layer in vgg.layers:
        layer.trainable = False
    d_model = network(vgg, inp_shape = in_shape, trainable = False)
    filename = os.path.join(current_path, weights_file)
    d_model.load_weights(filename)
    return d_model

def evaluate_ssim_impact(original_image, patched_image, size=10, save_path='./DeepISP/masked_images/'):
    # Assuming original_image and patched_image are already preprocessed and in the correct format
    # Convert to float32 for TensorFlow operations
    original_image = tf.cast(original_image, tf.float32)
    patched_image = tf.cast(patched_image, tf.float32)
    
    image_height, image_width = original_image.shape[1], original_image.shape[2]
    patch_ratio = (size ** 0.5) / 10
    patch_size = (int(image_height * patch_ratio), int(image_width * patch_ratio))

    top_left_x = (image_width - patch_size[0]) // 2
    top_left_y = (image_height - patch_size[1]) // 2
    
    # Create a white patch
    white_patch = np.zeros((patch_size[1], patch_size[0], 3), dtype=np.uint8) * 255
    
    # Overlay the white patch on the original image
    original_image_with_white_patch = original_image[0].numpy()
    original_image_with_white_patch[top_left_y:top_left_y + patch_size[1], top_left_x:top_left_x + patch_size[0], :] = white_patch
    
    # Overlay the white patch on the patched image
    patched_image_with_white_patch = patched_image[0].numpy()
    patched_image_with_white_patch[top_left_y:top_left_y + patch_size[1], top_left_x:top_left_x + patch_size[0], :] = white_patch
    
    # Convert the images to uint8 and ensure they are in a format that imageio can handle
    original_image_with_white_patch_uint8 = np.clip(original_image_with_white_patch, 0, 255).astype(np.uint8)
    patched_image_with_white_patch_uint8 = np.clip(patched_image_with_white_patch, 0, 255).astype(np.uint8)
    
    # Save the original image with the white patch and the patched image with the white patch
    imageio.imwrite(save_path + 'original_with_white_patch.png', original_image_with_white_patch_uint8)
    imageio.imwrite(save_path + 'patched_image_with_white_patch.png', patched_image_with_white_patch_uint8)
    
    # Calculate SSIM for the entire image
    total_ssim = tf.reduce_mean(tf.image.ssim(original_image[0], patched_image[0], max_val=1.0)).numpy()
    
    # Calculate SSIM for the non-patched areas
    non_patch_ssim = tf.reduce_mean(tf.image.ssim(original_image_with_white_patch_uint8, patched_image_with_white_patch_uint8, max_val=1.0)).numpy()

    print("The SSIM score between original and patch image is {0}".format(total_ssim))
    print("SSIM score of the image without the patch area is {0}".format(non_patch_ssim))
    
    return total_ssim, non_patch_ssim

# FGSM attack to optimize the patch
def fgsm_patch(image, model, epsilon, max_iterations, loss_threshold, size=10):
    original_image,_,_,_,_ = model.predict(image)
    
    image_height, image_width = image.shape[1], image.shape[2]
    patch_ratio = (size ** 0.5) / 10
    patch_size = (int(image_height * patch_ratio), int(image_width * patch_ratio)) # To adjust Patch Size

    # Patch Location - Centre of Image
    top_left_x = (image_width - patch_size[0]) // 2
    top_left_y = (image_height - patch_size[1]) // 2
    image = tf.cast(image, tf.float32)

    # patch = tf.Variable(tf.random.uniform([1, patch_size[1], patch_size[0], 4], dtype=tf.float32, minval=0, maxval=1))
    patch = tf.Variable(image[:, top_left_y:top_left_y + patch_size[1], top_left_x:top_left_x + patch_size[0], :], dtype=tf.float32)
    perturbation = tf.random.uniform(patch.shape, minval=-epsilon, maxval=epsilon, dtype=tf.float32)
    patch.assign(tf.clip_by_value(patch + perturbation, 0, 1))
    max_loss = float('-inf')
    best_patched_image = tf.identity(original_image)

    for i in range(max_iterations):
        with tf.GradientTape() as tape:
            tape.watch(patch)
            patched_image = tf.identity(image)
            indices = [[0, y, x, c] for y in range(top_left_y, top_left_y + patch_size[1])
                                    for x in range(top_left_x, top_left_x + patch_size[0])
                                    for c in range(4)]
            patch_values = tf.reshape(patch, [-1])
            patched_image = tf.tensor_scatter_nd_update(patched_image, indices, patch_values)
            output_with_patch,_, _, _, _ = model(patched_image, training=False)
            loss = 1 - tf.reduce_mean(tf.image.ssim(original_image[0], output_with_patch[0], max_val=1.0))
            print(f"Current Iteration : {i+1}, " , "SSIM Score : ", tf.image.ssim(original_image[0], output_with_patch[0], max_val=1.0).numpy(), ", Loss : ", loss.numpy())
        gradients = tape.gradient(loss, patch)
        patch.assign_add(epsilon * tf.sign(gradients))
        patch.assign(tf.clip_by_value(patch, 0, 1))
        
        if loss.numpy() > loss_threshold:
            print(f"Exceeded Loss Threshold ----- Exiting the Loop")
            break

        if loss.numpy() > max_loss:
            max_loss = loss.numpy()
            best_patched_image = tf.identity(output_with_patch)
    
    evaluate_ssim_impact(np.uint8(original_image*255.0),np.uint8(best_patched_image*255.0))

    return np.uint8(original_image*255.0), np.uint8(best_patched_image*255.0)


def process_raw_images(model):
    raw_imgs = load_testing_inp(dataset_dir, 224, 224)
    t1=time.time()
    out,_,_,_,_ = model.predict(raw_imgs)
    print(out.shape)
    t2=time.time()
    t = (t2-t1)/raw_imgs.shape[0]
    print(t)
    for i in range(out.shape[0]):
        I = np.uint8(out[i,:,:,:] * 255.0)
        imageio.imwrite(os.path.join(current_path, res_folder) + '/' +  str(i) + '.png', I)

def main():
    epsilon = 0.05 / 255.0  # Perturbation level
    max_iterations = 20 # Max Iterations for FGSM
    loss_threshold = 0.02 # Loss Threshold

    image_path = current_path + "input_raw_images/" + input_file_name + ".png"
    image = load_image(image_path)
    
    model = load_model() # Load DeepISP model

    original_image, patched_image = fgsm_patch(image, model, epsilon, max_iterations, loss_threshold, 10)

    output_path = os.path.join(current_path, res_folder) + '/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    save_image(patched_image[0,:,:,:] * 255.0, output_path + "patch_image_" + input_file_name + ".png")
    save_image(original_image[0,:,:,:] * 255.0, output_path + "original_image_" + input_file_name + ".png")
    print("Optimized patch generated and saved.")

if __name__ == "__main__":
    main()
