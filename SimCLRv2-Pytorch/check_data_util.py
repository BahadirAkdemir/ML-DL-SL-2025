import numpy as np
import tensorflow as tf
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

import tf2.data_util as tf_preprocess  # TensorFlow version
import pytorch.data_util as torch_preprocess  # PyTorch version
import random

from PIL import Image



SEED = 23  # Use any integer

# Python
random.seed(SEED)

# NumPy
np.random.seed(SEED)

# PyTorch
torch.manual_seed(SEED)

# TensorFlow
tf.random.set_seed(SEED)


# === Config ===
HEIGHT, WIDTH = 224, 224
CROP_PROPORTION = 0.5
np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)

# === Helper functions ===

def compare_images(name, img1, img2, show=True, atol=1e-2):
    """Compare two [H, W, 3] images."""
    diff = np.abs(img1 - img2)
    print(f"\n{name} comparison:")
    print(f"  Max pixel diff  : {diff.max():.6f}")
    print(f"  Mean pixel diff : {diff.mean():.6f}")
    print(f"  Allclose        : {np.allclose(img1, img2, atol=atol)}")

    if show:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img1)
        axs[0].set_title("TensorFlow")
        axs[0].axis("off")
        axs[1].imshow(img2)
        axs[1].set_title("PyTorch")
        axs[1].axis("off")
        plt.suptitle(name)
        plt.show()

def prepare_inputs(image_size=256):
    #img = (np.random.rand(image_size, image_size, 3) * 255).astype(np.uint8)
    img = Image.open('dog.PNG').convert('RGB').resize((224, 224))
    tf_img = tf.convert_to_tensor(img, dtype=tf.uint8)
    torch_img = TF.to_tensor(img).float()  # Already [0,1]
    return tf_img, torch_img.permute(1, 2, 0)  # To HWC for PyTorch funcs

# === Test: center_crop ===
tf_img, torch_img = prepare_inputs()
tf_cropped = tf_preprocess.center_crop(tf.image.convert_image_dtype(tf_img, tf.float32), HEIGHT, WIDTH, CROP_PROPORTION).numpy()
torch_cropped = torch_preprocess.center_crop(torch_img.permute(2, 0, 1), HEIGHT, WIDTH, CROP_PROPORTION).permute(1, 2, 0).numpy()
compare_images("Center Crop", tf_cropped, torch_cropped)

# === Test: crop_and_resize ===
tf_img, torch_img = prepare_inputs()
tf_crop_resize = tf_preprocess.crop_and_resize(tf.image.convert_image_dtype(tf_img, tf.float32), HEIGHT, WIDTH).numpy()
torch_crop_resize = torch_preprocess.crop_and_resize(torch_img.permute(2, 0, 1), HEIGHT, WIDTH).permute(1, 2, 0).numpy()
compare_images("Crop and Resize", tf_crop_resize, torch_crop_resize)

# === Test: gaussian_blur ===
tf_img, torch_img = prepare_inputs()
sigma = 1.0
kernel_size = HEIGHT // 10

tf_blur = tf_preprocess.gaussian_blur(tf.image.convert_image_dtype(tf_img, tf.float32), kernel_size=kernel_size, sigma=sigma).numpy()
torch_blur = torch_preprocess.gaussian_blur(torch_img.permute(2, 0, 1), kernel_size=kernel_size, sigma=sigma).permute(1, 2, 0).numpy()
compare_images("Gaussian Blur", tf_blur, torch_blur)

# === Test: preprocess_image (eval only) ===
tf_img, torch_img = prepare_inputs()

# TensorFlow output: [H, W, C]
tf_eval = tf_preprocess.preprocess_image(
    tf_img, HEIGHT, WIDTH, is_training=False
).numpy()

# PyTorch output: [C, H, W] → convert to [H, W, C]
torch_eval = torch_preprocess.preprocess_image(
    torch_img.permute(2, 0, 1), HEIGHT, WIDTH, is_training=False
).numpy()

compare_images("Full Preprocess (Eval)", tf_eval, torch_eval)