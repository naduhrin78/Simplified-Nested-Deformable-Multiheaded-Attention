import numpy as np
from PIL import Image
import cv2
import torch
import torchvision


class BinarizeMask(object):
    def __init__(self, threshold=0.6):
        self.threshold = threshold

    def __call__(self, mask):
        mask_array = np.array(mask) / 255.0
        binarized_mask = (mask_array > self.threshold).astype(np.uint8) * 255
        return Image.fromarray(binarized_mask)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert("RGB")
    img = img.resize((256, 256), Image.BICUBIC)
    return img


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    cv2.imwrite(filename, image_numpy)
    print("Image saved as {}".format(filename), end="\r")
