import numpy as np
import cv2

from utilities import get_image_grayscale

kernel_A = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]]) / 9.0

kernel_B = np.array([[0, -1, 0],
                     [-1, 5, -1],
                     [0, -1, 0]])

kernel_C = np.array([[-1, -0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])

# Laplacian kernel
kernel_D = np.array([[0, 1, 0],
                     [1, -4, 1],
                     [0, 1, 0]])



def apply_convolution(img, kernel: np.ndarray):
    # -1 means the output image will have the same depth as the input image
    return cv2.filter2D(img, -1, kernel)

def task5(image: str):
    img = get_image_grayscale(image)
    img_A = apply_convolution(img, kernel_A)
    img_B = apply_convolution(img, kernel_B)
    img_C = apply_convolution(img, kernel_C)
    img_D = apply_convolution(img, kernel_D)
    return img_A, img_B, img_C, img_D
