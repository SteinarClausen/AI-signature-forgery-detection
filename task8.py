import numpy as np
import matplotlib.pyplot as plt
import cv2

from utilities import get_image_grayscale, display_img

def task8a(image: str):
    img = get_image_grayscale(image)
    img_array = np.array(img)
    fourier_transform = np.fft.fft2(img_array)
    fourier_shifted = np.fft.fftshift(fourier_transform)
    magnitude_spectrum = np.log(np.abs(fourier_shifted) + 1)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum')
    plt.xlabel('Frequency')
    plt.ylabel('Frequency')
    plt.show()

def task8b(image: str):
    # frequency domain low-pass filtering
    img = get_image_grayscale(image)
    img_array = np.array(img)
    fourier_transform = np.fft.fft2(img_array)
    fourier_shifted = np.fft.fftshift(fourier_transform)
    rows, cols = img_array.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.uint8)
    r = 30  # radius of the low-pass filter
    cv2.circle(mask, (ccol, crow), r, 1, thickness=-1)
    fourier_shifted_filtered = fourier_shifted * mask
    fourier_inverse_shifted = np.fft.ifftshift(fourier_shifted_filtered)
    filtered_magnitude_spectrum = np.log1p(np.abs(fourier_shifted_filtered))
    img_back = np.fft.ifft2(fourier_inverse_shifted)
    img_back = np.abs(img_back)
    plt.imshow(filtered_magnitude_spectrum, cmap='gray')
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency')
    plt.ylabel('Frequency')
    plt.show()
    plt.imshow(img_back, cmap='gray')
    plt.title('Low-pass Filtered Image')
    plt.xlabel('Pixel')
    plt.ylabel('Pixel')
    plt.show()

    display_img(img)
    display_img(img_back)

