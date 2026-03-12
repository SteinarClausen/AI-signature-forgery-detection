import cv2
import matplotlib.pyplot as plt

def show_image_histogram(image: str, bins: int = 256):
    img = cv2.imread(image)
    plt.hist(img.ravel(), bins, [0, 256])
    plt.title('Histogram for ' + image)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()
