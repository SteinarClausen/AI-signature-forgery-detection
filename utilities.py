import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

def display_img(image):
    fig, ax = plt.subplots(1, 2, figsize=(5, 2.5))
    ax[0].imshow(image, cmap='gray')
    ax[0].axis('off')
    # ax[0].set_title('Image')
    ax[1].hist(image.ravel(), bins=256  , range=(0, 256))
    # ax[1].set_title('Histogram')
    ax[1].set_xlabel('Pixel Intensity')
    ax[1].set_ylabel('Frequency (log scale)')
    ax[1].set_yscale('log')
    fig.tight_layout()
    fig.savefig("figures/pixel_intensity_image")
    plt.show()
    plt.close()

def get_image_grayscale(image: str):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    return img

def threshold_image(image: np.ndarray, threshold: int):
    _, threshold_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return threshold_image

def get_image_binary(img: np.ndarray) -> np.ndarray:
    _, binary_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.count_nonzero(binary_image == 255) > np.count_nonzero(binary_image == 0):
        binary_image = cv2.bitwise_not(binary_image)
    return binary_image

def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))

def contrastive_loss(y_true, y_pred, margin=1.0):
    # y_true is 0 for real pairs, 1 for fakes
    # y_pred is the distance D
    y_true = tf.cast(y_true, tf.float32)
    
    loss_ekte = (1.0 - y_true) * tf.square(y_pred)
    loss_falsk = y_true * tf.square(tf.maximum(0.0, margin - y_pred))
    
    return tf.reduce_mean(loss_ekte + loss_falsk)


def pair_accuracy(y_true, y_pred):
    """Regner ut nøyaktighet basert på en Euklidsk avstandsterskel på 0.5"""
    threshold = 0.5
    
    y_true = tf.reshape(y_true, [-1, 1])
    y_true = tf.cast(y_true, tf.float32)
    
    # Guess 1 (fake) if distance > threshold, if not 0 (real)
    predictions = tf.cast(tf.math.greater(y_pred, threshold), tf.float32)
    
    correct = tf.cast(tf.math.equal(predictions, y_true), tf.float32)
    return tf.reduce_mean(correct)