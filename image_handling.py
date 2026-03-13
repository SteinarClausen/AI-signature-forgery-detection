import cv2
import numpy as np
import matplotlib.pyplot as plt

from utilities import get_image_binary


def rotate_image(img: np.ndarray, theta_deg: float, binary: bool = True) -> np.ndarray:
    """Rotate the image but keep the data without over-cropping."""
    height, width = img.shape
    center = (width / 2.0, height / 2.0)
    
    M = cv2.getRotationMatrix2D(center, theta_deg, 1.0)
    
    # Calculate cos and sin to find new dimensions
    cos_theta = np.abs(M[0, 0])
    sin_theta = np.abs(M[0, 1])
    
    # New dimensions
    new_width = int((height * sin_theta) + (width * cos_theta))
    new_height = int((height * cos_theta) + (width * sin_theta))
    
    # Move the image to the new centre
    M[0, 2] += (new_width / 2.0) - center[0]
    M[1, 2] += (new_height / 2.0) - center[1]
    
    interp_flag = cv2.INTER_NEAREST if binary else cv2.INTER_LINEAR
    
    rotated = cv2.warpAffine(
        img,
        M,
        (new_width, new_height),
        flags=interp_flag,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    
    return rotated


def cropped_height(binary_img: np.ndarray) -> int:
    y_values, _ = np.where(binary_img > 0)
    if y_values.size == 0:
        return 0
    return int(y_values.max() - y_values.min() + 1)


def rotation_height(binary_img: np.ndarray, theta_deg: float) -> int:
    return cropped_height(rotate_image(binary_img, theta_deg))


def finite_difference_gradient(binary_img: np.ndarray, theta_deg: float, h: float):
    """Central Finite difference function of df/dtheta"""
    return (rotation_height(binary_img, theta_deg + h) - rotation_height(binary_img, theta_deg - h)) / (2.0 * h)


def minimize_height_finite_difference(binary_img: np.ndarray,theta0: float = 0.0, h0: float = 1.0, max_iter: int = 80):
    theta = float(theta0)
    h = float(h0)
    lr = 0.8

    # Initilize the values
    best_theta = theta
    best_val = rotation_height(binary_img, theta)
    history = [(0, theta, best_val, h, 0.0)]

    for k in range(1, max_iter + 1):
        grad = finite_difference_gradient(binary_img, theta, h)

        if abs(grad) < 1e-10: # Threshold value
            neighborhood = [theta - h, theta, theta + h]
            vals = [rotation_height(binary_img, t) for t in neighborhood]
            i = int(np.argmin(vals))
            if vals[i] < best_val:
                theta = neighborhood[i]
                best_theta, best_val = theta, vals[i]
            h *= 0.7
            history.append((k, theta, rotation_height(binary_img, theta), h, grad))
            continue

        cand = float(np.clip(theta - lr * grad, -45.0, 45.0))
        f_theta = rotation_height(binary_img, theta)
        f_cand = rotation_height(binary_img, cand)

        if f_cand <= f_theta:
            theta = cand
            lr *= 1.05
            if f_cand < best_val:
                best_theta, best_val = theta, f_cand
        else:
            lr *= 0.5
            h *= 0.8

        history.append((k, theta, rotation_height(binary_img, theta), h, grad))

    return best_theta, best_val, history


def get_crop_dimensions(binary_img: np.ndarray) -> tuple[int, int, int, int]:
    """Gives the crop corners. MUST BE A BINARY IMAGE INPUT"""
    y_values, x_values = np.where(binary_img > 0)
    if y_values.size == 0 or x_values.size == 0:
        raise RuntimeError("No pixels found in image.")
    
    y0, y1 = y_values.min(), y_values.max() + 1
    x0, x1 = x_values.min(), x_values.max() + 1
    return y0, y1, x0, x1


def crop_image(img: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    return img[y0:y1, x0:x1]


def task7(image: str):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image}")
    binary_img = get_image_binary(img)

    # Run optimization using finite difference 
    theta_opt, h_opt, hist = minimize_height_finite_difference(binary_img, theta0=0.0, h0=1.0, max_iter=120)
    # Reference: global sweep for visualization/comparison
    angles = np.linspace(-45, 45, 901)
    heights = np.array([rotation_height(binary_img, a) for a in angles])
    theta_grid = float(angles[np.argmin(heights)])
    h_grid = int(heights.min())

    rot_opt = rotate_image(binary_img, theta_opt)

    orig_h = rotation_height(binary_img, 0.0)
    opt_h = rotation_height(binary_img, theta_opt)


    # Visualize original and optimally aligned images
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].imshow(binary_img, cmap="gray")
    ax[0].set_title(f"Original (θ=0°)\nHeight={orig_h}px")
    ax[0].axis("off")

    ax[1].imshow(rot_opt, cmap="gray")
    ax[1].set_title(f"Finite-difference aligned\nθ={theta_opt:.3f}°, Height={opt_h}px")
    ax[1].axis("off")

    ax[2].plot(angles, heights, lw=2)
    ax[2].axvline(theta_opt, color="red", ls="--", label="FD optimum")
    ax[2].set_xlabel("Rotation angle θ (degrees)")
    ax[2].set_ylabel("Cropped height (pixels)")
    ax[2].set_title("Objective function f(θ)")
    ax[2].legend()

    plt.tight_layout()
    plt.show()

    y0, y1, x0, x1 = get_crop_dimensions(rot_opt)

    rot_gray = rotate_image(img, theta_opt, False)
    rot_gray_cropped = rot_gray[y0:y1, x0:x1]

    mask = (rot_opt > 0)
    mask_cropped = (mask[y0:y1, x0:x1] * 255).astype(np.uint8)

    # Keep only the text and not the background.
    rot_gray_cropped_masked = cv2.bitwise_and(
        rot_gray_cropped,
        rot_gray_cropped,
        mask=mask_cropped,
    )

    plt.imshow(rot_gray_cropped_masked, cmap="gray")
    plt.show()
