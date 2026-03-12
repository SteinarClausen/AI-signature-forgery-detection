from utilities import threshold_image, get_image_grayscale, display_img

def task6(image: str):
    img = get_image_grayscale(image)
    thresholded_img = threshold_image(img, 200)
    display_img(img)
    display_img(thresholded_img)
