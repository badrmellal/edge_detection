import cv2
import os
from PIL import Image
import numpy as np

def read_image_grayscale(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
    return np.array(Image.open(image_path).convert('L'))

def show_image(image, window_name):
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window_name, image)
    cv2.waitKey(1)

def plot_histogram(image, window_name):
    hist = cv2.calcHist([image], [0], None, [256], [0,256])
    hist_image = np.zeros((300, 256), dtype=np.uint8)
    cv2.normalize(hist, hist, 0, hist_image.shape[0], cv2.NORM_MINMAX)
    for x in range(256):
        cv2.line(hist_image, (x, hist_image.shape[0]), (x, hist_image.shape[0]-int(hist[x])), 255)
    cv2.imshow(f"Histogram of {window_name}", hist_image)
    cv2.waitKey(1)

if __name__ == "__main__":
    path = '/Users/Apple/PycharmProjects/Image_processing/REGLE.PCX'

    # Read and display original image
    img = read_image_grayscale(path)
    show_image(img, "Original Image")
    plot_histogram(img, "Original Image")

    # Duplicate image and display histogram
    img_copy = img.copy()
    plot_histogram(img_copy, "Duplicate Image")

    # Negative using 255 - I
    img_neg = 255 - img
    show_image(img_neg, "Negative Image (255 - I)")
    plot_histogram(img_neg, "Negative Image (255 - I)")

    # Negative using uniform image subtraction
    uniform = np.full_like(img, 255)
    img_neg2 = cv2.subtract(uniform, img)
    show_image(img_neg2, "Negative Image (Uniform - I)")
    plot_histogram(img_neg2, "Negative Image (Uniform - I)")

    # Histogram stretching to 0-255
    min_val = np.min(img_neg2)
    max_val = np.max(img_neg2)
    img_stretched = ((img_neg2 - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
    show_image(img_stretched, "Histogram Stretched")
    plot_histogram(img_stretched, "Histogram Stretched")

    print("Press ESC to close all windows...")
    while True:
        if cv2.waitKey(0) == 27:
            break
    cv2.destroyAllWindows()
