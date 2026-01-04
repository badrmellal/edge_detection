import cv2
import os
from PIL import Image
import numpy as np

def image_histograms(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    window_name = os.path.basename(image_path)
    image = cv2.cvtColor(np.array(Image.open(image_path).convert('RGB')), cv2.COLOR_RGB2BGR)
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window_name, image)
    cv2.waitKey(1)

    # Create histogram image
    hist_image = np.zeros((300, 256, 3), dtype=np.uint8)
    colors = [(255,0,0), (0,255,0), (0,0,255)]  # B, G, R channels

    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0,256])
        cv2.normalize(hist, hist, 0, hist_image.shape[0], cv2.NORM_MINMAX)
        for x in range(256):
            cv2.line(hist_image, (x, hist_image.shape[0]), (x, hist_image.shape[0] - int(hist[x])), color)

    cv2.imshow(f"Histogram of {window_name}", hist_image)
    return image

def read_image_grayscale(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
    pil_image = Image.open(image_path).convert('L')
    image = np.array(pil_image)
    return image

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

def add_images(img1, img2):
    added = cv2.add(img1, img2)
    return added

def subtract_images(img1, img2):
    subtracted = cv2.subtract(img1, img2)
    return subtracted

if __name__ == '__main__':
    path1= '/Users/Apple/PycharmProjects/Image_processing/RONDELLE.PCX'
    path2= '/Users/Apple/PycharmProjects/Image_processing/SPOT.PCX'

    img1 = read_image_grayscale(path1)
    img2 = read_image_grayscale(path2)

    # Display original images
    show_image(img1, "Rondelle")
    show_image(img2, "Spot")

    # Plot original histograms
    plot_histogram(img1, "Rondelle")
    plot_histogram(img2, "Spot")

    # Image addition
    img_add = add_images(img1, img2)
    show_image(img_add, "Addition")
    plot_histogram(img_add, "Addition")

    # Image subtraction
    img_sub = subtract_images(img1, img2)
    show_image(img_sub, "Subtraction")
    plot_histogram(img_sub, "Subtraction")

    print("Press ESC to close all windows")
    while True:
        if cv2.waitKey(0) == 27:  # ESC key
            break
    cv2.destroyAllWindows()
