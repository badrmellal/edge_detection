import cv2
import os
from PIL import Image
import numpy as np

def read_gray(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.array(Image.open(path).convert('L'))

def show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(1)

def histogram(img, name):
    hist = cv2.calcHist([img], [0], None, [256], [0,256])
    h_img = np.zeros((300,256), dtype=np.uint8)
    cv2.normalize(hist, hist, 0, 300, cv2.NORM_MINMAX)
    for x in range(256):
        cv2.line(h_img, (x,300), (x,300-int(hist[x])), 255)
    cv2.imshow(f"Histogram {name}", h_img)
    cv2.waitKey(1)

def stretch_full(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

def stretch_range(img, a, b):
    min_i, max_i = img.min(), img.max()
    stretched = (img - min_i) * (b - a) / (max_i - min_i) + a
    return np.clip(stretched, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    path = "/Users/Apple/PycharmProjects/Image_processing/QUITO.PCX"

    img = read_gray(path)
    img1 = img.copy()
    img2 = img.copy()
    img3 = img.copy()

    show(img, "Original")
    histogram(img, "Original")

    full = stretch_full(img)
    show(full, "Full Stretch 0–255")
    histogram(full, "Full Stretch")

    r1 = stretch_range(img1, 50, 100)
    r2 = stretch_range(img2, 40, 120)
    r3 = stretch_range(img3, 35, 166)

    show(r1, "Stretch 50–100")
    show(r2, "Stretch 40–120")
    show(r3, "Stretch 35–166")

    histogram(r1, "50–100")
    histogram(r2, "40–120")
    histogram(r3, "35–166")

    while True:
        if cv2.waitKey(0) == 27:
            break
    cv2.destroyAllWindows()
