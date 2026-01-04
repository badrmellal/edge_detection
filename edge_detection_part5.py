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

def log_transform(img):
    img = img.astype(np.float32)
    c = 255 / np.log(1 + np.max(img))
    log_img = c * np.log(1 + img)
    return np.uint8(log_img)

if __name__ == "__main__":

    # QUITO
    quito = read_gray("QUITO.PCX")
    show(quito, "Quito Original")
    histogram(quito, "Quito Original")

    quito_eq = cv2.equalizeHist(quito)
    show(quito_eq, "Quito Equalized")
    histogram(quito_eq, "Quito Equalized")

    quito_eq2 = cv2.equalizeHist(quito_eq)
    show(quito_eq2, "Quito Equalized Twice")
    histogram(quito_eq2, "Quito Equalized Twice")

    # LENA
    lena = read_gray("LENA.PCX")
    show(lena, "Lena Original")
    histogram(lena, "Lena Original")

    lena_eq = cv2.equalizeHist(lena)
    show(lena_eq, "Lena Equalized")
    histogram(lena_eq, "Lena Equalized")

    # BOUGIES
    bougies = read_gray("BOUGIES.PCX")
    show(bougies, "Bougies Original")
    histogram(bougies, "Bougies Original")

    bougies_stretch = stretch_full(bougies)
    show(bougies_stretch, "Bougies Stretched")
    histogram(bougies_stretch, "Bougies Stretched")

    bougies_log = log_transform(bougies)
    show(bougies_log, "Bougies Log Transform")
    histogram(bougies_log, "Bougies Log Transform")

    while True:
        if cv2.waitKey(0) == 27:
            break
    cv2.destroyAllWindows()
