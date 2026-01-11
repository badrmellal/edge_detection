import numpy as np
import matplotlib.pyplot as plt
from skimage import data, util
from scipy.ndimage import convolve
from scipy.signal import convolve2d
from scipy.ndimage import median_filter
from skimage.io import imread

def main():
    image = data.camera()

    # Add salt-and-pepper noise
    noisy_image = util.random_noise(image, mode='s&p', amount=0.05)

    # Display the noisy image
    plt.imshow(noisy_image, cmap='gray')
    plt.title("Cameraman with Salt & Pepper Noise")
    plt.axis('off')
    plt.show()


def linear_filter():
    # Define filters
    w1 = np.ones((3, 3)) / 9
    w2 = np.array([[1, 1, 1],
                   [1, 2, 1],
                   [1, 1, 1]]) / 10
    w3 = np.array([[1, 2, 1],
                   [2, 4, 2],
                   [1, 2, 1]]) / 16
    w4 = np.ones((5, 5)) / 25

    filters = [w1, w2, w3, w4]

    # Load image
    img = data.camera()

    # Apply filters
    filtered_images = []
    for w in filters:
        filtered = convolve(img, w, mode='reflect')
        filtered_images.append(filtered)

    # Display results
    titles = ['w1', 'w2', 'w3', 'w4']

    img = imread('objects.pcx', as_gray=True)

    filtered = img.copy()
    for i in range(5):  # n times
        filtered = convolve(filtered, w3, mode='reflect')

    plt.imshow(filtered, cmap='gray')
    plt.title("w3 applied 5 times")
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(10, 8))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(filtered_images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.show()


def non_linear_filter():
    # Original image
    image = data.camera()

    # Add salt-and-pepper noise
    noisy_image = util.random_noise(image, mode='s&p', amount=0.05)

    # Apply 3x3 median filter
    img_median = median_filter(noisy_image, size=3)

    # Display results
    plt.figure(figsize=(10,4))

    plt.subplot(1,3,1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Cameraman")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(noisy_image, cmap='gray')
    plt.title("Cameraman1 (Noisy)")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(img_median, cmap='gray')
    plt.title("Median Filter (3x3)")
    plt.axis('off')

    plt.show()


def laplacian_filter():
    w5  = np.array([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]])

    w5p = np.array([[0, -1, 0],
                    [1,  4, -1],
                    [0, -1, 0]])

    w6  = np.array([[1, 1, 1],
                    [1, -8, 1],
                    [1, 1, 1]])

    w6p = np.array([[1, -1, -1],
                    [1,  8, -1],
                    [1, -1, -1]])

    laplacian_filters = [w5, w5p, w6, w6p]
    titles = ['w5', "w5'", 'w6', "w6'"]

    # Load image
    img = imread('objects.pcx', as_gray=True)

    plt.figure(figsize=(10,8))
    for i, w in enumerate(laplacian_filters):
        edge = convolve(img, w, mode='reflect')
        plt.subplot(2,2,i+1)
        plt.imshow(edge, cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.suptitle("Laplacian Edge Detection")
    plt.show()

def sobel_filter():
    ws1 = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]])

    ws2 = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])

    img = imread('cameraman.bmp', as_gray=True)

    # Apply Sobel filters
    sobel_x = convolve(img, ws1, mode='reflect')
    sobel_y = convolve(img, ws2, mode='reflect')

    # Gradient magnitude
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(sobel_x, cmap='gray')
    plt.title("Sobel Vertical (ws1)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(sobel_y, cmap='gray')
    plt.title("Sobel Horizontal (ws2)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(sobel_mag, cmap='gray')
    plt.title("Sobel Gradient Magnitude")
    plt.axis('off')

    plt.show()


if __name__ == "__main__":
    sobel_filter()
