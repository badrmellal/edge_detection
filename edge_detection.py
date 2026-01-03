import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load image in grayscale
image = cv2.imread("/Users/Apple/PycharmProjects/Image_processing/IMG_7635.JPG", cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError("Image not found or unable to load")

# Display image
plt.figure()
plt.imshow(image, cmap='gray')
plt.title("Original Grayscale Image")
plt.axis("off")
plt.show()

# Apply Gaussian Blur
blurred_image = cv2.GaussianBlur(
    image,
    ksize=(5, 5),
    sigmaX=0
)

# Display blurred image
plt.figure()
plt.imshow(blurred_image, cmap='gray')
plt.title("Gaussian Blurred Image")
plt.axis("off")
plt.show()


# Sobel gradient
sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
# Edge magnitude
sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_magnitude = np.uint8(sobel_magnitude)

plt.figure()
plt.imshow(sobel_magnitude, cmap='gray')
plt.title("Sobel Edge Detection")
plt.axis("off")
plt.show()

# Prewitt kernels
prewitt_x = np.array([[ -1, 0,  1],[ -1, 0, 1],[ -1, 0, 1]])
prewitt_y = np.array([[ 1, 1, 1],[ 0, 0, 0],[ -1, -1, -1]])

# Apply filter
prewitt_x_img = cv2.filter2D(blurred_image, -1, prewitt_x)
prewitt_y_img = cv2.filter2D(blurred_image, -1, prewitt_y)

# Edge magnitude
prewitt_magnitude = np.sqrt(prewitt_x_img**2 + prewitt_y_img**2)
prewitt_magnitude = np.uint8(prewitt_magnitude)

plt.figure()
plt.imshow(prewitt_magnitude, cmap='gray')
plt.title("Prewitt Edge Detection")
plt.axis("off")
plt.show()


# Roberts kernels
roberts_x = np.array([[1, 0],
                      [0, -1]])

roberts_y = np.array([[0, 1],
                      [-1, 0]])

# Apply filters
roberts_x_img = cv2.filter2D(blurred_image, -1, roberts_x)
roberts_y_img = cv2.filter2D(blurred_image, -1, roberts_y)

# Edge magnitude
roberts_magnitude = np.sqrt(roberts_x_img**2 + roberts_y_img**2)
roberts_magnitude = np.uint8(roberts_magnitude)

plt.figure()
plt.imshow(roberts_magnitude, cmap='gray')
plt.title("Roberts Edge Detection")
plt.axis("off")
plt.show()

canny_low = cv2.Canny(blurred_image, 50, 100)
canny_high = cv2.Canny(blurred_image, 100, 200)

plt.figure()
plt.imshow(canny_low, cmap='gray')
plt.imshow(canny_high, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis("off")
plt.show()