import cv2
import matplotlib.pyplot as plt
import numpy as np

def process_grayscale_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Gray Image", gray_image)
    # Calculate histogram
    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # plot histogram (line plot
    plt.figure()
    plt.plot(histogram)
    plt.title("Grayscale Histogram (Line Plot)")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()

    # plot histogram (bar chart)
    plt.figure()
    plt.bar(range(256), histogram.flatten(), width=1)
    plt.title("Grayscale Histogram (Bar Chart)")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()

def process_color_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")

    # Split the image into its color channels
    b, g, r = cv2.split(image)
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

    # Plot histograms for each channel
    plt.figure()
    plt.plot(hist_b, color='blue', label='Blue Channel')
    plt.plot(hist_g, color='green', label='Green Channel')
    plt.plot(hist_r, color='red', label='Red Channel')
    plt.title("Color Histogram (Line Plot)")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


def cumulative_histogram(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # Calculate cumulative histogram
    cumulative_hist = np.cumsum(hist)

    # Normalize
    cumulative_hist_norm = cv2.normalize(
        cumulative_hist, None, 0, 255, cv2.NORM_MINMAX
    )

    # Plot cumulative histogram
    plt.figure()
    plt.plot(cumulative_hist_norm, color='black')
    plt.title("Cumulative Grayscale Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Cumulative Frequency (Normalized)")
    plt.show()

if __name__ == "__main__":
    cumulative_histogram("/Users/Apple/PycharmProjects/Image_processing/IMG_7635.JPG")