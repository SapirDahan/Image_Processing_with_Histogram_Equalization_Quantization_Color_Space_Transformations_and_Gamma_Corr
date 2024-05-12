from ex1_utils import *
from gamma import gammaDisplay
import numpy as np
import matplotlib.pyplot as plt
import time


def histEqDemo(img_path: str, rep: int):
    img = imReadAndConvert(img_path, rep)

    # Call hsitogramEqualize and get the equalized image, original histogram, and equalized histogram
    imgeq, histOrg, histEq = histogramEqualize(img)

    # Normalize pixel values to be within [0, 1] for grayscale images
    img_norm = img
    imgeq_norm = imgeq

    # Display the original and equalized images
    plt.figure(figsize=(12, 6))

    # Plot the original image
    plt.subplot(1, 2, 1)
    print(img.shape, img.dtype)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap='gray')
    plt.title('Original Image')

    # Plot the equalized image
    imgeq_norm_display = np.clip(imgeq_norm, 0, 1)
    plt.subplot(1, 2, 2)


    plt.imshow(cv2.cvtColor(imgeq_norm_display, cv2.COLOR_BGR2RGB), cmap='gray')
    plt.title('Equalized Image')

    plt.show()

    # Plot the histograms of the original and equalized images
    plt.figure(figsize=(12, 6))

    # Plot the histogram of the original image
    plt.bar(np.arange(256), histOrg, color='b', alpha=0.7)

    # Calculate cumulative sum of histogram values
    cumsum_histOrg = np.cumsum(histOrg)

    # Scale the cumulative sum line to the height of the highest bar in the original histogram
    max_histOrg_value = max(histOrg)
    cumsum_histOrg_scaled = cumsum_histOrg * (max_histOrg_value / max(cumsum_histOrg))

    # Plot the cumulative sum line for the original histogram
    plt.plot(np.arange(256), cumsum_histOrg_scaled, color='orange', label='Cumulative Sum (Original)')

    plt.title('Histogram of Original Image')
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')

    plt.legend()

    plt.show()

    # Plot the histogram of the equalized image
    plt.figure(figsize=(12, 6))

    plt.bar(np.arange(256), histEq, color='r', alpha=0.7)


    # Calculate cumulative sum of histogram values for equalized image
    cumsum_histEq = np.cumsum(histEq)

    # Scale the cumulative sum line to the height of the highest bar in the equalized histogram
    max_histEq_value = max(histEq)
    cumsum_histEq_scaled = cumsum_histEq * (max_histEq_value / max(cumsum_histEq))

    # Plot the cumulative sum line for the equalized histogram
    plt.plot(np.arange(256), cumsum_histEq_scaled, color='orange', label='Cumulative Sum (Equalized)')

    plt.title('Histogram of Equalized Image')
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')

    plt.legend()

    plt.show()



def quantDemo(img_path: str, rep: int):
    imOrig = imReadAndConvert(img_path, 1)
    nQuant = 4
    nIter = 3

    imQuant, errors = quantizeImage(imOrig, nQuant, nIter)

    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(imOrig, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Plot the quantized image
    plt.subplot(1, 2, 2)
    plt.imshow(imQuant[-1].reshape(imOrig.shape), cmap='gray')
    plt.title('Quantized Image')
    plt.axis('off')

    plt.show()

    # Plot the final graph showing the progress of errors during quantization
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(errors)), errors, 'r.-')
    plt.title('Quantization Error Progress')
    plt.xlabel('Iteration')
    plt.ylabel('Error')

    plt.show()


def main():
    img_path = 'dark.jpg'

    # Display Original Image
    print("Displaying Original Image...")
    imDisplay(img_path, LOAD_RGB)

    # Convert Color Spaces and Display
    img = imReadAndConvert(img_path, LOAD_RGB)
    print("Converting Color Spaces and Displaying...")
    yiq_img = transformRGB2YIQ(img)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.title('Original RGB Image')
    plt.axis('off')

    # Normalize YIQ image for display
    yiq_img_display = np.clip(yiq_img, 0, 1)
    plt.subplot(1, 2, 2)
    plt.imshow(yiq_img_display)
    plt.title('Converted YIQ Image')
    plt.axis('off')
    plt.show()

    # Image Quantization and Display
    print("Performing Image Quantization...")
    quantDemo(img_path, LOAD_GRAY_SCALE)

    # Histogram Equalization and Display
    print("Performing Histogram Equalization...")
    histEqDemo(img_path, LOAD_RGB)

    # Gamma Correction Display
    print("Performing Gamma Correction...")
    gammaDisplay(img_path, LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()

