"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: grayscale(1) or RGB(2)
    :return: The image np array
    """

    # Read the image file
    image = cv2.imread(filename, cv2.IMREAD_COLOR if representation == 2 else cv2.IMREAD_GRAYSCALE)

    # Normalize the pixel values to the range [0, 1]
    image = image.astype(np.float32) / 255.0

    return image


def imDisplay(filename: str, representation: int) -> None:
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: grayscale(1) or RGB(2)
    :return: None
    """
    # Read and convert the image
    image = imReadAndConvert(filename, representation)

    # Display the image using matplotlib
    plt.figure()
    if representation == 1:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.axis('off')
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    # RGB to YIQ transformation matrix
    rgb2yiq = np.array([[0.299, 0.587, 0.114],
                        [0.595716, -0.274453, -0.321263],
                        [0.211456, -0.522591, 0.311135]])

    # Reshape the RGB image for matrix multiplication
    imgRGB_reshaped = imgRGB.reshape(-1, 3).T

    # Perform the transformation
    imYIQ = np.dot(rgb2yiq, imgRGB_reshaped).T.reshape(imgRGB.shape)

    return imYIQ


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    # YIQ to RGB transformation matrix
    yiq2rgb = np.linalg.inv(np.array([[0.299, 0.587, 0.114],
                                      [0.595716, -0.274453, -0.321263],
                                      [0.211456, -0.522591, 0.311135]]))

    # Reshape the YIQ image for matrix multiplication
    imgYIQ_reshaped = imgYIQ.reshape(-1, 3).T

    # Perform the transformation
    imRGB = np.dot(yiq2rgb, imgYIQ_reshaped).T.reshape(imgYIQ.shape)

    return imRGB


def histogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Equalizes the histogram of an image
    :param imgOrig: Original image
    :return: (imgEq,histOrg,histEQ)
    """

    # Check if the input image is RGB
    if len(imgOrig.shape) == 3 and imgOrig.shape[2] == 3:
        # Convert RGB image to YIQ color space
        imYIQ = transformRGB2YIQ(imgOrig)
        Y = imYIQ[:, :, 0]  # Extract the Y channel
    else:
        Y = imgOrig  # Use the input image as is

    # Normalize the Y channel values to [0, 255]
    Y_norm = (Y * 255).astype(np.uint8)

    # Calculate the histogram of the original image
    histOrg, _ = np.histogram(Y_norm.flatten(), bins=256, range=[0, 256])

    # Calculate the normalized Cumulative Sum (CumSum)
    cumsum = histOrg.cumsum()

    # Create a LookUpTable (LUT) for histogram equalization
    LUT = ((cumsum - cumsum.min()) * 255 / (cumsum.max() - cumsum.min())).astype(np.uint8)

    # Apply histogram equalization using the LookUpTable
    Y_eq = LUT[Y_norm.flatten()].reshape(Y_norm.shape)

    # Normalize the equalized image back to [0, 1]
    imEq = Y_eq / 255.0

    # Calculate the histogram of the equalized image
    histEq, _ = np.histogram(Y_eq.flatten(), bins=256, range=[0, 256])

    # If the input image was RGB, convert the equalized Y channel back to RGB
    if len(imgOrig.shape) == 3 and imgOrig.shape[2] == 3:
        imYIQ[:, :, 0] = imEq  # Update the Y channel with the equalized values
        imRGB = transformYIQ2RGB(imYIQ)  # Convert YIQ back to RGB
        imEq = np.float32(imRGB)  # Update the equalized image with RGB values

    return imEq, histOrg, histEq


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
    Quantizes a grayscale image into nQuant colors.

    :param imOrig: The original image (Grayscale)
    :param nQuant: Number of colors to quantize the image to
    :param nIter: Number of optimization loops
    :return: (List[qImage_i], List[error_i])
    """
    if len(imOrig.shape) == 3 and imOrig.shape[2] == 3:
        # Convert RGB to grayscale if necessary
        imOrig = np.dot(imOrig[..., :3], [0.2989, 0.5870, 0.1140])

    # Normalize the image
    imOrig = imOrig / 255.0

    pixels = imOrig.flatten().reshape(-1, 1)
    quantized_images = []
    errors = []

    # Perform K-Means clustering to find nQuant clusters in the data
    kmeans = KMeans(n_clusters=nQuant, max_iter=nIter, n_init=1)
    kmeans.fit(pixels)

    for i in range(nIter):
        # Assign pixels to their closest centroids
        labels = kmeans.predict(pixels)

        # Create quantized image based on the labels
        qImage = np.zeros_like(pixels)
        for j in range(nQuant):
            qImage[labels == j] = kmeans.cluster_centers_[j]

        # Calculate error as mean squared error between original and quantized image
        error = np.mean((pixels - qImage) ** 2)

        # Reshape quantized image back to original shape
        qImage = qImage.reshape(imOrig.shape)

        quantized_images.append(qImage * 255)  # Scale back to original range
        errors.append(error)

    return (quantized_images, errors)


