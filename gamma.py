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
import cv2
import numpy as np

from ex1_utils import LOAD_GRAY_SCALE


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    # Load the image
    img = cv2.imread(img_path)

    # Convert image to grayscale if rep is 1
    if rep == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define a function for gamma correction
    def adjust_gamma(gamma):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        corrected_img = cv2.LUT(img, table)
        cv2.imshow("Gamma Correction", corrected_img)

    # Create a window for displaying images
    cv2.namedWindow("Gamma Correction")

    # Create trackbar for adjusting gamma value
    cv2.createTrackbar("Gamma", "Gamma Correction", 100, 200, lambda x: adjust_gamma(x / 100))

    # Display original image
    cv2.imshow("Gamma Correction", img)

    # Wait for the user to adjust the trackbar
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
