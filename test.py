from ex1_utils import *
from gamma import gammaDisplay
import numpy as np
import matplotlib.pyplot as plt
import time


imOrig = imReadAndConvert('dark.jpg', 1)
nQuant = 4
nIter = 10

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
