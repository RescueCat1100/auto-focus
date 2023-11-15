import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from skimage.draw import disk
import cv2
from matplotlib import pyplot as plt
defocusKernelDims = [5]


def DefocusBlur_random(img):
    kernelidx = np.random.randint(0, len(defocusKernelDims))
    kerneldim = defocusKernelDims[kernelidx]
    return DefocusBlur(img, kerneldim)


def DefocusBlur(img, dim):
    imgarray = np.array(img, dtype="float32")
    kernel = DiskKernel(dim)
    for i in range(imgarray.shape[-1]):
        # frame[:,:,i] = convolve2d(frame[:,:,i], psf, mode="same")
        convolved = convolve2d(imgarray[:, :, i], kernel, mode='same',
                               fillvalue=255.0).astype("uint8")
        img = Image.fromarray(convolved)
    return img


def DiskKernel(dim):
    kernelwidth = dim
    kernel = np.zeros((kernelwidth, kernelwidth), dtype=np.float32)
    circleCenterCoord = dim / 2
    circleRadius = circleCenterCoord + 1
    rr, cc = disk((circleCenterCoord, circleCenterCoord),
                  circleRadius, shape=kernel.shape)
    plt.imshow(disk((circleCenterCoord, circleCenterCoord),
                    circleRadius, shape=kernel.shape))
    plt.show()
    print(rr)
    print(cc)
    print(kernel)
    kernel[rr, cc] = 1
    print(kernel)
    if (dim == 3 or dim == 5):
        kernel = Adjust(kernel, dim)

    normalizationFactor = np.count_nonzero(kernel)
    print(normalizationFactor)
    kernel = kernel / normalizationFactor
    print(kernel)
    return kernel


def Adjust(kernel, kernelwidth):
    kernel[0, 0] = 0
    kernel[0, kernelwidth-1] = 0
    kernel[kernelwidth-1, 0] = 0
    kernel[kernelwidth-1, kernelwidth-1] = 0
    return kernel


img = cv2.imread('img.jpg')
img = DefocusBlur_random(img)
plt.imshow(img, cmap='gray')
plt.show()
