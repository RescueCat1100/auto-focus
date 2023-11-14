import numpy as np
import matplotlib.image as pltim
import matplotlib.pyplot as plt
from PIL import Image


def Convolution(image: np.ndarray, kernel) -> np.ndarray:
    if len(image.shape) == 3:
        m_i, n_i, c_i = image.shape

    # if the image is gray then we won't be having an extra channel so handling it
    elif len(image.shape) == 2:
        image = image[..., np.newaxis]
        m_i, n_i, c_i = image.shape
    else:
        raise Exception('Shape of image not supported')

    m_k, n_k = kernel.shape

    y_strides = m_i - m_k + 1  # possible number of strides in y direction
    x_strides = n_i - n_k + 1  # possible number of strides in x direction

    img = image.copy()
    output_shape = (m_i-m_k+1, n_i-n_k+1, c_i)
    output = np.zeros(output_shape, dtype=np.float32)

    count = 0  # taking count of the convolution operation being happening

    output_tmp = output.reshape(
        (output_shape[0]*output_shape[1], output_shape[2])
    )

    for i in range(y_strides):
        for j in range(x_strides):
            for c in range(c_i):
                sub_matrix = img[i:i+m_k, j:j+n_k, c]

                output_tmp[count, c] = np.sum(sub_matrix * kernel)

            count += 1

    output = output_tmp.reshape(output_shape)

    return output


def Gray(image):
    gray_img = np.dot(image, [0.299, 0.587, 0.114]).astype(np.uint8)
    # plt.imshow(gray_img, cmap='gray', vmin=0, vmax=255)
    # plt.title("Gray Scale")
    # plt.show()
    return gray_img


def Sobel(image, filter):
    sobel_img = Convolution(image, filter)
    # plt.imshow(sobel_img, cmap='gray')
    # plt.show()
    return sobel_img


def GaussianBlur(img: np.ndarray, sigma, filter_shape):
    if filter_shape == None:
        # generating filter shape with the sigma(standard deviation) given
        _ = 2 * int(4 * sigma + 0.5) + 1
        filter_shape = [_, _]

    elif len(filter_shape) != 2:
        raise Exception('shape of argument `filter_shape` is not a supported')

    m, n = filter_shape

    m_half = m // 2
    n_half = n // 2

    gaussian_filter = np.zeros((m, n), np.float32)

    for y in range(-m_half, m_half):
        for x in range(-n_half, n_half):
            normal = 1 / (2.0 * np.pi * sigma**2.0)
            exp_term = np.exp(-(x**2.0 + y**2.0) / (2.0 * sigma**2.0))
            gaussian_filter[y+m_half, x+n_half] = normal * exp_term

    blurred = Convolution(img, gaussian_filter)
    blurred_image = blurred.astype(np.uint8)
    # plt.imshow(blurred_image, cmap='gray', vmin=0, vmax=255)
    # plt.title("Gaussian Blur")
    # plt.show()
    return blurred_image


def Enhanced(image, filter):
    enhanced_img = Convolution(image, filter)
    return enhanced_img


class RetargetedImage:
    imageDirectory = ""
    image = None
    grayImage = None

    def __init__(self, imageDirectory):
        self.imageDirectory = imageDirectory
        self.image = pltim.imread(self.imageDirectory)
        # self.grayImage = gray(self.image)

    def showOriginalImage(self):
        plt.imshow(self.image)
        plt.show()

    def showImage(self):
        # Convert to Gray Scale
        gray_img = Gray(self.image)

        # Blur with Gaussian
        blurred_image = GaussianBlur(gray_img, 10, (3, 3))

        # Init Sobel filter Kernel
        sobel_filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_filter_y = np.flip(sobel_filter_x.T, axis=0)

        # Apply Sobel edge dectection consecutively
        sobel_img_x = Sobel(blurred_image, sobel_filter_x)
        sobel_img_y = Sobel(blurred_image, sobel_filter_y)

        # Get the edge detection output
        sobel_output = np.sqrt(np.square(sobel_img_x) + np.square(sobel_img_y))
        # Take the average of sobel_output to fit it to 255 size
        sobel_output *= 255.0 / sobel_output.max()

        # Init enhancing filter
        enhanced_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        enhanced_sobel_img = Convolution(sobel_output, enhanced_filter)
        # plt.imshow(enhanced_sobel_img, cmap='gray')
        plt.imshow(sobel_output, cmap='gray')
        # print(sobel_output[399])
        plt.show()


example1 = RetargetedImage("img.jpg")
example1.showImage()
