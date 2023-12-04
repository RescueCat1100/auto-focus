import cv2
import numpy as np
import time
import threading
import matplotlib.image as pltim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
    gray_img = np.dot(image, [0.299, 0.587, 0.114]).astype(np.float32)
    return gray_img


def Filter2D(input_image, kernel):
    """
    My own implementation of filter2D from the OpenCV library
    :param input_image: Input image
    :param kernel: Value of the filter to be used
    :return: Resulting convolution of the input image
    """
    # Get size of kernel
    m = kernel.shape[0]
    n = kernel.shape[1]

    # Pad image with 1px of 0's on each side
    input_image = cv2.copyMakeBorder(
        input_image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    # Store size of image minus the padding
    dim_y = input_image.shape[0] - m + 1
    dim_x = input_image.shape[1] - n + 1

    # Create a new, zeroed array (image)
    new_image = np.zeros((dim_y, dim_x))

    # Loop through each pixel and apply the filter
    for i in range(dim_y):
        for j in range(dim_x):
            # Take the sum of each pixel and then multiply by the kernel (filter)
            conv = (np.sum(input_image[i:i + m, j:j + n] * kernel))

            # Threshold image to prevent clipping
            if conv < 0:
                new_image[i][j] = 0
            elif conv > 255:
                new_image[i][j] = 255
            else:
                new_image[i][j] = conv
    return new_image


def Sobel(image, filter):
    sobel_img = Convolution(image, filter)
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
    blurred_image = blurred.astype(np.float32)
    return blurred_image


def findObject(bb):
    while (True):
        status, image = camera.read()
        # Convert to Gray Scale
        gray_img = Gray(image)
        # Blur with Gaussian
        blurred_image = GaussianBlur(gray_img, 8, (2, 2))
        # Sobe filter
        sobel_filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_filter_y = np.flip(sobel_filter_x.T, axis=0)
        sobel_img_x = Sobel(blurred_image, sobel_filter_x)
        sobel_img_y = Sobel(blurred_image, sobel_filter_y)
        sobel_output = np.sqrt(np.square(sobel_img_x) + np.square(sobel_img_y))
        sobel_output *= 255.0 / sobel_output.max()
        kernel = np.ones((3, 3), np.uint8)

        # Binary edge map
        threshold_value = 20
        lower_bound = 180
        upper_bound = lower_bound + threshold_value
        binary_edge_map = cv2.inRange(sobel_output, lower_bound, upper_bound)

        row_sums = np.sum((binary_edge_map), axis=1)
        col_sums = np.sum((binary_edge_map), axis=0)
        # Find non-zero indices (rows and columns with white pixels)
        rows = np.nonzero(row_sums)[0]
        cols = np.nonzero(col_sums)[0]
        bb[0] = min(cols)
        bb[1] = min(rows)
        bb[2] = max(cols)
        bb[3] = max(rows)

# Print bound


def imageProcess(image, bb):
    min_col = bb[0]
    min_row = bb[1]
    max_col = bb[2]
    max_row = bb[3]
    for j in range(min_col, max_col):
        image[min_row][j] = 0
        image[max_row][j] = 0
    for i in range(min_row, max_row):
        image[i][min_col] = 0
        image[i][max_col] = 0
    return image


# Real-time video processing
camera = cv2.VideoCapture(0)


def playVideo(bb):
    while True:
        status, frame = camera.read()

        frame = imageProcess(frame, bb)
        cv2.imshow("frame", frame)

        if cv2.waitKey(10) == 13:
            break
    cv2.destroyAllWindows()
    camera.release()


bb = np.array([0, 0, 0, 0], dtype=int)
t1 = threading.Thread(target=playVideo, args=(bb,))
t2 = threading.Thread(target=findObject, args=(bb,))

t1.start()
t2.start()

t1.join()
t2.join()

print('done')
