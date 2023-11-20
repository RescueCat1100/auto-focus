import numpy as np
import matplotlib.image as pltim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2


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


def my_filter_2d(input_image, kernel):
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


def Enhanced(image: np.ndarray, boundingbox):
    x, y, w, h = boundingbox
    array = image.copy()
    enhance_kernel = np.array(
        ([0, -1, 0], [-1, 5, -1], [0, -1, 0])).astype(np.float32)
    print(array.shape)
    print(type(array))
    obj_region = array[y:y+h, x:x+w]
    result_roi = np.zeros_like(obj_region)
    for i in range(obj_region.shape[2]):  # Loop over each color channel
        result_roi[:, :, i] = my_filter_2d(obj_region[:, :, i], enhance_kernel)

    array[y:y+h, x:x+w] = result_roi
    return array

    shape = array
    source_array = np.zeros((shape[0], shape[1], 3))

    # except:
    # s2 = 0
    # source_array[i, j] = (
    # r / kernel_weight, g / kernel_weight, b / kernel_weight)

    # cap the values
    np.putmask(source_array, source_array > 255, 255)
    np.putmask(source_array, source_array < 0, 0)

    return source_array


"""
def Bokeh(image):
    # read input and convert to grayscale
    img = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)

    # do dft saving as complex output
    dft_img = np.fft.fft2(img, axes=(0, 1))

    # create circle mask
    radius = 2
    mask = np.zeros_like(img)
    cy = mask.shape[0] // 2
    cx = mask.shape[1] // 2
    cv2.circle(mask, (cx, cy), radius, 255, -1)[0]

    # blur the mask slightly to antialias
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    # roll the mask so that center is at origin and normalize to sum=1
    mask_roll = np.roll(mask, (cy, cx), axis=(0, 1))
    mask_norm = mask_roll / mask_roll.sum()

    # take dft of mask
    dft_mask_norm = np.fft.fft2(mask_norm, axes=(0, 1))

    # apply dft_mask to dft_img
    dft_shift_product = np.multiply(dft_img, dft_mask_norm)

    # do idft saving as complex output
    img_filtered = np.fft.ifft2(dft_shift_product, axes=(0, 1))

    # combine complex real and imaginary components to form (the magnitude for) the original image again
    img_filtered = np.abs(img_filtered).clip(0, 255).astype(np.float32)

    cv2.imshow("ORIGINAL", img)
    cv2.imshow("MASK", mask)
    cv2.imshow("FILTERED DFT/IFT ROUND TRIP", img_filtered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # write result to disk
    cv2.imwrite("lena_512_gray_mask.png", mask)
    cv2.imwrite("lena_dft_numpy_lowpass_filtered_rad32.jpg", img_filtered)
    pass

"""


def Object(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold input image as mask
    mask = cv2.threshold(gray, 150, 160, cv2.THRESH_BINARY)[1]

    # negate mask
    # mask = 255 - mask

    # apply morphology to remove isolated extraneous noise
    # use borderconstant of black since foreground touches the edges
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # anti-alias the mask -- blur then stretch
    # blur alpha channel
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2, sigmaY=2,
                            borderType=cv2.BORDER_DEFAULT)

    # linear stretch so that 127.5 goes to 0, but 255 stays 255
    mask = (2*(mask.astype(np.float32))-255.0).clip(0, 255).astype(np.uint8)

    # put mask into alpha channel
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask

    # save resulting masked image
    cv2.imwrite('person_transp_bckgrnd.png', result)

    # display result, though it won't show transparency
    cv2.imshow("INPUT", img)
    cv2.imshow("GRAY", gray)
    cv2.imshow("MASK", mask)
    cv2.imshow("RESULT", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pass


def white_balance(img):
    result = np.copy(img)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - \
        ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - \
        ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    return result


"""
def LensDefocus(img, dim):

    imgarray = np.array(img, dtype="uint8")
    kernel_w = dim
    kernel = np.zeros((kernel_w, kernel_w), dtype=np.uint8)
    kernel = DiskKernel(dim)
    pass

"""


def Show(image):
    img = white_balance(image)
    # Convert to Gray Scale
    gray_img = Gray(img)
    print('slow here 1')
    # Blur with Gaussian
    blurred_image = GaussianBlur(gray_img, 5, (40, 40))
    # gray_img = Gray(blurred_image)
    blurred_image2 = cv2.GaussianBlur(gray_img, (3, 3), 15)
    f, axarr = plt.subplots(2, 2)
    print('slow here 2')
    sobel_filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_filter_y = np.flip(sobel_filter_x.T, axis=0)
    print('slow here 3')
    sobel_img_x = Sobel(blurred_image, sobel_filter_x)
    sobel_img_y = Sobel(blurred_image, sobel_filter_y)
    print('slow here 4')
    sobel_output = np.sqrt(np.square(sobel_img_x) + np.square(sobel_img_y))
    print('slow here 5')
    sobel_output *= 255.0 / sobel_output.max()
    enhanced_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    kernel = np.ones((3, 3), np.uint8)

    """
    for row in sobel_output:
        for pixel_value in row:
            print(pixel_value, end=' ')
        print()
    """
    threshold_value = 20
    lower_bound = 180
    upper_bound = lower_bound + threshold_value
    fig, ax = plt.subplots(1, 2)
    binary_edge_map = cv2.inRange(sobel_output, lower_bound, upper_bound)

    row_sums = np.sum((binary_edge_map), axis=1)
    col_sums = np.sum((binary_edge_map), axis=0)
    # print(row_sums, col_sums)
    # Find non-zero indices (rows and columns with white pixels)
    rows = np.nonzero(row_sums)[0]
    cols = np.nonzero(col_sums)[0]
    # print(rows, cols)
    # Calculate bounding box coordinates
    x, y, w, h = min(cols), min(rows), max(cols) - \
        min(cols), max(rows) - min(rows)
    # Display the result
    bb = x, y, w, h
    print(bb)
    # ax.imshow(sobel_output)
    ax[0].imshow(image)
    enhanced_img = Enhanced(image, bb)
    ax[1].imshow(enhanced_img)
    # plt.show()

    bb = patches.Rectangle((x, y), w, h, linewidth=2,
                           edgecolor='r', facecolor='none')
    ax[0].add_patch(bb)
    crop_img = sobel_output[y:y+h, x:x+w]
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(sobel_output, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(sobel_output, cv2.MORPH_CLOSE, kernel)
    img = white_balance(image)
    axarr[0, 0].imshow(gray_img, cmap='gray')
    axarr[1, 0].imshow(img)
    axarr[1, 1].imshow(crop_img, cmap='gray')
    axarr[0, 1].imshow(binary_edge_map, cmap='gray')
    # plt.show()
    """
    sobel_output = np.array(sobel_output, np.uint8)
    contours, _ = cv2.findContours(
        sobel_output[y:y+h, x:x+w], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(sobel_output)
    cv2.drawContours(mask[y:y+h, x:x+w], contours, -1,
                     (255, 255, 255), thickness=cv2.FILLED)

    # Use the mask to extract the object
    object_mask = cv2.bitwise_and(binary_edge_map, mask)
    # Display the result
    cv2.imshow('Object Mask', object_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    #######
    mask = np.zeros_like(image)

    # Draw a white rectangle on the mask within the bounding box
    cv2.rectangle(mask, (x, y), (x + w, y + h),
                  (255, 255, 255), thickness=cv2.FILLED)

    # Use the mask to blend the original image and a blurred version
    blurred_img = cv2.GaussianBlur(image, (21, 21), 0)
    out = np.where(mask != 0, image, blurred_img)
    # Object(out)
    # Display the result
    _, ax2 = plt.subplots()
    ax2.imshow(out)
    plt.show()
    pass


imgPath = 'img4.jpg'
img = pltim.imread(imgPath)
Show(img)
