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


def Enhanced(image, filter):
    enhanced_img = Convolution(image, filter)
    return enhanced_img


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


def LensDefocus(img, dim):

    imgarray = np.array(img, dtype="uint8")
    kernel_w = dim
    kernel = np.zeros((kernel_w, kernel_w), dtype=np.uint8)
    kernel = DiskKernel(dim)
    pass


def Show(image):
    # Convert to Gray Scale
    gray_img = Gray(image)
    print('slow here 1')
    # Blur with Gaussian
    blurred_image = GaussianBlur(gray_img, 5, (2, 2))
    # gray_img = Gray(blurred_image)
    blurred_image2 = cv2.GaussianBlur(gray_img, (3, 3), 15)
    f, axarr = plt.subplots(2, 2)
    print('slow here 2')
    axarr[0, 0].imshow(blurred_image, cmap='gray')

    axarr[1, 0].imshow(gray_img)
    # plt.show()
    # Init Sobel filter Kernel
    sobel_filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_filter_y = np.flip(sobel_filter_x.T, axis=0)
    print('slow here 3')
    # Apply Sobel edge dectection consecutively
    sobel_img_x = Sobel(blurred_image, sobel_filter_x)
    sobel_img_y = Sobel(blurred_image, sobel_filter_y)
    print('slow here 4')
    # Get the edge detection output
    sobel_output = np.sqrt(np.square(sobel_img_x) + np.square(sobel_img_y))
    # Take the average of sobel_output to fit it to 255 size
    print('slow here 5')
    sobel_output *= 255.0 / sobel_output.max()
    # Bokeh(image)
    # Init enhancing filter
    enhanced_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    # Optional: Perform morphological operations for refinement
    kernel = np.ones((3, 3), np.uint8)

    axarr[1, 1].imshow(sobel_output, cmap='gray')
    # plt.show()

    # Threshold the edge map to create a binary image
    """
    for row in sobel_output:
        for pixel_value in row:
            print(pixel_value, end=' ')
        print()
    """
    # Assuming edge_map is your ndarray binary edge map
    threshold_value = 10
    lower_bound = 190
    upper_bound = lower_bound + threshold_value
    # Create a new binary image
    # binary_edge_map = (sobel_output > lower_bound).astype(np.uint8) * 255
    fig, ax = plt.subplots()
    # binary_edge_map = (binary_edge_map < upper_bound).astype(np.uint8) * 255
    binary_edge_map = cv2.inRange(sobel_output, lower_bound, upper_bound)
    # contours, _ = cv2.findContours(
    #    binary_edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for contour in contours:
    #    x, y, w, h = cv2.boundingRect(contour)
    #    rect = patches.Rectangle(
    #        (x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
    #    ax.add_patch(rect)
    axarr[0, 1].imshow(binary_edge_map, cmap='gray')
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

    ax.imshow(image)

    bb = patches.Rectangle((x, y), w, h, linewidth=2,
                           edgecolor='r', facecolor='none')
    ax.add_patch(bb)
    sobel_output = np.array(sobel_output, np.uint8)
    contours, _ = cv2.findContours(
        sobel_output[y:y+h, x:x+w], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plt.show()
    mask = np.zeros_like(sobel_output)
    cv2.drawContours(mask[y:y+h, x:x+w], contours, -1,
                     (255, 255, 255), thickness=cv2.FILLED)

    # Use the mask to extract the object
    object_mask = cv2.bitwise_and(image, mask)

    # Display the result
    cv2.imshow('Object Mask', object_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #######
    mask = np.zeros_like(image)

    # Draw a white rectangle on the mask within the bounding box
    cv2.rectangle(mask, (x, y), (x + w, y + h),
                  (255, 255, 255), thickness=cv2.FILLED)

    # Use the mask to blend the original image and a blurred version
    blurred_img = cv2.GaussianBlur(image, (21, 21), 0)
    out = np.where(mask != 0, image, blurred_img)

    # Display the result
    cv2.imshow('Blurred Image', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.destroyAllWindows()

    pass


imgPath = 'img.jpg'
img = pltim.imread(imgPath)
Show(img)
