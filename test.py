import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'img.jpg'
original_image = cv2.imread(image_path)

# Create a mask (you can create a mask using various methods)
# For example, creating a circular mask for a rectangular image
mask = np.zeros_like(original_image)
height, width, _ = original_image.shape
center = (width // 2, height // 2)
radius = min(center[0], center[1])
cv2.circle(mask, center, radius, (255, 255, 255), -1)

# Apply the mask to the image
masked_image = cv2.bitwise_and(original_image, mask)

# Display the results
plt.subplot(131), plt.imshow(cv2.cvtColor(original_image,
                                          cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(132), plt.imshow(cv2.cvtColor(
    mask, cv2.COLOR_BGR2RGB)), plt.title('Mask')
plt.subplot(133), plt.imshow(cv2.cvtColor(
    masked_image, cv2.COLOR_BGR2RGB)), plt.title('Masked Image')
plt.show()
