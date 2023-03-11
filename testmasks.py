import cv2
import numpy as np
# Load the original image and its corresponding mask
original_image = cv2.imread("crop20_data_3.jpg")
mask = cv2.imread("crop20_mask_3.png", cv2.IMREAD_GRAYSCALE)
folder_path = "/home/Yasmine/PycharmProjects/PFE/"
# Apply the mask to the original image
masked_image = cv2.bitwise_and(original_image, original_image, mask=mask)

output_path = "masked_image-4.png"
blurred_image = cv2.GaussianBlur(masked_image, (7, 7), 0)
#sharpened images
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpened_image = cv2.filter2D(blurred_image, -1, kernel)

cv2.imwrite(output_path, sharpened_image)


# Show the result
cv2.imshow("Original Image", original_image)
cv2.imshow("Mask", mask)
cv2.waitKey(0)
