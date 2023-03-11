import cv2
import glob
import os

# Create the directory for the masked images if it does not exist
if not os.path.exists('masked_images'):
    os.makedirs('masked_images')

# Load the list of images and masks
image_list = glob.glob('images/*.jpg')
mask_list = glob.glob('masks/*.jpg')

# Iterate over each image and its corresponding mask
for i, (image_path, mask_path) in enumerate(zip(image_list, mask_list)):
    # Load the original image and its corresponding mask
    original_image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(original_image, original_image, mask=mask)


    # Save the masked image
    masked_image_path = os.path.join('masked_images', f'masked_image_{i}.jpg')
    cv2.imwrite(masked_image_path, masked_image)

    # Show the result
    cv2.imshow(f'Original Image {i}', original_image)
    cv2.imshow(f'Mask {i}', mask)
    cv2.imshow(f'Masked Image {i}', masked_image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
