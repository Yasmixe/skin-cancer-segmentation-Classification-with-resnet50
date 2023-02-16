from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import PIL
import PIL.ImageEnhance
import pydicom
import os
import stat
from PIL import Image
import numpy as np

folder_dir = r'/home/Yasmine/PycharmProjects/Dataaugmentation/skin cancer data with mask/ISIC2018_Task1_Training_GroundTruth/'

st = os.stat(folder_dir)
os.chmod(folder_dir, st.st_mode | stat.S_IWOTH)  # Force open

for image in os.listdir(folder_dir):
    Inter = image
    print(Inter)

    fichier = folder_dir + image

    print(fichier)
    image = []

    # Opening Image

    if '.dcm' in fichier:
        image = pydicom.dcmread(fichier)
        image = image.pixel_array.astype(float)
    else:
        image2 = Image.open(fichier)
        image2 = np.asfarray(image2)
        image = image2

    image_scaled = (np.maximum(image, 0) / image.max()) * 255 # Changing image values to 0 - 255
    image_finale = np.uint8(image_scaled)  # Conversion to Integer

    # Data Augmentation

    # Cropping
    Y = 0

    for i in image_finale:
        X = len(i)
        Y += 1
    # Cropping

    for i in [30, 51]:
        cropX = int(X*i/100)
        cropY = int(Y*i/100)

        # Crop values equalizer
        if cropX % 2 == 1:
            FcropX = (cropX + 1)/2
        else:
            FcropX = cropX/2

        if cropY % 2 == 1:
            FcropY = (cropY + 1)/2
        else:
            FcropY = cropY/2

        if i == 30:
            image_save = Image.fromarray(image_finale)
            image_save.save('Rescaled.PNG')

        # Actual Cropping
        image_finale = Image.open('Rescaled.PNG')
        cropped = image_finale.crop((int(FcropX), int(FcropY), X-int(FcropX), Y-int(FcropY)))
        SavePath = r'/home/Yasmine/PycharmProjects/Dataaugmentation/augmented_mask/' + 'crop' + str(i) + '_' + Inter
        cropped.save(SavePath)

    # Applying Median Filter
    #nehiw noise
    for i in [3]:
        PreImage1 = Image.open('Rescaled.PNG')
        PreImage1 = PreImage1.filter(PIL.ImageFilter.MedianFilter(size=i))
        SavePath = r'/home/Yasmine/PycharmProjects/Dataaugmentation/augmented_mask/'+ 'median' + str(i) + '_' + Inter
        PreImage1.save(SavePath)

        # Brightness Change

    # Rotation

    for i in [90,180]:
        PreImage3 = Image.open('Rescaled.PNG')
        InterVar = PreImage3.rotate(i)
        SavePath = r'/home/Yasmine/PycharmProjects/Dataaugmentation/augmented_mask/' + 'rotate' + str(i) + '_' + Inter
        InterVar.save(SavePath)