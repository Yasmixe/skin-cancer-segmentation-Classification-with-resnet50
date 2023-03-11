import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import keras
import tensorflow.keras.backend as K
from skimage.transform import resize
from tqdm import tqdm
import cv2
W=H=256
def iou(y_true, y_pred, smooth=1):
    y_true = K.expand_dims(y_true, axis=-1)
    y_pred = K.expand_dims(y_pred, axis=-1)
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    iou_score = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou_score


def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)  ## (H, W, 3)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return x                                ## (1, 256, 256, 3)

np.random.seed(42)
tf.random.set_seed(42)
import keras
import tensorflow.keras.backend as K

with keras.utils.custom_object_scope({'iou': iou}):
    model = tf.keras.models.load_model("/home/Yasmine/PycharmProjects/PFEE/model70.h5")

image_path = "/home/Yasmine/PycharmProjects/PFEE/augmented_test_malign/crop20_3.jpg"
x = read_image(image_path)
y_pred = model.predict(x)[0] > 0.5
y_pred = np.squeeze(y_pred, axis=-1)
y_pred = y_pred.astype(np.int32)

# save the mask
mask_path = "/home/Yasmine/PycharmProjects/PFEE/mask.jpg"
import os
extension = ".jpg" # Change extension to desired format (e.g. .jpg, .bmp, etc.)
cv2.imwrite(os.path.join(mask_path + extension), y_pred.astype(np.uint8) * 255)
