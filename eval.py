import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, jaccard_score,recall_score, precision_score, SCORERS



from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D,MaxPool2D, Conv2DTranspose, Input, Activation, Concatenate, CenterCrop
from tensorflow.keras import Model
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import schedules, Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model, CustomObjectScope
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm


def load_data(dataset_path, split=0.2):
    images = sorted(glob(os.path.join(dataset_path, "/home/Yasmine/PycharmProjects/Dataaugmentation/skin cancer data with mask/ISIC2018_Task1-2_Training_Input/", "*.jpg")))
    masks = sorted(glob(os.path.join(dataset_path, "/home/Yasmine/PycharmProjects/Dataaugmentation/skin cancer data with mask/ISIC2018_Task1_Training_GroundTruth/", "*.png")))

    test_size = int(len(images) * split)

    train_x, valid_x = train_test_split(images, test_size=test_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=test_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)



H = 256
W = 256
#creation du path
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)  ## (H, W, 3)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return ori_x, x                                ## (256, 256, 3)

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ## (H, W)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.int32)           ## (256, 256)
    return ori_x, x

np.random.seed(42)
tf.random.set_seed(42)

create_dir("results")
model = tf.keras.models.load_model("files/model.h5")

dataset_path = "/home/Yasmine/PycharmProjects/Dataaugmentation/skin cancer data with mask/"
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)
SCORE = []

def  save_results(ori_x, ori_y, y_pred, save_image_path):
     line = np.ones((H, 10, 3)) * 225
     ori_y = np.expand_dims(ori_y, axis = -1)
     ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1)
     y_pred = np.expand_dims(y_pred, axis =-1)
     y_pred = np.concatenate([y_pred, y_pred, y_pred], axis =-1)
     cat_images = np.concatenate([ori_x, line, ori_y, line, y_pred*255], axis =1)
     cv2.imwrite(save_image_path, cat_images)



for x, y in tqdm(zip(test_x, test_y), total = len(test_x)):
    name = x.split("/")[-1]
    ori_x, x = read_image(x)
    ori_y, y = read_mask(y)

    #predicting the masks
    y_pred = model.predict(x)[0] > 0.5
    y_pred = np.squeeze(y_pred, axis = -1)
    y_pred = y_pred.astype(np.int32)

    save_image_path = f"results/{name}"
    save_results(ori_x, ori_y, y_pred, save_image_path)

    y= y.flatten()
    y_pred = y_pred.flatten()
    acc_value = accuracy_score(y, y_pred)
    f1_value = f1_score(y, y_pred, labels=[0,1], average="binary")
    jac_value = jaccard_score(y, y_pred, labels=[0,1], average="binary")
    rec_value = recall_score(y, y_pred, labels=[0,1], average="binary")
    precision_value = precision_score(y, y_pred, labels=[0,1], average="binary")

    SCORE.append([name,acc_value, f1_value, jac_value, rec_value, precision_value])
    score = [s[1:] for s in SCORE]
    score = np.mean(score, axis=0)