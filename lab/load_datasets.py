from tkinter.messagebox import RETRY
from cv2 import resize
from helper_functions import (
    shuffle_list,
    check_if_cat_mac,
    check_if_cat_windows,
)
from os.path import abspath
from random import seed
from sys import platform
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob
import matplotlib.pyplot as plt
import numpy as np

seed(418)
current_directory = abspath("")

if platform == "darwin":
    train_images = shuffle_list(
        [
            (plt.imread(file_path), check_if_cat_mac(file_path, return_one_hot=True))
            for file_path in glob.glob(
                f"{current_directory}/experiment_small_dataset/train/*.jpg"
            )
        ]
    )

    val_images = shuffle_list(
        [
            (plt.imread(file_path), check_if_cat_mac(file_path, return_one_hot=True))
            for file_path in glob.glob(
                f"{current_directory}/experiment_small_dataset/val/*.jpg"
            )
        ]
    )

    test_images = shuffle_list(
        [
            (plt.imread(file_path), check_if_cat_mac(file_path, return_one_hot=True))
            for file_path in glob.glob(
                f"{current_directory}/experiment_small_dataset/test/*.jpg"
            )
        ]
    )

if platform == "win32":
    train_images = shuffle_list(
        [
            (
                plt.imread(file_path),
                check_if_cat_windows(file_path, return_one_hot=True),
            )
            for file_path in glob.glob(
                f"{current_directory}/experiment_small_dataset/train/*.jpg"
            )
        ]
    )

    val_images = shuffle_list(
        [
            (
                plt.imread(file_path),
                check_if_cat_windows(file_path, return_one_hot=True),
            )
            for file_path in glob.glob(
                f"{current_directory}/experiment_small_dataset/val/*.jpg"
            )
        ]
    )

    test_images = shuffle_list(
        [
            (
                plt.imread(file_path),
                check_if_cat_windows(file_path, return_one_hot=True),
            )
            for file_path in glob.glob(
                f"{current_directory}/experiment_small_dataset/test/*.jpg"
            )
        ]
    )


resized_train_images = [
    (resize(image[0], (32, 32)), image[1]) for image in train_images
]
resized_val_images = [(resize(image[0], (32, 32)), image[1]) for image in val_images]
resized_test_images = [(resize(image[0], (32, 32)), image[1]) for image in test_images]


X_train = np.array([image[0] for image in resized_train_images]).astype("float32") / 255
X_val = np.array([image[0] for image in resized_val_images]).astype("float32") / 255
X_test = np.array([image[0] for image in resized_test_images]).astype("float32") / 255
X_train.shape

y_train = np.array([image[1] for image in resized_train_images])
y_val = np.array([image[1] for image in resized_val_images])
y_test = np.array([image[1] for image in resized_test_images])
y_train.shape

train_image_generator = ImageDataGenerator(
    rotation_range=90,
    shear_range=0.2,
    zoom_range=0.2,
    height_shift_range=0.2,
    width_shift_range=0.2,
)
test_image_generator = ImageDataGenerator()

train_val_generator = train_image_generator.flow(X_train, y_train, batch_size=32)
val_generator = test_image_generator.flow(X_val, y_val, batch_size=32)

train_generator = test_image_generator.flow(
    np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)), batch_size=32
)
test_generator = test_image_generator.flow(X_test, y_test, batch_size=32)
