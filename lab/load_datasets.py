from cv2 import resize
from helper_functions import shuffle_list, check_if_cat
from os.path import abspath
from random import seed
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob
import matplotlib.pyplot as plt
import numpy as np

seed(418)  # I'm a teapot :D
current_directory = abspath("")

train_images = shuffle_list(
    [
        (plt.imread(file_path), check_if_cat(file_path, return_one_hot=True))
        for file_path in glob.glob(
            f"{current_directory}/experiment_small_dataset/train/*.jpg"
        )
    ]
)

val_images = shuffle_list(
    [
        (plt.imread(file_path), check_if_cat(file_path, return_one_hot=True))
        for file_path in glob.glob(
            f"{current_directory}/experiment_small_dataset/val/*.jpg"
        )
    ]
)

test_images = shuffle_list(
    [
        (plt.imread(file_path), check_if_cat(file_path, return_one_hot=True))
        for file_path in glob.glob(
            f"{current_directory}/experiment_small_dataset/test/*.jpg"
        )
    ]
)

train_image_generator = ImageDataGenerator(
    rotation_range=90,
    shear_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    height_shift_range=0.2,
    width_shift_range=0.2,
)
val_image_generator = ImageDataGenerator()


def create_generators(image_size, *, just_train_generator=False, no_transforms=False):
    y_train = np.array([image[1] for image in train_images])
    y_val = np.array([image[1] for image in val_images])

    resized_train_images = [(resize(image[0], image_size)) for image in train_images]
    resized_val_images = [(resize(image[0], image_size)) for image in val_images]

    X_train = (
        np.array([image for image in resized_train_images]).astype("float32") / 255
    )
    X_val = np.array([image for image in resized_val_images]).astype("float32") / 255

    if just_train_generator:
        resized_test_images = [
            (resize(image[0], image_size), image[1]) for image in test_images
        ]
        X_test = (
            np.array([image[0] for image in resized_test_images]).astype("float32")
            / 255
        )
        return (
            train_image_generator.flow(
                np.concatenate((X_train, X_val)),
                np.concatenate((y_train, y_val)),
                batch_size=32,
            ),
            X_test,
            np.array([image[1] for image in test_images]),
        )

    val_generator = val_image_generator.flow(X_val, y_val, batch_size=32)

    if no_transforms:
        train_generator = val_image_generator.flow(X_train, y_train, batch_size=32)
        return (train_generator, val_generator)

    train_generator = train_image_generator.flow(X_train, y_train, batch_size=32)

    return (train_generator, val_generator)
