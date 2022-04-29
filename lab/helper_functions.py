import matplotlib.pyplot as plt
from random import shuffle, seed


def plot_sample_images(image_list: list):
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for index, ax in enumerate(axes.flatten()):
        ax.imshow(image_list[index][0], cmap="gray")
        ax.set_title(image_list[index][1])
        ax.axis("off")
    fig.tight_layout()


def check_if_cat_windows(file_path, return_one_hot: bool = False):
    if file_path.split("/")[-1][6:].split(".")[0] == "cat":
        if return_one_hot:
            return 0
        return "cat"
    else:
        if return_one_hot:
            return 1
        return "dog"


def check_if_cat_mac(file_path, return_one_hot: bool = False):
    if file_path.split("/")[-1].split(".")[0] == "cat":
        if return_one_hot:
            return 0
        return "cat"
    else:
        if return_one_hot:
            return 1
        return "dog"


def shuffle_list(list_x: list):
    shuffle(list_x)
    return list_x
