import matplotlib.pyplot as plt


def display_images(data, rows=2, cols=5, figsize=(12, 4)):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    for i, ax in enumerate(axes.flatten()):
        ax.imshow(data[i, :, :], cmap="gray")
        ax.axis("off")
    fig.subplots_adjust(wspace=0, hspace=0.1, bottom=0)
