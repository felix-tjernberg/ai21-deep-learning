from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from helper_functions import flatten_list

import matplotlib.pyplot as plt
import pandas as pd


def model_mlp_2_classes(
    dense_and_dropout_pairs,
    input_shape,
    *,
    name: str = "transferacne_model_2_classes",
    optimizer=Adam(learning_rate=0.001),
    metrics=["acc"],
    flatten_layer_dropout_rate=0.3,
):
    model = Sequential(
        [Dense(shape=input_shape), Dropout(flatten_layer_dropout_rate)]
        + flatten_list(
            [
                create_Dense_Dropout_pairs(dense_and_dropout_pair)
                for dense_and_dropout_pair in dense_and_dropout_pairs
            ]
        )
        + [Dense(1, activation="sigmoid", name="output_layer")],
        name=name,
    )
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=metrics)
    return model


def model_transference_2_classes_combined_method(
    base_model,
    mlp_node_dropout_pairs,
    *,
    name: str = "transferacne_model_2_classes",
    optimizer=Adam(learning_rate=0.001),
    metrics=["acc"],
    flatten_layer_dropout_rate=0.3,
):
    base_model.trainable = False

    model = Sequential(
        [base_model, Flatten(), Dropout(flatten_layer_dropout_rate)]
        + flatten_list(
            [
                create_Dense_Dropout_pairs(node_dropout_pair)
                for node_dropout_pair in mlp_node_dropout_pairs
            ]
        )
        + [Dense(1, activation="sigmoid", name="output_layer")],
        name=name,
    )
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=metrics)
    return model


def create_Conv2D_MaxPooling2D_pairs(
    kernel_number,
    *,
    strides=2,
    kernel_size=(3, 3),
    input_shape=(32, 32, 3),  # print(X_train.shape[1:]) -> (32, 32, 3)
):
    return [
        Conv2D(
            kernel_number,
            padding="same",
            kernel_size=kernel_size,
            activation="relu",
            kernel_initializer="he_normal",
            input_shape=input_shape,
        ),
        MaxPooling2D(pool_size=(2, 2), strides=strides),
    ]


def create_Dense_Dropout_pairs(node_dropout_pair):
    return [
        Dense(node_dropout_pair[0], activation="relu", kernel_initializer="he_normal"),
        Dropout(node_dropout_pair[1]),
    ]


def model_cnn_2_classes(
    cnn_kernel_numbers: list,
    mlp_node_dropout_pairs,
    *,
    name: str = "cnn_model_2_classes",
    optimizer=Adam(learning_rate=0.001),
    metrics=["acc"],
    flatten_layer_dropout_rate=0.3,
    strides=2,
    kernel_size=(3, 3),
    input_shape=(32, 32, 3),
):
    flatten_layer = [Flatten(), Dropout(flatten_layer_dropout_rate)]
    model = Sequential(
        flatten_list(
            [
                create_Conv2D_MaxPooling2D_pairs(
                    kernel_number,
                    strides=strides,
                    kernel_size=kernel_size,
                    input_shape=input_shape,
                )
                for kernel_number in cnn_kernel_numbers
            ]
        )
        + flatten_layer
        + flatten_list(
            [
                create_Dense_Dropout_pairs(node_dropout_pair)
                for node_dropout_pair in mlp_node_dropout_pairs
            ]
        )
        + [Dense(1, activation="sigmoid", name="output_layer")],
        name=name,
    )
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=metrics)
    return model


def plot_and_print_model_metrics(metrics: pd.DataFrame):
    _, ax = plt.subplots(1, 2, figsize=(12, 4))
    print(f'val_acc: {metrics["val_acc"].sort_values().iloc[-1]}')
    metrics[["loss", "val_loss"]].plot(ax=ax[0], grid=True)
    metrics[["acc", "val_acc"]].plot(ax=ax[1], grid=True)


def fit_then_evaluate_model(
    model,
    train_generator,
    val_generator,
    *,
    steps_per_epoch=50,  # int(len(X_train) / 32) -> 50
    validation_steps=12,  #  int(len(X_val) / 32) -> 12
    model_fit_verbosity=0,
    patience=10,
):
    early_stopper = EarlyStopping(
        monitor="val_acc", mode="max", patience=patience, restore_best_weights=True
    )
    model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=100000,
        callbacks=[early_stopper],
        validation_data=val_generator,
        validation_steps=validation_steps,
        verbose=model_fit_verbosity,
    )
    metrics = pd.DataFrame(model.history.history)
    plot_and_print_model_metrics(metrics)

    return model
