import os

import numpy as np
from keras import (
    Sequential,
    regularizers,
)
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dropout,
    Dense,
)

from lib import (
    layer_to_rust,
    weight_to_rust,
)

file_path = os.path.dirname(__file__)


def compute(layer, inpt):
    out = layer(inpt).numpy()
    data_rs = layer_to_rust(layer)
    data_rs += "\n" + weight_to_rust("INPUT", inpt)
    data_rs += "\n" + weight_to_rust("EXPECT", out)
    print(data_rs)
    breakpoint()
    return out


WIDTH, HEIGHT, RGB_CHANNELS = 5, 5, 2
BATCH = 1
marvin = np.load(f"{file_path}/marvin.npz")["X"]
IMG_WIDTH, IMG_HEIGHT, CHANNEL = marvin.shape
inpt = marvin[np.newaxis, ...]


model = Sequential(
    [
        Conv2D(
            4,
            3,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(0.001),
            name="conv_layer1",
            input_shape=(IMG_WIDTH, IMG_HEIGHT, 1),
        ),
        MaxPooling2D(name="max_pooling1", pool_size=(2, 2)),
        Conv2D(
            4,
            3,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(0.001),
            name="conv_layer2",
        ),
        MaxPooling2D(name="max_pooling2", pool_size=(2, 2)),
        Flatten(),
        Dropout(0.2),
        Dense(
            40,
            activation="relu",
            kernel_regularizer=regularizers.l2(0.001),
            name="hidden_layer1",
        ),
        Dense(
            1,
            activation="sigmoid",
            kernel_regularizer=regularizers.l2(0.001),
            name="output",
        ),
    ]
)
model.load_weights(f"{file_path}/diy-alexa/model/trained.keras")

for layer in model.layers:
    inpt = compute(layer, inpt)
