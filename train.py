import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.activations import relu
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

def model():
    inputs = keras.Input(shape=(32, 32, 3))
    x = Conv2D(32, 3)(inputs)
    x = BatchNormalization()(x)
    x = relu(x)
    x = MaxPooling2D()(x)
    x = Conv2D(64, 3)(x)
    x = BatchNormalization()(x)
    x = relu(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, 3)(x)
    x = BatchNormalization()(x)
    x = relu(x)
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

model = model()
model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(lr=3e-4),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=64, epochs=100, verbose=2)
model.save('Output/tf_model.h5')