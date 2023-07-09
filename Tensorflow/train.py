import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import cifar10

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


model = tf.keras.models.Sequential()
model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[32,32,3]))
model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=2, strides=2, padding="valid"))
model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=2, strides=2, padding='valid'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=['sparse_categorical_accuracy'])
model.fit(x_train, y_train, epochs=25)

print(model.evaluate(x_test, y_test))
# pred = model.predict(x_test)
# print(pred[:10])
# print(y_test[:10])

model.save('Output/tf_model.h5')