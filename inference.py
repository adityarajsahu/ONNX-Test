import onnxruntime as rt

import tensorflow as tf
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_test = x_test.astype('float32') / 255.0

sess = rt.InferenceSession("Output/onnx_model.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

pred = sess.run([label_name], {input_name: x_test})[0]
print(pred)