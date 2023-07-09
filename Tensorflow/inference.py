import onnxruntime as rt

import tensorflow as tf
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_test = x_test.astype('float32') / 255.0

sess = rt.InferenceSession("Output/onnx_model.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

pred = sess.run([label_name], {input_name: x_test})[0]

test_sample_size = pred.shape[0]
correct_pred = 0
# print(y_test[:10])
# print(pred[:10])

for i in range(test_sample_size):
    l = list(pred[i])
    if l.index(max(l)) == y_test[i]:
        correct_pred += 1
        
print("Accuracy of fp32 model: {:.2f}%".format((correct_pred * 100) / test_sample_size))