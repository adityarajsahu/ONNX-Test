import tensorflow as tf
import tf2onnx

model = tf.keras.models.load_model('Output/tf_model.h5')
tf2onnx.convert.from_keras(model, output_path='Output/onnx_model.onnx')