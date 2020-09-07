import tensorflow as tf
import numpy as np

x_test = np.array([502692.0591, 5026.920591, 26076.53486], dtype='float32')

# load Keras model
# load_model = tf.keras.models.load_model('saved_model/tf_model')
# test = np.vstack([x_test])
# classes = load_model.predict(test)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="saved_model/tflite_model/converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

test = np.vstack([x_test])
interpreter.set_tensor(input_details[0]['index'], test)

interpreter.invoke()

classes = interpreter.get_tensor(output_details[0]['index'])

if classes[0] > 0.5:
    print("is Cancerous")
else:
    print("is not Cancerous")