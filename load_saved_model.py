import tensorflow as tf
import numpy as np

x_test = [499206.5769, 4992.065769, 25895.72974]

load_model = tf.keras.models.load_model('saved_model/tf_model')

test = np.vstack([x_test])
classes = load_model.predict(test)

if classes[0] > 0.5:
    print("is Cancerous")
else:
    print("is not Cancerous")
