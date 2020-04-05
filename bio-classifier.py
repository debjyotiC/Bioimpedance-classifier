import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import tensorboard
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

df_train = pd.read_csv('data-sets/gf_calculation.csv')

# impedance data
data_air = df_train['Air']
data_water = df_train['Water']

data_train = np.array([data_air, data_water], dtype=float)
data_label = np.array([1.0, 2.0], dtype=float)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, input_shape=(100, ), activation=tf.nn.relu),
    tf.keras.layers.Dense(80, activation=tf.nn.relu),
    tf.keras.layers.Dense(50, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.1), metrics=['accuracy'])
history = model.fit(data_train, data_label, epochs=500, verbose=True)


# epoch_count = range(1, len(history.history['loss']) + 1)
#
# # save performance data
# values = {'Epoch': epoch_count, 'Loss': history.history['loss'], 'Accuracy': history.history['accuracy']}
# df_w = pd.DataFrame(values, columns=['Epoch', 'Loss', 'Accuracy'])
# df_w.to_csv("data-sets/results.csv", index=None, header=True)
#
# fig, axs = plt.subplots(2, 1)
# # plot loss
# axs[0].plot(history.history['loss'], color='Green')
# axs[0].set_xlabel('Epoch')
# axs[0].set_ylabel('Loss')
# axs[0].grid(True)
#
# # plot accuracy
# axs[1].plot(history.history['accuracy'], color='Red')
# axs[1].set_xlabel('Epoch')
# axs[1].set_ylabel('Accuracy')
# axs[1].grid(True)
# plt.show()
