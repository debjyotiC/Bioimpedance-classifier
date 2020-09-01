import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

df_train = pd.read_csv('data-sets/data-gen.csv')

data_thp_100 = df_train['100_THP']
data_thp_30 = df_train['30_THP']

data_thp_10 = df_train['15_THP']
data_pha_100 = df_train['PHA']

test = df_train['10_THP']

data_train = np.array([data_thp_100, data_thp_30, data_thp_10, data_pha_100], dtype=float)
data_label = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)
data_test = np.array([test], dtype=float)


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='softmax', input_shape=(100, )),
    tf.keras.layers.Dense(60, activation='softmax'),
    tf.keras.layers.Dense(10, activation='softmax'),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

history = model.fit(data_train, data_label, epochs=100, verbose=True)

predictions = model.predict(data_test)

print(np.argmax(predictions[0]))

# epoch_count = range(1, len(history.history['loss']) + 1)

# save performance data
# values = {'Epoch': epoch_count, 'Loss': history.history['loss'], 'Accuracy': history.history['accuracy']}
# df_w = pd.DataFrame(values, columns=['Epoch', 'Loss', 'Accuracy'])
# df_w.to_csv("data-sets/results.csv", index=None, header=True)

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