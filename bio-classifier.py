import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

df_train = pd.read_csv('data-sets/cleaned-data.csv')

data_thp_90 = df_train['90_THP']
data_thp_10 = df_train['10_THP']

data_test0 = df_train['20_THP']


data_train = np.array([data_thp_90, data_thp_10], dtype=float)
data_label = np.array([0.0, 1.0], dtype=float)

data_test = np.array([data_test0], dtype=float)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, activation='sigmoid'),
    tf.keras.layers.Dense(50, activation='sigmoid'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(data_train, data_label, epochs=20, verbose=True)

predictions = model.predict(data_test)

print(np.argmax(predictions[0]))

epoch_count = range(1, len(history.history['loss']) + 1)

# save performance data
values = {'Epoch': epoch_count, 'Loss': history.history['loss'], 'Accuracy': history.history['accuracy']}
df_w = pd.DataFrame(values, columns=['Epoch', 'Loss', 'Accuracy'])
df_w.to_csv("data-sets/results.csv", index=None, header=True)

fig, axs = plt.subplots(2, 1)
# plot loss
axs[0].plot(history.history['loss'], color='Green')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].grid(True)

# plot accuracy
axs[1].plot(history.history['accuracy'], color='Red')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].grid(True)
plt.show()