import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


df_train = pd.read_csv('data-sets/sensor_data_cancer_train.csv')
data_thp = df_train['100%_THP']   # cancerous cell
data_pha = df_train['PHA(-)']     # non-cancerous cell

data_cell = np.array([data_thp, data_pha])
data_label = np.array([1, 0])

df_test = pd.read_csv('data-sets/sensor_data_cancer_test.csv')
data_test = np.array(df_test['datagot'])

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(100, )),
    tf.keras.layers.Dense(80, activation='sigmoid'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
history = model.fit(data_cell, data_label, epochs=100)

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


