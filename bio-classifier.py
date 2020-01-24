import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import tensorboard
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

df_train = pd.read_csv('data-sets/sensor_data_cancer_train.csv')
df_test = pd.read_csv('data-sets/sensor_data_cancer_test.csv')

# cancerous cell
data_thp_100 = df_train['100%_THP']
data_thp_80 = df_train['80%_THP']
data_thp_60 = df_train['60%_THP']
data_thp_50 = df_train['50%_THP']
data_thp_40 = df_train['40%_THP']
data_thp_30 = df_train['30%_THP']
data_thp_20 = df_train['20%_THP']
data_thp_10 = df_train['10%_THP']

# non-cancerous cell
data_pha = df_train['PHA(-)']
data_air = df_train['Air']
# test data
data_got = df_test['datagot']


data_train = np.array([data_thp_100, data_thp_80, data_thp_60, data_thp_50, data_thp_40, data_thp_30, data_thp_20,
                      data_thp_10, data_pha, data_air])
data_label = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
data_test = np.array([data_thp_100])

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, input_shape=(100, ), activation=tf.nn.relu),
    tf.keras.layers.Dense(80, activation=tf.nn.relu),
    tf.keras.layers.Dense(50, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.1), metrics=['accuracy'])
history = model.fit(data_train, data_label, epochs=500, verbose=True)

predictions = model.predict(data_test)

print(predictions)

epoch_count = range(1, len(history.history['loss']) + 1)

# save performance data
values = {'Epoch': epoch_count, 'Loss': history.history['loss'], 'Accuracy': history.history['accuracy']}
df_w = pd.DataFrame(values, columns=['Epoch', 'Loss', 'Accuracy'])
df_w.to_csv("data-sets/results.csv", index=None, header=True)

#fig, axs = plt.subplots(2, 1)
# plot loss
#axs[0].plot(history.history['loss'], color='Green')
#axs[0].set_xlabel('Epoch')
#axs[0].set_ylabel('Loss')
#axs[0].grid(True)

# plot accuracy
#axs[1].plot(history.history['accuracy'], color='Red')
#axs[1].set_xlabel('Epoch')
#axs[1].set_ylabel('Accuracy')
#axs[1].grid(True)
#plt.show()
