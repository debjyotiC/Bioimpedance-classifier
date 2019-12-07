import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

df_train = pd.read_csv('data-sets/sensor_data_cancer_train.csv')
data_thp = df_train['100%_THP']   # cancerous cell
data_pha = df_train['PHA(-)']     # non-cancerous cell
data_label = np.array([1, 0])
data_cell = np.array([data_thp, data_pha])
df_test = pd.read_csv('data-sets/sensor_data_cancer_test.csv')
data_test = np.array(df_test['datagot'])

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(100, )),
    tf.keras.layers.Dense(50, activation='softmax'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(1, activation='softmax')
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(data_cell, data_label, epochs=10)


# plt.xlabel('Freq(Hz)')
# plt.ylabel('Impedance(Ohm)')
# plt.plot(data_thp, label="THP")
# plt.plot(data_pha, label="PHA")
# plt.legend(loc='upper right')
# plt.grid()
# plt.show()



