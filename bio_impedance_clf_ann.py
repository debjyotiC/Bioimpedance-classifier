import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
df = pd.read_csv('data-sets/three-stat.csv').dropna()\
    .reset_index(drop=True)

x = df.drop(columns=['S No.', 'Value', 'Label'])
y = df['Label']

x_train, x_test, y_train, y_test = train_test_split(x, y)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1024, input_dim=3, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.RMSprop(0.0001), metrics=['accuracy'])
model_out = model.fit(x_train, y_train, epochs=500, validation_data=[x_test, y_test])

print("Training accuracy: {}".format(np.mean(model_out.history['accuracy'])))
print("Validation accuracy: {}".format(np.mean(model_out.history['val_accuracy'])))