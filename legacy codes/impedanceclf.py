import pandas as pd
import numpy as np
import tensorflow as tf
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

df_dataset = pd.read_csv('../data-sets/data-gen.csv')

data_thp_100 = [np.max(df_dataset['100_THP']), np.min(df_dataset['100_THP']), np.average(df_dataset['100_THP'])]
data_thp_70 = [np.max(df_dataset['70_THP']), np.min(df_dataset['70_THP']), np.average(df_dataset['70_THP'])]

data_thp_30 = [np.max(df_dataset['30_THP']), np.min(df_dataset['30_THP']), np.average(df_dataset['30_THP'])]
data_thp_10 = [np.max(df_dataset['10_THP']), np.min(df_dataset['10_THP']), np.average(df_dataset['10_THP'])]

train_examples = np.array([data_thp_100, data_thp_10]).astype(np.float32)
print(train_examples.shape)
train_labels = np.array([[0.0], [1.0]]).astype(np.float32)
print(train_labels.shape)

test_examples = np.array(data_thp_30).astype(np.float32)
test_labels = np.array([1.0]).astype(np.float32)

train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
# test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, activation='sigmoid'),
    tf.keras.layers.Dense(50, activation='sigmoid'),
    tf.keras.layers.Dense(2)
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset, epochs=20, verbose=True)