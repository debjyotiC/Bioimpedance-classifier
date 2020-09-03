import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score

df = pd.read_csv('data-sets/three-stat.csv').dropna().reset_index(drop=True)

x = df.drop(columns=['S No.', 'Value', 'Label']).to_numpy(dtype=float)
y = df['Label'].to_numpy(dtype=float)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=False, test_size=0.4)


model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda seq: seq / (10 ** 5)),
    tf.keras.layers.Dense(100, input_dim=3, activation='softmax'),
    tf.keras.layers.Dense(50, activation='softmax'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), metrics=['accuracy'])


model_out = model.fit(x_train, y_train, epochs=200, validation_data=[x_test, y_test])

print("Training accuracy: {:.5f}".format(np.mean(model_out.history['accuracy'])))
print("Validation accuracy: {:.5f}".format(np.mean(model_out.history['val_accuracy'])))

y_prediction = model.predict(x_test)
rounded = [round(x[0]) for x in y_prediction]
y_cap_prediction = np.array(rounded, dtype='int64')

print(confusion_matrix(y_test, y_cap_prediction))

print("ANN Precision score:{}".format(precision_score(y_test, y_cap_prediction)))
print("ANN Accuracy score:{}".format(accuracy_score(y_test, y_cap_prediction)))
