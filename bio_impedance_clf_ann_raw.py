import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score

df = pd.read_csv('data-sets/impedance-transpose.csv').dropna().reset_index(drop=True)

x = df.drop(columns=['Frequency', 'Label']).to_numpy(dtype='float64')
y = df['Label'].to_numpy(dtype='float64')

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=False, test_size=0.4)


model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda seq: seq / (10 ** 5)),
    tf.keras.layers.Dense(200, input_dim=100, activation='softmax'),
    tf.keras.layers.Dense(170, activation='relu'),
    tf.keras.layers.Dense(150, activation='relu'),
    tf.keras.layers.Dense(90, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

model_out = model.fit(x_train, y_train, epochs=170, validation_data=[x_test, y_test])

model.save('saved_model/tf_model_2')

print("Training accuracy: {:.5f}".format(np.mean(model_out.history['accuracy'])))
print("Validation accuracy: {:.5f}".format(np.mean(model_out.history['val_accuracy'])))

y_prediction = model.predict(x_test)
rounded = [round(x[0]) for x in y_prediction]
y_cap_prediction = np.array(rounded, dtype='int64')

print(confusion_matrix(y_test, y_cap_prediction))

print("ANN Precision score:{}".format(precision_score(y_test, y_cap_prediction)))
print("ANN Accuracy score:{}".format(accuracy_score(y_test, y_cap_prediction)))

epoch_count = range(1, len(model_out.history['loss']) + 1)

# save performance data
values = {'Epoch': epoch_count, 'Loss': model_out.history['loss'], 'Accuracy': model_out.history['accuracy']}
df_w = pd.DataFrame(values, columns=['Epoch', 'Loss', 'Accuracy'])
df_w.to_csv("data-sets/results-2.csv", index=None, header=True)

fig, axs = plt.subplots(2, 1)
# plot loss
axs[0].plot(model_out.history['loss'], color='Green')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].grid(True)

# plot accuracy
axs[1].plot(model_out.history['accuracy'], color='Red')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].grid(True)
plt.show()
