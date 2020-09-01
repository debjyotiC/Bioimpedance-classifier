import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import os

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

# test data
data_got = df_test['datagot']

data_train = np.array([data_thp_100, data_thp_80, data_thp_60, data_thp_50, data_thp_40, data_thp_30,
                       data_thp_20, data_thp_10, data_pha], dtype=float)
data_label = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=float)
print(len(data_train))
# parameters
learning_rate = 0.1
epochs = 5000
batch_size = 3

# data place holders
input_data = tf.placeholder(tf.float32, [None, 100], name='input')
output_data = tf.placeholder(tf.float32, [1, ], name='output')

# l1 layer
w1 = tf.Variable(tf.random_normal([100, 100], stddev=0.03), name='w1')
b1 = tf.Variable(tf.random_normal([100]), name='b1')
output_nn_1 = tf.add(tf.matmul(input_data, w1), b1)

# l2 layer
w2 = tf.Variable(tf.random_normal([100, 80], stddev=0.03), name='w2')
b2 = tf.Variable(tf.random_normal([80]), name='b2')
output_nn_2 = tf.add(tf.matmul(output_nn_1, w2), b2)

# l3 layer
w3 = tf.Variable(tf.random_normal([80, 20], stddev=0.03), name='w3')
b3 = tf.Variable(tf.random_normal([20]), name='b3')
output_nn_3 = tf.add(tf.matmul(output_nn_2, w3), b3)

# l4 layer
w4 = tf.Variable(tf.random_normal([20, 9], stddev=0.03), name='w4')
b4 = tf.Variable(tf.random_normal([9]), name='b4')
output_nn = tf.add(tf.matmul(output_nn_3, w4), b4)

# error calculation
error = tf.reduce_mean((output_data - output_nn) ** 2)
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(error)

# define an accuracy assessment operation
output_max = tf.argmax(output_data, 0)
predicted_max = tf.argmax(output_nn, 1)
correct_prediction = tf.equal(output_max, predicted_max)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    total_batch = int(len(data_label) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            _, c = sess.run([optimiser, error], feed_dict={input_data: [data_train[i]], output_data: [data_label[i]]})
        avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))

    