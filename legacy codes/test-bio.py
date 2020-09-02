import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
tf.disable_v2_behavior()

df_train = pd.read_csv('../data-sets/data-gen.csv')

data_thp_100 = df_train['100_THP']
data_thp_30 = df_train['30_THP']

data_thp_10 = df_train['15_THP']
data_pha_100 = df_train['PHA']

test = df_train['10_THP']

data_train = np.array([data_thp_100, data_thp_30, data_thp_10, data_pha_100], dtype=float)
data_label = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)
data_test = np.array([test], dtype=float)

label_list = ['Cancerous', 'Non-cancerous']

# parameters
learning_rate = 0.3
epochs = 20

# data place holders
input_data = tf.placeholder(tf.float32, [None, 100], name='input')
output_data = tf.placeholder(tf.float32, [4, ], name='output')

# l0 layer
input_no = 100
w = tf.Variable(tf.random_normal([100, 80], stddev=0.03), name='w')
b = tf.Variable(tf.random_normal([80]), name='b')
output_nn_0 = tf.nn.softmax(tf.add(tf.matmul(input_data, w), b))

# l1 layer
neurons_l1 = 80
w1 = tf.Variable(tf.random_normal([80, 50], stddev=0.03), name='w1')
b1 = tf.Variable(tf.random_normal([50]), name='b1')
output_nn_1 = tf.nn.softmax(tf.add(tf.matmul(output_nn_0, w1), b1))

# l2 layer
neurons_l2 = 50
output_no = 2
w2 = tf.Variable(tf.random_normal([50, 2], stddev=0.03), name='w2')
b2 = tf.Variable(tf.random_normal([2]), name='b2')
output_nn = tf.nn.softmax(tf.add(tf.matmul(output_nn_1, w2), b2))

# error calculation
y_clipped = tf.clip_by_value(output_nn, 1e-10, 0.9999999)
error = -tf.reduce_mean(tf.reduce_sum(output_nn * tf.log(y_clipped) + (1 - output_nn) * tf.log(1 - y_clipped), axis=1))

# accuracy
accuracy = tf.metrics.accuracy(tf.argmax(output_data, 1), tf.argmax(output_nn, 1))


optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(error)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for epoch in range(epochs):
        _, c = sess.run([optimiser, error], feed_dict={input_data: data_train, output_data: data_label})
        print("{}/{}========={:.5f}".format(epoch+1, epochs, c))
    # print(sess.run([output_nn[0], tf.argmax(output_nn[0])], feed_dict={input_data: data_test}))
    print(sess.run(tf.gather(label_list, tf.argmax(output_nn, 1)), feed_dict={input_data: data_test}))


