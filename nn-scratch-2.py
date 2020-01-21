import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

celsius_q = np.array([[-40.], [-10.], [0.], [8.], [15.], [22.], [38.]], dtype=float)
fahrenheit_a = np.array([[-40.], [14.], [32.], [46.], [59.], [72.], [100.]], dtype=float)

# parameters
learning_rate = 0.1
epochs = 400
batch_size = 2

# data place holders
input_data = tf.placeholder(tf.float32, [None, 1], name='b')
output_data = tf.placeholder(tf.float32, [None, 1], name='b')

# l1 layer
n_features = 1
n_neurons_in_h1 = 1
w1 = tf.Variable(tf.random_normal([n_features, n_neurons_in_h1], stddev=0.03), name='w1')
b1 = tf.Variable(tf.random_normal([n_neurons_in_h1]), name='b1')
output_nn_1 = tf.add(tf.matmul(input_data, w1), b1)

# l2 layer
n_neurons_in_h2 = 1
w2 = tf.Variable(tf.random_normal([n_neurons_in_h1, n_neurons_in_h2], stddev=0.03), name='w2')
b2 = tf.Variable(tf.random_normal([n_neurons_in_h1]), name='b2')
output_nn_2 = tf.add(tf.matmul(output_nn_1, w2), b2)

# l3 layer
n_neurons_in_h2 = 1
n_classes = 1
w3 = tf.Variable(tf.random_normal([n_neurons_in_h2, n_classes], stddev=0.03), name='w3')
b3 = tf.Variable(tf.random_normal([n_neurons_in_h1]), name='b3')
output_nn = tf.add(tf.matmul(output_nn_2, w3), b3, name='output_layer')

# error and optimization
error = tf.reduce_mean((output_data - output_nn) ** 2)
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(error)

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(output_data, 1), tf.argmax(output_nn, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    total_batch = int(len(fahrenheit_a) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            _, c = sess.run([optimiser, error], feed_dict={input_data: [celsius_q[i]], output_data: [fahrenheit_a[i]]})
        avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
    print(sess.run(output_nn, feed_dict={input_data: [[100.00]]}))

