import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

celsius_q = np.array([[-40.], [-10.], [0.], [8.], [15.], [22.], [38.]], dtype=float)
fahrenheit_a = np.array([[-40.], [14.], [32.], [46.], [59.], [72.], [100.]], dtype=float)

# parameters
learning_rate = 0.1
epochs = 500
batch_size = 2


# data place holders
input_data = tf.placeholder(tf.float32, [None, 1], name='b')
output_data = tf.placeholder(tf.float32, [None, 1], name='b')

# l0 layer
w0 = tf.Variable(tf.random_normal([1, 1], stddev=0.03), name='w0')
b0 = tf.Variable(tf.random_normal([1]), name='b0')
output_nn = tf.add(tf.matmul(input_data, w0), b0)


# error and optimization
error = tf.reduce_mean((output_data - output_nn)**2)
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



