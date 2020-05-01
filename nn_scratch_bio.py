import pandas as pd
import numpy as np
import os
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
df_train = pd.read_csv('data-sets/gf_calculation.csv')

# impedance data
data_air = df_train['Air']
data_water = df_train['Water']

data_train = np.array([data_air, data_water], dtype=float)
data_label = np.array([1.0, 2.0], dtype=float)

# parameters
learning_rate = 1e-2
epochs = 400


# data place holders
input_data = tf.placeholder(tf.float32, [None, 100], name='input')
output_data = tf.placeholder(tf.float32, [1, ], name='output')

# l1 layer
w1 = tf.Variable(tf.random_normal([100, 100], stddev=0.03), name='w1')
b1 = tf.Variable(tf.random_normal([100]), name='b1')
output_nn_1 = tf.nn.softmax(tf.add(tf.matmul(input_data, w1), b1))

# l2 layer
w2 = tf.Variable(tf.random_normal([100, 80], stddev=0.03), name='w2')
b2 = tf.Variable(tf.random_normal([80]), name='b2')
output_nn_2 = tf.nn.softmax(tf.add(tf.matmul(output_nn_1, w2), b2))

# l3 layer
w3 = tf.Variable(tf.random_normal([80, 50], stddev=0.03), name='w3')
b3 = tf.Variable(tf.random_normal([50]), name='b3')
output_nn_3 = tf.nn.softmax(tf.add(tf.matmul(output_nn_2, w3), b3))

# l4 layer
w4 = tf.Variable(tf.random_normal([50, 2], stddev=0.03), name='w4')
b4 = tf.Variable(tf.random_normal([2]), name='b4')
output_nn = tf.nn.softmax(tf.add(tf.matmul(output_nn_3, w4), b4), name='output_layer')

# error calculation
y_clipped = tf.clip_by_value(output_nn, 1e-10, 0.9999999)
error = -tf.reduce_mean(tf.reduce_sum(output_nn * tf.log(y_clipped) + (1 - output_nn) * tf.log(1 - y_clipped), axis=1))
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(error)

# define an accuracy assessment operation
output_max = tf.argmax(data_label, 0)
predicted_max = tf.argmax(output_nn, 1)
correct_prediction = tf.equal(output_max, predicted_max)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    total_batch = int(len(data_label))
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            _, c, output = sess.run([optimiser, error, output_nn], feed_dict={input_data: [data_train[i]],
                                                                              output_data: [data_label[i]]})
        avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), "Output:", output[0])
    print(sess.run(output_nn, feed_dict={input_data: [np.array(data_water)]}))
    save_path = saver.save(sess, "model_save_2/model")
    print(f"Model saved in path:{save_path}")
    tf.train.write_graph(sess.graph_def, 'model_save_2', 'save-tensor-2.pb')
