import json
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

with open('cluster_spec/clusterspec.json', 'r') as f:
    clusterspec = json.load(f)
cluster = tf.train.ClusterSpec(clusterspec)


features = tf.placeholder(tf.float32, [None, 3])
labels = tf.placeholder(tf.float32, [None, 1])

# random weights
W = tf.Variable([[10.0], [000.0], [0.200]], tf.float32)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    predict = tf.nn.sigmoid(tf.matmul(features, W))
    print(sess.run(predict, feed_dict={features: [[0, 1, 1]]}))
    lbls= [[0], [1], [1], [0]]
    print(sess.run(predict, feed_dict={features: [[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]], labels: lbls}))

    # error = labels - predict
    error = tf.reduce_mean((labels - predict)**2)
    # Training
    optimizer = tf.train.GradientDescentOptimizer(10)
    train = optimizer.minimize(error)
    for i in range(100):
        sess.run(train, feed_dict={features: [[0, 1, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]], labels: lbls})
        training_cost = sess.run(error, feed_dict={features: [[0, 1, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]], labels: lbls})
        classe = sess.run((labels-predict), feed_dict={features: [[0, 1, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]], labels: lbls})
        print('Training cost = ', training_cost, 'W = ', classe)
        print(sess.run(predict, feed_dict={features: [[0, 1, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]]}))

