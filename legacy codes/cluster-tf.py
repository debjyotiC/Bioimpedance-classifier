import json
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

with open('cluster_spec/clusterspec.json', 'r') as f:
    clusterspec = json.load(f)

cluster = tf.train.ClusterSpec(clusterspec)

a = tf.constant(3.0)
b = tf.constant(2.0)

with tf.device("/job:worker/task:0"):
    x = tf.add(a, b)
    y = tf.multiply(a, b)
    z = tf.add(x, y)

with tf.Session('grpc://192.168.0.104:2222', config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(z))

