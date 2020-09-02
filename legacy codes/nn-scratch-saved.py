import tensorflow.compat.v1 as tf
import os
import numpy as np
import pandas as pd

tf.disable_v2_behavior()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sess = tf.Session()

df_train = pd.read_csv('../data-sets/gf_calculation.csv')

# impedance data
data_air = df_train['Air']
data_water = df_train['Water']

# First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('../model_save_2/model.meta')
saver.restore(sess, tf.train.latest_checkpoint('../model_save_2'))

# Now, let's access and create placeholders variables and
# create feed-dict to feed new data
graph = tf.get_default_graph()
input_data = graph.get_tensor_by_name("input:0")


# Now, access the op that you want to run.
op_to_restore = graph.get_tensor_by_name("output_layer:0")

feed_dict = {input_data: [data_water]}
out_put = sess.run(op_to_restore, feed_dict)
print(out_put)
