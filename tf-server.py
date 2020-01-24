from flask import Flask, render_template
from flask_restful import Resource, Api, reqparse
import secrets
import numpy as np
import tensorflow.compat.v1 as tf
import os

tf.disable_v2_behavior()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sess = tf.Session()

app = Flask(__name__)
api = Api(app)

api_key_write = secrets.token_urlsafe(16).upper()  # generate write API key


saver = tf.train.import_meta_graph('model_save/model.meta')
saver.restore(sess, tf.train.latest_checkpoint('model_save'))

graph = tf.get_default_graph()
input_data = graph.get_tensor_by_name("input:0")
op_to_restore = graph.get_tensor_by_name("output_layer:0")


class AddData(Resource):
    @staticmethod
    def get():
        parser = reqparse.RequestParser()
        parser.add_argument('api_key', type=str)
        parser.add_argument('field', type=float)
        data = parser.parse_args()
        if str(data['api_key']).upper() == api_key_write:
            data = data['field']
            feed_dict = {input_data: [[data]]}
            out_put = sess.run(op_to_restore, feed_dict)[0][0]
            forresponse = out_put.item()
            return {"data": forresponse}
        else:
            return {"data": 'insertion error'}


api.add_resource(AddData, '/update', endpoint='update')

if __name__ == '__main__':
    print(api_key_write)
    app.run(host='192.168.26.18', port=5000)
