# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: tiangc <tianguicheng@chuangxin.com>
# Date:   2018/11/16
import tensorflow as tf
import math
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

data_dir = "/mnt/hdfs/data/mnist-data"
mnist = input_data.read_data_sets(data_dir, one_hot=True)
print("len of train images: ", len(mnist.train.images))
checkpoint_dir = "hdfs://phoenix-001.phoenix.com:8020/test/ckpt/"

IMAGE_PIXELS = 28
hid_w = tf.Variable(
    tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, 100],
                        stddev=1.0 / IMAGE_PIXELS), name='hid_w')
hid_b = tf.Variable(tf.zeros([100]), name='hid_b')

sm_w = tf.Variable(tf.truncated_normal([100, 10],
                                       stddev=1.0 / math.sqrt(
                                           100)),
                   name='sm_w')
sm_b = tf.Variable(tf.zeros([10]), name='sm_b')

x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
hid = tf.nn.relu(hid_lin)

y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
pre = tf.arg_max(y, dimension=1)

with tf.Session() as sess:
    restorer = tf.train.Saver()
    check_point = tf.train.get_checkpoint_state(checkpoint_dir)
    if not check_point:
        print("ckpt is none.")
    restorer.restore(sess, check_point.model_checkpoint_path)
    pre_ = sess.run(pre, feed_dict={x: mnist.validation.images})
    print("predict: ", len(pre_))
    print(np.sum((pre_ == np.argmax(mnist.validation.labels, axis=1))))
