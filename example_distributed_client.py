# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: tiangc <tianguicheng@chuangxin.com>
# Date:   2018/11/12

import tensorflow as tf
from tensorflow.python.client import timeline


with tf.device('/job:ps/task:0/cpu:0'):
    input_data = tf.Variable(
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]],
            name="input_data")
    b = tf.Variable([[1.], [1.], [2.]], name="w")

inputs = tf.split(input_data, 2)
outputs = []

# Track statistics of the run using Timeline
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

'''
图内复制模式
只有worke0 (localhost:2223)会创建client
'''
with tf.Session("grpc://localhost:2223") as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 2 means 2 workers
    for i in range(2):
        with tf.device("/job:worker/task:%d/cpu:0" % i):
            print(sess.run(inputs[i]))
            outputs.append(tf.matmul(inputs[i], b))
    with tf.device('/job:ps/task:0/cpu:0'):
        output = tf.concat(outputs, axis=0)
        print(sess.run(output, options=run_options, run_metadata=run_metadata))

    # for tensorboard
    tf.summary.FileWriter("logs/", sess.graph)

    # Create timeline and write it to a json file
    tl = timeline.Timeline(step_stats=run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timeline_client.json', 'w') as f:
        f.write(ctf)