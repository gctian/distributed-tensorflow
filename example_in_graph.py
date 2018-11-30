# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: tiangc <tianguicheng@chuangxin.com>
# Date:   2018/11/12

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.python.client import timeline

tf.app.flags.DEFINE_string("ps_hosts", "localhost:2222", "ps hosts")
tf.app.flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224", "worker hosts")
tf.app.flags.DEFINE_string("job_name", "worker", "'ps' or'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS


def main():
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    '''
    下面两行代码，对于所有节点来说是一样的
    '''
    # create cluster, 创建集群信息
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    # create the server， 在当前节点上启动server，并传入集群信息，这样当前节点就可以和集群中的节点通信了
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

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

    if FLAGS.job_name == 'ps':
        server.join()
    else:
        # 图内复制，只在worker0上创建client
        with tf.Session("grpc://localhost:2223") as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            for i in range(2):
                with tf.device("/job:worker/task:%d/gpu:0" % i):
                    print("now is worker %d: " % i)
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


if __name__ == "__main__":
    main()