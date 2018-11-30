# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: tiangc <tianguicheng@chuangxin.com>
# Date:   2018/11/6

# encoding:utf-8
import math
import tempfile
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

flags = tf.app.flags
IMAGE_PIXELS = 28
# 定义默认训练参数和数据路径
flags.DEFINE_string('data_dir', '/mnt/hdfs/data/mnist-data', 'Directory  for storing mnist data')
flags.DEFINE_integer('hidden_units', 100, 'Number of units in the hidden layer of the NN')
flags.DEFINE_integer('train_steps', 5000, 'Number of training steps to perform')
flags.DEFINE_integer('batch_size', 100, 'Training batch size ')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
# 定义分布式参数
# 参数服务器parameter server节点
flags.DEFINE_string('ps_hosts', '10.18.119.32:22221', 'Comma-separated list of hostname:port pairs')
# 两个worker节点
flags.DEFINE_string('worker_hosts', '10.18.119.32:22222, 10.18.119.32:22223',
                    'Comma-separated list of hostname:port pairs')
# 设置job name参数
flags.DEFINE_string('job_name', 'worker', 'job name: worker or ps')
# 设置任务的索引
flags.DEFINE_integer('task_index', 0, 'Index of task within the job')
# 选择异步并行，同步并行
flags.DEFINE_bool("issync", False, "是否采用分布式的同步模式")
flags.DEFINE_bool("is_train", True, "是否train")

FLAGS = flags.FLAGS


class MyStopAtStepHook(tf.train.StopAtStepHook):

    def after_create_session(self, session, coord):
        if self._last_step is None:
            global_step = session.run(self._global_step_tensor)
            self._last_step = global_step + self._num_steps
            print("now global_step is %d after create session, num_steps: %d, last_step:%d :"
                  % (global_step, self._num_steps, self._last_step))

    def after_run(self, run_context, run_values):
        global_step = run_values.results
        if global_step >= self._last_step:
            print("global_step is %d when stop." % global_step)
            run_context.request_stop()


# ref https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/tools/dist_test/python/mnist_replica.py
def main():

    if FLAGS.job_name is None or FLAGS.job_name == '':
        raise ValueError('Must specify an explicit job_name !')
    else:
        print( 'job_name : %s' % FLAGS.job_name)
    if FLAGS.task_index is None or FLAGS.task_index == '':
        raise ValueError('Must specify an explicit task_index!')
    else:
        print( 'task_index : %d' % FLAGS.task_index)

    ps_spec = FLAGS.ps_hosts.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')

    # 创建集群
    num_workers = len(worker_spec)
    cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == 'ps':
        # ps节点只需要管理TensorFlow中的变量，不需要执行训练的过程。
        # server.join()会一直停在这条语句上
        server.join()

    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    print("len of train images: ", len(mnist.train.images))
    worker_device = '/job:worker/task:%d/cpu:0' % FLAGS.task_index
    """
    1 tf.train.replica_device_setter函数会自动将所有的参数分配到参数服务器上，而
        计算分配到当前的计算服务器上 
    
    2. tf.train.replica_device_setter()会根据job名，将with内的Variable op放到ps tasks，
        将其他计算op放到worker tasks。默认分配策略是轮询
    """
    with tf.device(tf.train.replica_device_setter(
            cluster=cluster,
            worker_device=worker_device
    )):
        # global_step = tf.Variable(0, name='global_step', trainable=False)  # 创建纪录全局训练步数变量
        global_step = tf.train.get_or_create_global_step()

        hid_w = tf.Variable(tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
                                                stddev=1.0 / IMAGE_PIXELS), name='hid_w')
        hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name='hid_b')

        sm_w = tf.Variable(tf.truncated_normal([FLAGS.hidden_units, 10],
                                               stddev=1.0 / math.sqrt(FLAGS.hidden_units)), name='sm_w')
        sm_b = tf.Variable(tf.zeros([10]), name='sm_b')

        x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
        y_ = tf.placeholder(tf.float32, [None, 10])

        hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
        hid = tf.nn.relu(hid_lin)

        y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
        cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

        opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
        hooks = [MyStopAtStepHook(last_step=10000)]
        # 采用同步模式：
        if FLAGS.issync:
            print("is_sync:true")
            opt = tf.train.SyncReplicasOptimizer(opt,
                                                 replicas_to_aggregate=num_workers,
                                                 total_num_replicas=num_workers)
            # create the hook which handles initialization and queues.
            hooks.append(opt.make_session_run_hook(FLAGS.task_index == 0))

        train_step = opt.minimize(cross_entropy, global_step=global_step)
        train_dir = "hdfs://phoenix-001.phoenix.com:8020/test/ckpt"

        is_chief = (FLAGS.task_index == 0)
        if is_chief:
            print( 'Worker %d: Initailizing session...' % FLAGS.task_index)
        else:
            print( 'Worker %d: Waiting for session to be initaialized...' % FLAGS.task_index)

        local_step = 0
        best_val_loss = 10000.0
        time_begin = time.time()
        print('Traing begins @ %f' % time_begin)
        """
        if_chief: 设定task_index为0的任务为主任务，用于负责变量初始化、做checkpoint、保存summary和复原
           还负责输出日志和保存模型， 非chiefworker要等chief worker完成初始化后才可以执行计算

        """
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=is_chief,
                                               checkpoint_dir=train_dir,
                                               hooks=hooks) as mon_sess:
            while not mon_sess.should_stop():
                batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
                train_feed = {x: batch_xs, y_: batch_ys}

                _, step, loss = mon_sess.run([train_step, global_step, cross_entropy], feed_dict=train_feed)
                local_step += 1

                now = str(datetime.now())
                print('time: %s | worker: %d | traing step:%d | global step:%d | loss: %f' % (
                    now, FLAGS.task_index, local_step, step, loss))

                # 每隔1000步打印下验证集结果
                if (step+1) % 1000 == 0:
                    val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
                    val_xent = mon_sess.run(cross_entropy, feed_dict=val_feed) / len(mnist.validation.images)
                    if val_xent < best_val_loss:
                        best_val_loss = val_xent
                    print( 'At global step: %d, validation cross entropy = %g' % (step, best_val_loss))

        time_end = time.time()
        print( 'Training ends @ %f' % time_end)
        train_time = time_end - time_begin
        print( 'Worker %d | Training elapsed time: %f s | Train step: %d | best val loss: %f' %
               (FLAGS.task_index, train_time, local_step, best_val_loss))


if __name__ == '__main__':
    main()