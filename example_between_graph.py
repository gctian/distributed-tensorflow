# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: tiangc <tianguicheng@chuangxin.com>
# Date:   2018/11/13


import tensorflow as tf
import numpy as np
from datetime import datetime
import os, tempfile
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Define parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 0.03, 'Initial learning rate.')
tf.app.flags.DEFINE_integer('steps_to_validate', 1000, 'Steps to validate and print loss')
# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_bool("is_sync", False, "using synchronous training or not")
tf.app.flags.DEFINE_integer("num_workers", 2, "number of workers")
# Hyperparameters
learning_rate = FLAGS.learning_rate
steps_to_validate = FLAGS.steps_to_validate

ps_hosts = FLAGS.ps_hosts.split(",")
worker_hosts = FLAGS.worker_hosts.split(",")
cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
# job_name, task_index 通过命令行传入
server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

train_X = np.random.rand(100).astype(np.float32).reshape(-1)
train_Y = 2 * train_X + 10  # W=2, b=10

if FLAGS.job_name == "ps":
    # 如果是ps任务，程序就join到这里，作为参数更新的服务，等待其他worker节点来提交参数更新的梯度
    server.join()
elif FLAGS.job_name == "worker":
    # replica_device_setter， 会自动分配参数到ps节点，默认是轮流分配
    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):
        # 这里的global_step的值，是可以所有计算节点共享的， 在执行optimizer的minimize的时候，
        # 会自动加1， 所以可以通过这个可以知道所有的计算节点一共计算了多少步了
        global_step = tf.Variable(0, name='global_step', trainable=False)

        X = tf.placeholder(tf.float32)
        y = tf.placeholder(tf.float32)

        weight = tf.get_variable("weight", [1], tf.float32,
                                 initializer=tf.random_normal_initializer())
        biase = tf.get_variable("biase", [1], tf.float32,
                                initializer=tf.random_normal_initializer())
        pred = tf.multiply(X, weight) + biase

        loss_value = tf.reduce_mean(tf.square(y - pred))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # The StopAtStepHook handles stopping after running given steps.
        # 设置停止条件，在2000步之后自动停止，可以设置多个停止条件
        hooks = [tf.train.StopAtStepHook(num_steps=2000)]
        # 同步训练
        if FLAGS.is_sync:
            # ref https://www.tensorflow.org/api_docs/python/tf/train/SyncReplicasOptimizer
            optimizer = tf.train.SyncReplicasOptimizer(optimizer,
                                                       replicas_to_aggregate=FLAGS.num_workers,
                                                       total_num_replicas=FLAGS.num_workers)
            # create the hook which handles initialization and queues.
            hooks.append(optimizer.make_session_run_hook(FLAGS.task_index == 0))
        train_op = optimizer.minimize(loss_value, global_step=global_step)

        # 设置GPU使用显存比例，这里禁止使用GPU，全在CPU上操作
        gpu_fraction = 0.1
        gpu_options = tf.GPUOptions(
                per_process_gpu_memory_fraction=gpu_fraction)
        config = tf.ConfigProto(gpu_options=gpu_options)
        """
        1. The MonitoredTrainingSession takes care of session initialization, restoring 
        from a checkpoint, saving to a checkpoint, and closing when done or an error occurs.
        2. 默认第0个worker是chief worker, 非chief_worker会等待chief_worker完成初始化操作,
        制定task_index为0的任务为主任务，用于负责变量初始化、做checkpoint、保存summary和复原
        2. 这种情况下是异步执行的，每一个step打印一下信息会发现，step是交叉执行的，一个worker训练完就开始更新
        创建session时，用的是server.target, 说明是图间复制模式
        3. chief worker负责模型参数初始化等工作，这个过程中，其他worker节点要等worker节点完成初始化工作，才开始跑计算，
            可以看到这里并没有执行sess.run(init_op)类似操作
        """
        # 如果ps节点不在同一个机器上，那么ckpt_dir必须用hdfs路径，这样才可以共享
        ckpt_dir = "hdfs://phoenix-001.phoenix.com:8020/test/ckpt/"
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(FLAGS.task_index == 0),
                                               checkpoint_dir=ckpt_dir,
                                               save_checkpoint_secs=60,
                                               hooks=hooks) as mon_sess:
            
            while not mon_sess.should_stop():
                # mon_sess.run handles AbortedError in case of preempted PS.
                _, loss, step = mon_sess.run([train_op, loss_value, global_step],
                                             feed_dict={X: train_X,
                                                        y: train_Y})
                # 这里为了测试同步、异步的效果，设置每个step都打印
                # if step % steps_to_validate == 0:
                w, b = mon_sess.run([weight, biase])
                print("time: %s, step: %d, weight: %f, biase: %f, loss: %f" % (
                        str(datetime.now()), step, w, b, loss))