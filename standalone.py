# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: tiangc <tianguicheng@chuangxin.com>
# Date:   2018/11/29


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


def one_gpu():
    '''
    单机单卡
    对于单机单卡，可以把参数和定义都放在GPU上，不过如果模型较大参数较多，全放在GPU上显存不够时
    可以把参数定义放在CPU上，如下所示
    '''
    with tf.device("/cpu:0"):
        w = tf.Variable(tf.constant([[1.0, 2.0], [4.0, 5.0]]), name="w")
        b = tf.Variable(tf.constant([[1.0], [2.0]]), name="b")

    with tf.device("/gpu:0"):
        addwb = tf.add(w, b)
        mulwb = tf.matmul(w, b)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        val1, val2 = sess.run([addwb, mulwb])
        print(val1)
        print(val2)


def _allocate_variable(name, shape, initializer, dtype=tf.float32):
    # 分配变量，Tensorflow 会自动处理变量在不同设备间的通信问题，因而可以放在GPU上，也可以放在CPU上
    # 如果是单机单卡，都放在GPU上比较快 （无需显式指定device, tf自动分配即可)
    # 如果是单机多卡，则放在CPU上略快
    with tf.device('/cpu:0'): # 强制放在主内存上
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    print('%s: %s' % (var.op.name, var.device))
    return var

# 创建网络 y=xw+b
def tower(input_tensor, target_tensor, scope, dims=[]):
    for i, d in enumerate(dims):
        with tf.variable_scope('affine%d' % i) as varscope:  # 仅仅用于生成变量的全名，与存放设备无关
            w = _allocate_variable('w', shape=[input_tensor.get_shape()[1], d], initializer=tf.truncated_normal_initializer(0, 1));
            b = _allocate_variable('b', shape=[], initializer=tf.zeros_initializer);
        input_tensor = tf.matmul(input_tensor, w) + b
        input_tensor = tf.nn.relu(input_tensor)

    with tf.variable_scope('affine_last') as varscope:  # 仅仅用于生成变量的全名，与存放设备无关
#         w = _allocate_variable('w', shape=[input_tensor.get_shape()[1], 1], initializer=tf.truncated_normal_initializer(0, 1));
        w = _allocate_variable('w', shape=[input_tensor.get_shape()[1], 1], initializer=tf.constant_initializer(value=1));
        b = _allocate_variable('b', shape=[], initializer=tf.zeros_initializer);

    y = tf.matmul(input_tensor, w) + b
    l = tf.reduce_mean(tf.square(y - target_tensor))
    tf.add_to_collection('losses', l)
    return y, l

# 合并所有tower上的梯度，取平均， 对于单机多卡程序，这段代码是通用的
def average_tower_grads(tower_grads):
    print('towerGrads:')
    idx = 0
    for grads in tower_grads:  # grads 为 一个list，其中元素为 梯度-变量 组成的二元tuple
        print('grads---tower_%d' % idx)
        for g_var in grads:
            print(g_var)
            print('\t%s\n\t%s' % (g_var[0].op.name, g_var[1].op.name))
        idx += 1

    if(len(tower_grads) == 1):
        return tower_grads[0]
    avgGrad_var_s = []
    for grad_var_s in zip(*tower_grads):
        grads = []
        v = None
        for g, v_ in grad_var_s:
            g = tf.expand_dims(g, 0)
            grads.append(g)
            v = v_
        all_g = tf.concat(grads, 0)
        avg_g = tf.reduce_mean(all_g, 0, keep_dims=False)
        avgGrad_var_s.append((avg_g, v))
    return avgGrad_var_s

def generate_towers(NUM_GPU=2, dim_in=1, dims=None):
    if(dims is None): dims = []

    input_tensor = tf.placeholder(tf.float32, shape=[None, dim_in], name='input')
    target_tensor = tf.placeholder(tf.float32, shape=[None, dim_in], name='target')
    input_tensors = tf.split(input_tensor, NUM_GPU)  # batch_size必须可以被NUM_GPU整除
    target_tensors = tf.split(input_tensor, NUM_GPU)

    towerGrads = []
    lr = 1e-2
    opt = tf.train.GradientDescentOptimizer(lr)
    for i in range(NUM_GPU):
        if i == 0:
            with tf.device('/gpu:0'):
                with tf.name_scope('tower_%d' % i) as scope:
                    input_sub = input_tensors[i]
                    print("device:%s" % input_sub.device)
                    target_sub = target_tensors[i]
                    y, loss = tower(input_tensor=input_sub, target_tensor=target_sub, scope=scope, dims=dims)
                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()
                    grads = opt.compute_gradients(loss)
                    towerGrads.append(grads)
        else:
            with tf.device('/cpu:0'):
                with tf.name_scope('tower_%d' % i) as scope:
                    input_sub = input_tensors[i]
                    print("device:%s" % input_sub.device)
                    target_sub = target_tensors[i]
                    y, loss = tower(input_tensor=input_sub, target_tensor=target_sub, scope=scope, dims=dims)
                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()
                    grads = opt.compute_gradients(loss)
                    towerGrads.append(grads)
    avgGrad_var_s = average_tower_grads(towerGrads)
    apply_gradient_op = opt.apply_gradients(avgGrad_var_s, global_step=None)

    print('ALL variables:')
    for v in tf.global_variables():
        print('\t%s' % v.op.name)

    return input_tensor, target_tensor, y, loss, apply_gradient_op


if __name__ == '__main__':
    sess = tf.Session()
    NUM_GPU = 2  # 楼主机器只有一块GPU，这里NUM_GPU=1，计算过程都在GPU上, NUM_GPU=2，一半在GPU计算，一半在CPU计算
    dim_in = 2  # 输入变量x 的维度
    dims = [64, 32] #隐层单元数，设置为[]时表示 y=xw+b的线性变换，否则表示多层的全连接网络
    batch_size = 2000

    input_tensor, target_tensor, y, loss, apply_gradient_op = generate_towers(NUM_GPU=NUM_GPU, dim_in=dim_in, dims=dims)
    sess.run(tf.global_variables_initializer())

    inputs = np.random.rand(batch_size, dim_in)
    targets = inputs * 2 + 1
    feed_dict = {input_tensor: inputs, target_tensor: targets}

    import time
    tstart = time.time()
    for i in range(10000):
        _, l = sess.run([apply_gradient_op, loss], feed_dict=feed_dict)  #will print w, b
        # print(l)
    telapse = time.time() - tstart
    print(u'%d块GPU用时: %.2fs' % (NUM_GPU, telapse))