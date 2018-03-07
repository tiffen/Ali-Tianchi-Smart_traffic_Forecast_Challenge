# -*- coding: utf-8 -*-

"""
Created on Fri Jun  9 17:34:36 2017
@author: 李雲
前馈神经网络
使用 dropout 方法，relu 激活函数
"""

import tensorflow as tf

hidden1=100
hidden2=100

def inference(x, keep_prob, input_node, output_node, regularizer=None):
    
    with tf.variable_scope('layer1'):
        x=tf.nn.dropout(x,keep_prob[0])
        weights=tf.get_variable('weights',shape=[input_node,hidden1],initializer=tf.truncated_normal_initializer(stddev=0.1),dtype=tf.float32)
        biases=tf.get_variable('biases',shape=[hidden1],initializer=tf.zeros_initializer,dtype=tf.float32)
        layer1=tf.nn.elu(tf.matmul(x,weights)+biases)
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(weights))
    
    with tf.variable_scope('layer2'):
        layer1=tf.nn.dropout(layer1,keep_prob[1])
        weights=tf.get_variable('weights',shape=[hidden1,hidden2],initializer=tf.truncated_normal_initializer(stddev=0.1),dtype=tf.float32)
        biases=tf.get_variable('biases',shape=[hidden2],initializer=tf.zeros_initializer,dtype=tf.float32)
        layer2=tf.nn.elu(tf.matmul(layer1,weights)+biases)
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(weights))

    with tf.variable_scope('layer3'):
        layer2=tf.nn.dropout(layer2,keep_prob[2])
        weights=tf.get_variable('weights',shape=[hidden2,output_node],initializer=tf.truncated_normal_initializer(stddev=0.1),dtype=tf.float32)
        biases=tf.get_variable('biases',shape=[output_node],initializer=tf.zeros_initializer,dtype=tf.float32)
        output=tf.nn.relu(tf.matmul(layer2,weights)+biases)
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(weights))
        
    return output















































