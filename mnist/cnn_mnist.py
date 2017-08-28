from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)
def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features['x'], [-1,28,28,1])

    #Convolutional layer 1
    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 32,
        kernel_size = [5,5],
        padding = "same"
        activation = tf.nn.elu)

    #Add a max pooling layer 1
    pool1 = tf.layers.max_pooling2d(
        inputs = conv1, pool_size = [2,2], strides = 2)

    #Convolutional layer 2 and max pooling layer 2
    conv2 = tf.layers.conv2d(
        inputs = pool1,
        filters = 64,
        kernel_size = [5,5],
        padding = "same",
        activation = tf.nn.elu)
    pool2 = tf.layers.max_pooling2d(
        inputs = conv2, pool_size = [2,2], strides = 2)

    #Dense Layer (start using fully connected)
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    pool2_flatten = tf.reshape(pool2, [-1,7*7*64])
    dense = tf.layers.dense(inputs = pool2_flatten, units = 1024, activation = tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs= dense, rate = .5, training= is_training)

    #Logits layer fully connected forward then softmax
    logits = tf.layers.dense(inputs = dropout, units = 10)

    predictions = {
        "classes":tf.argmax(input = logits, axis = 1)
        "probabilities": tf.nn.softmax(logits, name = "softmax_tensor") 
    }
    #check if making prediction. if yes then no backwardprop
    if(mode == tf.estimator.ModeKeys.PREDICT):
        return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)

    #Calculate Loss func
    onehot_labels = tf.one_hot(indices = tf.cast(labels, tf.int32), depth = 10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits = logits)

    #train
    if(is_training):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1 = .9, beta2 = .995, epsilon=1e-07)
        train_op = optimizer.minimize(
            loss = loss
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



if __name__ == "__main__":
    tf.app.run()
