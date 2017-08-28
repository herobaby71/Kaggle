from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import _pickle as pickle

tf.logging.set_verbosity(tf.logging.INFO)

def getCIFAR10_test(path = "cifar-10-batches-py/test_batch"):
    with open(path, 'rb') as f:
        raw_data = pickle.load(f, encoding = 'bytes')
        X, y = np.array(raw_data[b'data'], dtype = np.float32)/255, np.array(raw_data[b'labels'])
        X = np.transpose(np.reshape(X, [-1,3,32,32]), (0,2,3,1))
        print(X.shape)
        return(X,y)
        
def getCIFAR10_train(path = "cifar-10-batches-py/"):
    train_data_set = np.zeros([50000, 3,32, 32], dtype = np.float32)
    train_labels = np.zeros([50000,], dtype= int)
    
    file_name = path + "data_batch_"
    for i in range(1,6):
        file = open(file_name + str(i), 'rb')
        raw_data = pickle.load(file, encoding='bytes')
        
        batch_labels = np.array(raw_data[b'labels'])
        batch_imgs= np.array(raw_data[b'data'], dtype = np.float32) / 255 #normalize
        batch_imgs = np.reshape(batch_imgs, [-1,3,32,32])

        #append
        m = len(batch_imgs)
        train_data_set[m*(i-1):m*i, :] = batch_imgs
        train_labels[m*(i-1):m*i] = batch_labels
        
        file.close()
    del batch_imgs,batch_labels
    train_data_set=train_data_set.transpose(0,2,3,1)
    labels_dict = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9:"truck"}
    return (train_data_set, train_labels, labels_dict)

def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features['x'], [-1,32,32,3])

    #Convolutional layer 1
    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 32,
        kernel_size = [5,5],
        padding = "same",
        activation = None)
    batnorm1  = tf.layers.batch_normalization(inputs = conv1)
    act1 = tf.nn.elu(batnorm1)
    #Add a max pooling layer 1
    pool1 = tf.layers.max_pooling2d(
        inputs = act1, pool_size = [2,2], strides = 2)

    #Convolutional layer 2 and max pooling layer 2
    conv2_branch1 = tf.layers.conv2d(
        inputs = pool1,
        filters = 48,
        kernel_size = [1,1],
        padding = "same",
        activation = None)
    batnorm2_branch1 = tf.layers.batch_normalization(inputs = conv2_branch1)
    act2_branch1 =  tf.nn.elu(batnorm2_branch1)
    
    conv2_branch2 = tf.layers.conv2d(
        inputs = act2_branch1,
        filters = 64,
        kernel_size = [5,5],
        padding = "same",
        activation = None)
    batnorm2_branch2 = tf.layers.batch_normalization(inputs = conv2_branch2)
    act2_branch2 = tf.nn.elu(batnorm2_branch2)
    pool2 = tf.layers.max_pooling2d(
        inputs = act2_branch2, pool_size = [2,2], strides = 2)

    #Convolutional layer 3 -> batch normalization 3 -> activation 3-> maxpooling layer 3 
    conv3_branch1 = tf.layers.conv2d(
        inputs = pool2,
        filters = 96,
        kernel_size = [1,1],
        padding = "same",
        activation = None) 
    batnorm3_branch1 =  tf.layers.batch_normalization(inputs = conv3_branch1)
    act3_branch1 =  tf.nn.elu(batnorm3_branch1)
    
    conv3_branch2 = tf.layers.conv2d(
        inputs = act3_branch1,
        filters = 128,
        kernel_size = [5,5],
        padding = "same",
        activation = None)
    batnorm3_branch2 = tf.layers.batch_normalization(inputs = conv3_branch2)
    act3_branch2 =  tf.nn.elu(batnorm3_branch2)
    pool3 = tf.layers.max_pooling2d(
        inputs = act3_branch2, pool_size = [2,2], strides = 2)
    
    #Dense Layer (start using fully connected)
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    pool3_flatten = tf.reshape(pool3, [-1,4*4*128])
    dense = tf.layers.dense(inputs = pool3_flatten, units = 2048, activation = tf.nn.elu)
    dropout = tf.layers.dropout(
        inputs= dense, rate = .4, training= is_training)
    
    #Logits layer fully connected forward then softmax
    logits = tf.layers.dense(inputs = dropout, units = 10)

    predictions = {
        "classes":tf.argmax(input = logits, axis = 1),
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
            loss = loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    eval_data, eval_labels = getCIFAR10_test()
    train_data, train_labels, label_name = getCIFAR10_train()
    
    #create the estimator
    cifar_classifier = tf.estimator.Estimator(
        model_fn = cnn_model_fn, model_dir = "/home/thanh1998tom/assignment1/Tensorflow/tmp")
    
    #Set up logging for predictions
    #Log the values in the "Softmax" tensor with label probabilities
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    
    #train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x= {"x": train_data},
        y = train_labels,
        batch_size = 100,
        num_epochs = None,
        shuffle = True
    )
    
    cifar_classifier.train(
        input_fn = train_input_fn,
        steps = 100000,
        hooks = [logging_hook]
    )
    
    #evaluate the test set
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = cifar_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x":test_data},
        num_epochs = 1,
        shuffle = False
    )
    pred_results = list(cifar_classifier.predict(input_fn = pred_input_fn))
    pred_classes = np.reshape(np.array([p["classes"] for p in  pred_results]), (10000,1))
    counts = np.reshape(np.arange(0,10000)+1, (10000,1))
    sol = np.append(counts, pred_classes, axis = 1)
    np.savetxt("mnist_test_pred.csv", sol,fmt = '%i', delimiter=',')


tf.app.run()
