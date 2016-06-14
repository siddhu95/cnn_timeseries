import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import read_fmri_util as rf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from bnf import *
import sklearn as sk
import sys

def basic_CNN(lr_rate,num_filt_1,num_filt_2,num_fc_1,num_fc_2):
    
    """Hyperparameters"""
    filt_1 = [num_filt_1,5]     #Number of filters in first conv layer
    filt_2 = [num_filt_2,5]      #Number of filters in second conv layer
    num_fc_1 = num_fc_1       #Number of neurons in first fully connected layer
    num_fc_2 = num_fc_2       #Number or neurons in second fully connected layer
    max_iterations = 4000
    batch_size = 10
    dropout = 0.5       #Dropout rate in the fully connected layer
    plot_row = 5        #How many rows do you want to plot in the visualization
    learning_rate = lr_rate
    input_cent = True   # Do you want to center the x,y,z coordinates? 
    sl = 137           #sequence length
    ratio = 0.8         #Ratio for train-val split
    crd = 264             #How many coordinates you feed
    sl_pad = 2
    D = (sl+sl_pad-filt_1[1])/1+1              
    #Explanation on D: We pad the input sequence at the basket-side. There is more
    # information and we dont want to lose it in the border effect.
    # The /1 is when future implementation want to play with different strides
    plot_every = 100    #How often do you want terminal output for the performances



    """Load the data"""
    data,labels,p_id = rf.read_data('/home/siddhu/FBIRN/original_res/ROI_files/masked','/home/siddhu/FBIRN/original_res/mat_format',[3])
    print('We have %s observations with a sequence length of %s '%(data.shape[0],sl))
    #print('We have %s observations with a sequence length of %s '%(N,sl))

    #Demean the data conditionally
    if input_cent:
      data = rf.standardize(data)

    #Shuffle the data
    (X_train,X_val,y_train,y_val) = rf.random_split(data,labels,ratio=0.8)
    
    N = X_train.shape[0]
    Nval = X_val.shape[0]
    data = None  #we don;t need to store this big matrix anymore
    
    # Organize the classes
    num_classes = len(np.unique(y_train))
    base = np.min(y_train)  #Check if data is 0-based
    if base != 0:
        y_train -=base
        y_val -= base


    #For sanity check we plot a random collection of lines
    # For better visualization, see the MATLAB script in this project folder
    # if False:
    #     plot_basket(X_train,y_train)


    #Proclaim the epochs
    epochs = np.floor(batch_size*max_iterations / N)
    #print('Train with approximately %d epochs' %(epochs))

    # Nodes for the input variables
    x = tf.placeholder("float", shape=[None, crd,sl], name = 'Input_data')
    y_ = tf.placeholder(tf.int64, shape=[None], name = 'Ground_truth')
    keep_prob = tf.placeholder("float")
    bn_train = tf.placeholder(tf.bool)          #Boolean value to guide batchnorm

    # Define functions for initializing variables and standard layers
    #For now, this seems superfluous, but in extending the code
    #to many more layers, this will keep our code
    #read-able
    def weight_variable(shape, name):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial, name = name)

    def bias_variable(shape, name):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial, name = name)

    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


    with tf.name_scope("Reshaping_data") as scope:
      x_feed = tf.expand_dims(x,dim=3, name = 'x_feed')
      x_pad = tf.pad(x_feed,[[0,0],[0,0],[0,sl_pad],[0,0]])


    """Build the graph"""
    # ewma is the decay for which we update the moving average of the 
    # mean and variance in the batch-norm layers
    with tf.name_scope("Conv1") as scope:
      W_conv1 = weight_variable([crd, filt_1[1], 1, filt_1[0]], 'Conv_Layer_1')
      b_conv1 = bias_variable([filt_1[0]], 'bias_for_Conv_Layer_1')
      a_conv1 = tf.add(tf.nn.conv2d(x_pad,W_conv1,strides=[1,1,1,1],padding='VALID'),b_conv1)
      size1 = tf.shape(a_conv1)

    with tf.name_scope('Batch_norm_conv1') as scope:
    #    ewma = tf.train.ExponentialMovingAverage(decay=0.99)                  
    #    bn_conv1 = ConvolutionalBatchNormalizer(num_filt_1, 0.001, ewma, True)           
    #    update_assignments = bn_conv1.get_assigner() 
    #    a_conv1 = bn_conv1.normalize(a_conv1, train=bn_train) 
        a_conv1_bn = batch_norm(a_conv1,filt_1[0],bn_train,'bn1')
        h_conv1 = tf.nn.relu(a_conv1_bn)
        a_conv1_hist = tf.histogram_summary('a_conv1_bn',a_conv1_bn)
        a_conv1_hist1 = tf.histogram_summary('a_conv1',a_conv1)

    with tf.name_scope("Conv2") as scope:
      W_conv2 = weight_variable([1, filt_2[1], filt_1[0], filt_2[0]], 'Conv_Layer_2')
      b_conv2 = bias_variable([filt_2[0]], 'bias_for_Conv_Layer_2')
      a_conv2 = conv2d(h_conv1, W_conv2) + b_conv2

    with tf.name_scope('Batch_norm_conv2') as scope:
    #    bn_conv2 = ConvolutionalBatchNormalizer(num_filt_2, 0.001, ewma, True)           
    #    update_assignments = bn_conv2.get_assigner() 
    #    a_conv2 = bn_conv2.normalize(a_conv2, train=bn_train) 
        a_conv2 = batch_norm(a_conv2,filt_2[0],bn_train,'bn2')
        h_conv2 = tf.nn.relu(a_conv2)

    with tf.name_scope("Fully_Connected1") as scope:
      W_fc1 = weight_variable([D*filt_2[0], num_fc_1], 'Fully_Connected_layer_1')
      b_fc1 = bias_variable([num_fc_1], 'bias_for_Fully_Connected_Layer_1')
      h_conv2_flat = tf.reshape(h_conv2, [-1, D*filt_2[0]])
      h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

    with tf.name_scope("Fully_Connected2") as scope:
      W_fc2 = weight_variable([num_fc_1,num_fc_2], 'Fully_Connected_layer_2')
      b_fc2 = bias_variable([num_fc_2], 'bias_for_Fully_Connected_Layer_2')
      h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)  


    with tf.name_scope("Output") as scope:
        #postfix _o represent variables for output layer
      h_o_drop = tf.nn.dropout(h_fc2, keep_prob)
      W_o = tf.Variable(tf.truncated_normal([num_fc_2, 1], stddev=0.1),name = 'W_o')
      b_o = tf.Variable(tf.constant(0.1, shape=[1]),name = 'b_o')
      h_o = tf.matmul(h_o_drop, W_o) + b_o
      sm_o = tf.sigmoid(h_o)

    with tf.name_scope("Sigmoid") as scope:
        loss = tf.square(sm_o-tf.to_float(y_))
        cost = tf.reduce_mean(loss)
        loss_summ = tf.scalar_summary("cross entropy_loss", cost)
    with tf.name_scope("train") as scope:
        tvars = tf.trainable_variables()
        #We clip the gradients to prevent explosion
        grads = tf.gradients(cost, tvars)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients = zip(grads, tvars)
        train_step = optimizer.apply_gradients(gradients)
        # The following block plots for every trainable variable
        #  - Histogram of the entries of the Tensor
        #  - Histogram of the gradient over the Tensor
        #  - Histogram of the grradient-norm over the Tensor
        numel = tf.constant([[0]])
        for gradient, variable in gradients:
          if isinstance(gradient, ops.IndexedSlices):
            grad_values = gradient.values
          else:
            grad_values = gradient

          numel +=tf.reduce_sum(tf.size(variable))  

          h1 = tf.histogram_summary(variable.name, variable)
          h2 = tf.histogram_summary(variable.name + "/gradients", grad_values)
          h3 = tf.histogram_summary(variable.name + "/gradient_norm", clip_ops.global_norm([grad_values]))
        #tf.gradients returns a list. We cannot fetch a list. therefore we fetch the tensor that is the 0-th element of the list
        vis = tf.gradients(loss, x_feed)[0]
    with tf.name_scope("Evaluating_accuracy") as scope:
        correct_prediction = tf.equal(tf.argmax(h_o,1), y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        accuracy_summary = tf.scalar_summary("accuracy", accuracy)


    #Define one op to call all summaries    
    merged = tf.merge_all_summaries()

    # For now, we collect performances in a Numpy array.
    # In future releases, I hope TensorBoard allows for more
    # flexibility in plotting
    perf_collect = np.zeros((4,int(np.floor(max_iterations /100))))

    with tf.Session() as sess:
      writer = tf.train.SummaryWriter('/home/siddhu/FBIRN/cnn/log/', sess.graph)

      sess.run(tf.initialize_all_variables())

      step = 0      # Step is a counter for filling the numpy array perf_collect
      for i in range(max_iterations):
        batch_ind = np.random.choice(N,batch_size,replace=False)

        check = sess.run([size1],feed_dict={ x: X_val, y_: y_val, keep_prob: 1.0, bn_train : False})    
        #print check[0]

        if i==0:
            # Use this line to check before-and-after test accuracy
            result = sess.run(accuracy, feed_dict={ x: X_val, y_: y_val, keep_prob: 1.0, bn_train : False})
            acc_test_before = result
        if i%100 == 0:
          #Check training performance
          result = sess.run([accuracy,cost],feed_dict = { x: X_train, y_: y_train, keep_prob: 1.0, bn_train : False})
          perf_collect[0,step] = result[0] 
          perf_collect[1,step] = result[1]        

          #Check validation performance
          result = sess.run([accuracy,cost,merged], feed_dict={ x: X_val, y_: y_val, keep_prob: 1.0, bn_train : False})
          acc = result[0]
          perf_collect[2,step] = acc
          perf_collect[3,step] = result[1]

          #Write information to TensorBoard
          summary_str = result[2]
          writer.add_summary(summary_str, i)
          writer.flush()  #Don't forget this command! It makes sure Python writes the summaries to the log-file
          #print(" Validation accuracy at %s out of %s is %s" % (i,max_iterations, acc))
          step +=1
        sess.run(train_step,feed_dict={x:X_train[batch_ind], y_: y_train[batch_ind], keep_prob: dropout, bn_train : True})
      #In the next line we also fetch the softmax outputs 
      result = sess.run([accuracy,numel,sm_o, x_pad], feed_dict={ x: X_val, y_: y_val, keep_prob: 1.0, bn_train : False})
      acc_test = result[0]
    tf.reset_default_graph()
    return acc_test

def main(argv):
  lr_rate=float(argv[0])
  num_filt_1,num_filt_2,num_fc_1,num_fc_2 = map(lambda x:int(x),[x for x in argv[1:]])
  print (num_filt_1,num_filt_2,num_fc_1,num_fc_2,lr_rate)
  acc = basic_CNN(lr_rate,num_filt_1,num_filt_2,num_fc_1,num_fc_2)
  
  print acc

if __name__ == "__main__":
  main(sys.argv[1:])