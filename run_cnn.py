import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from sklearn.cross_validation import train_test_split
import scipy.io


X = scipy.io.loadmat('images.mat')
X = np.transpose(X['images'])
X = np.reshape(X,(X.shape[0],28,28,1))
y = scipy.io.loadmat('labels.mat')
y = y['labels']
y = np.reshape(y,(y.shape[0]))
y[y==8] = 0 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X=None
y=None

filt_1 = [32,3]
filt_2 = [32,3]
no_channels = 1
num_fc1 = 128
learning_rate = 0.01
N=8437
batch_size = 100

x_ = tf.placeholder("float", shape=[None,X_train.shape[1],X_train.shape[2],no_channels], name = 'Input_data')
y_ = tf.placeholder(tf.int64, shape=[None], name = 'Ground_truth')

def weight_variable(shape, name):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name = name)

def bias_variable(shape, name):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name = name)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxpool(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='VALID')



with tf.name_scope("Conv1") as scope:
	W_conv1 = weight_variable([filt_1[1], filt_1[1], no_channels, filt_1[0]], 'Conv_Layer_1')
	b_conv1 = bias_variable([filt_1[0]], 'bias_for_Conv_Layer_1')
	a_conv1 = tf.add(tf.nn.conv2d(x_,W_conv1,strides=[1,1,1,1],padding='VALID'),b_conv1)
	
with tf.name_scope("MaxPool1") as scope:
	a_conv1_mp = maxpool(a_conv1)

with tf.name_scope("Conv2") as scope:
	W_conv2 = weight_variable([filt_2[1], filt_2[1], filt_1[0], filt_2[0]], 'Conv_Layer_2')
	b_conv2 = bias_variable([filt_2[0]], 'bias_for_Conv_Layer_2')
	a_conv2 = tf.add(tf.nn.conv2d(a_conv1_mp,W_conv2,strides=[1,1,1,1],padding='VALID'),b_conv2)

with tf.name_scope("MaxPool2") as scope:
	a_conv2_mp = maxpool(a_conv2)
	
with tf.name_scope("FC1") as scope:
	a_conv2_mp_flat = tf.reshape(a_conv2_mp,[-1,5*5*32])
 	W_fc = weight_variable([800,num_fc1], 'FC_1')
 	b_fc = bias_variable([num_fc1], 'bias_for_FC_1')
 	a_fc = tf.nn.sigmoid(tf.add(tf.matmul(a_conv2_mp_flat,W_fc),b_fc))

keep_prob = tf.placeholder(tf.float32)
# bn_train = tf.placeholder(tf.bool) 


with tf.name_scope("Readout") as scope:
	a_fc_drop = tf.nn.dropout(a_fc, keep_prob)
	W_op = weight_variable([num_fc1,1], 'Weights_Readout')
 	b_op = bias_variable([1], 'bias_for_Readout')
 	a_op = tf.nn.sigmoid(tf.add(tf.matmul(a_fc_drop,W_op),b_op))

with tf.name_scope("Loss") as scope:
	loss = tf.square(a_op-tf.to_float(y_))
	cost = tf.reduce_mean(loss)
	loss_summ = tf.scalar_summary('squared loss', cost)

with tf.name_scope("train") as scope:
    tvars = tf.trainable_variables()
    grads = tf.gradients(cost, tvars)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = zip(grads, tvars)
    train_step = optimizer.apply_gradients(gradients)

with tf.name_scope("Evaluating_accuracy") as scope:
    correct_prediction = tf.equal(tf.to_int64(a_op>0.5), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy_summary = tf.scalar_summary("accuracy", accuracy)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	for i in range(1000):
		batch_ind = np.random.choice(N,batch_size,replace=False)
		sess.run(train_step,feed_dict={x_:X_train[batch_ind], y_: y_train[batch_ind], keep_prob:0.5})
		print('Running Iteration %s : ',i)
		if i%10 == 0:
			result = sess.run([accuracy],feed_dict = { x_: X_train, y_: y_train, keep_prob:1.0})
			print ('Accuracy on Train :%s',result[0])
			result = sess.run([accuracy],feed_dict = { x_: X_test, y_: y_test, keep_prob:1.0})
			print ('Accuracy on Test :%s',result[0])




