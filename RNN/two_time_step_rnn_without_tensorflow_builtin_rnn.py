import numpy as np
import tensorflow as tf

#Fixed input size
n_inputs=3
n_neurons=5

X0=tf.placeholder(tf.float32,[None,	n_inputs])
X1=tf.placeholder(tf.float32,[None,	n_inputs])
X2=tf.placeholder(tf.float32,[None,	n_inputs])

Wx=tf.Variable(tf.random_normal(shape=[n_inputs,	n_neurons],dtype=tf.float32))
Wy=tf.Variable(tf.random_normal(shape=[n_neurons,n_neurons],dtype=tf.float32))
b=tf.Variable(tf.zeros([1,	n_neurons],	dtype=tf.float32))

#t=0, t=1 and t=3 i.e unrolled for 3 time steps
Y0=tf.tanh(tf.matmul(X0,Wx)+b)
Y1=tf.tanh(tf.matmul(Y0,Wy)+tf.matmul(X1,Wx)+b)
Y2=tf.tanh(tf.matmul(Y1,Wy)+tf.matmul(X2,Wx)+b)

init=tf.global_variables_initializer()

X0_batch=np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
X1_batch=np.array([[12,13,14],[15,16,17],[18,19,20],[21,22,23]])
X2_batch=np.array([[24,25,26],[27,28,29],[30,31,32],[33,34,35]])

with tf.Session() as sess:
	init.run()
	Y0_val,Y1_val,Y2_val=sess.run([Y0,Y1,Y2],feed_dict={X0:X0_batch,X1:X1_batch,X2:X2_batch})
	print "Time 0 : "
	print Y0_val
	print "Time 1 : "
	print Y1_val
	print "Time 2 : "
	print Y2_val

	