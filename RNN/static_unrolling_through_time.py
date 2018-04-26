import tensorflow as tf
import numpy as np

#Fixed input size
n_inputs=3
n_neurons=5

X0=tf.placeholder(tf.float32,[None,n_inputs])
X1=tf.placeholder(tf.float32,[None,n_inputs])
X2=tf.placeholder(tf.float32,[None,n_inputs])

basic_cell=tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs,states=tf.contrib.rnn.static_rnn(basic_cell,[X0,X1,X2],dtype=tf.float32)

Y0,Y1,Y2=output_seqs

init=tf.global_variables_initializer()

with tf.Session() as sess:
	init.run()
	X0_batch=np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
	X1_batch=np.array([[12,13,14],[15,16,17],[18,19,20],[21,22,23]])
	X2_batch=np.array([[24,25,26],[27,28,29],[30,31,32],[33,34,35]])
	Y0_val,Y1_val,Y2_val=sess.run([Y0,Y1,Y2],feed_dict={X0:X0_batch,X1:X1_batch,X2:X2_batch})
	print "Time 0 : "
	print Y0_val
	print "Time 1 : "
	print Y1_val
	print "Time 2 : "
	print Y2_val
	print states