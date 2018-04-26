import tensorflow as tf
import numpy as np

#Fixed input size
n_inputs=3
n_neurons=5
n_steps=3

X=tf.placeholder(tf.float32,[None,n_steps,n_inputs])
seq_length=tf.placeholder(tf.float32,[None])

basic_cell=tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs,states=tf.nn.dynamic_rnn(basic_cell,X,dtype=tf.float32,sequence_length=seq_length)

seq_length_batch=np.array([3,2,3,3])

init=tf.global_variables_initializer()

with tf.Session() as sess:
	init.run()
	X_batch	=	np.array([	
								[[0,1,2],[12,13,14],[24,25,26]],	#Instance 0 for [t=0,t=1,t=2]
								[[3,4,5],[15,16,17],[0,0,0]],	#Instance 1 for [t=0,t=1,t=2]
								[[6,7,8],[18,19,20],[30,31,32]],	#Instance 2 for [t=0,t=1,t=2]
								[[9,10,11],[21,22,23],[33,34,35]],	#Instance 3 for [t=0,t=1,t=2]
				])
	[y_val,state_val]=sess.run([outputs,states],feed_dict={X:X_batch,seq_length:seq_length_batch})
	
	for i in range(len(y_val)):
		print "Instance : "+str(i)
		for j in range(len(y_val[0])):
			print "Time "+str(j)+" : "+str(y_val[i][j])
	print state_val