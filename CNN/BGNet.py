import tensorflow as tf
import numpy as np
import time
import os
import cv2
import glob
import math

class BGNet:
	def __init__(self):
		#path for the data
		self.train_path='train/'
		self.training_classes=['dogs','cats']
		self.test_path='test/*.jpg'
		self.img_width=64
		self.img_height=64
		
		#hyperparameters for bgnet
		self.epoch              = 50
		self.batch_size         = 32
		self.learning_rate      = 0.00005
		
		#parameter for bgnet
		self.RGB                = 3
		self.none               = -1
		self.last_axis_vertical = -1 
		self.pool_value         = 2
		self.kernel_size        = [3,3]
		self.filter_sizes       = [32,64,128,256]
		self.conv_stride        = 1
		self.keep_prob          = 0.5
		self.input              = [self.none,self.img_width,self.img_height,self.RGB]
		self.pool_size          = [self.pool_value,self.pool_value]
		self.pool_stride        = (self.pool_value ,self.pool_value)
		#self.fc_units           = 23*23*self.filter_sizes[len(self.filter_sizes)-1]
		#self.flatten_dimensions = [self.last_axis_vertical,self.fc_units]
		self.output_units       = 2
		self.end_batch          = self.batch_size 
		
		print "\nInitialization Completed\n" 

	def load_images(self,path):
		img = cv2.imread(path)
		resized = cv2.resize(img,(self.img_width,self.img_height),interpolation=cv2.INTER_LINEAR)
		return resized
	
	def load_train(self):
		X_train=[]
		X_train_id=[]
		y_train=[]
		start_time=time.time()
		print "Loading training images..."
		folders=[self.training_classes[0],self.training_classes[1]]
		for fld in folders:
			index=folders.index(fld)
			print '     Loading {} files - Class {}'.format(fld, index)
			path=os.path.join(self.train_path,fld,'*g')
			files=glob.glob(path)
			for fl in files:
				flbase=os.path.basename(fl)
				img=self.load_images(fl)
				X_train.append(img)
				if(index==0):
					y_train.append([1,0])
				else:
					y_train.append([0,1])
		print '     Training data load time: {} seconds\n'.format(round(time.time()-start_time, 2))
		return X_train,y_train
	
	def load_test(self):
		files=sorted(glob.glob(self.test_path))
		print 'Loaded {} test images...'.format(len(files))
		X_test=[]
		X_test_id=[]
		start_time=time.time()
		for fl in files:
			flbase=os.path.basename(fl)
			img=self.load_images(fl)
			X_test.append(img)
		print '     Testing data load time: {} seconds\n'.format(round(time.time()-start_time, 2))
		return X_test
	
	def batch_data(self,step,data,labels):
		length=len(data)
		self.end_batch
		data_batch=[]
		labels_batch=[]
		if step==0:
			self.end_batch=self.batch_size-1
		elif step+1==self.steps(data):
			self.end_batch=length-1
		else:
			self.end_batch=self.end_batch+self.batch_size
		ii=self.end_batch-self.batch_size+1
		while(ii<=self.end_batch):
			data_batch.append(data[ii])
			labels_batch.append(labels[ii])
			ii=ii+1
		data=np.asarray(data_batch)
		labels=np.asarray(labels_batch)
		if(step==0):
			start_batch=0
		else:
			start_batch=self.end_batch-self.batch_size+1
		print"         Batch from "+str(start_batch)+" - "+str(self.end_batch)
		return data,labels
	
	def steps(self,data):
		batch_size=float(self.batch_size)
		length=len(data)
		steps=int(math.ceil(length/batch_size))
		return steps
	
	def layers(self,X,Y):	
		
		initializer = tf.contrib.layers.xavier_initializer()
		regularizer = tf.contrib.layers.l2_regularizer(0.001)
		
		input_layer           = tf.reshape(X,self.input,name="reshape")

		convolutional_layer11  = tf.layers.conv2d(inputs=input_layer,filters=self.filter_sizes[0],kernel_size=self.kernel_size,padding="same",activation=tf.nn.relu,name="conv11",kernel_initializer=tf.contrib.layers.xavier_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),use_bias=True,bias_initializer=initializer,bias_regularizer=regularizer)
		convolutional_layer12  = tf.layers.conv2d(inputs=convolutional_layer11,filters=self.filter_sizes[0],kernel_size=self.kernel_size,padding="same",activation=tf.nn.relu,name="conv12",kernel_initializer=tf.contrib.layers.xavier_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),use_bias=True,bias_initializer=initializer,bias_regularizer=regularizer)
		max_pooling_layer1    = tf.layers.max_pooling2d(inputs=convolutional_layer12,pool_size=self.pool_size,strides=self.pool_stride,name="pool1")
		
		print max_pooling_layer1.shape
		
		convolutional_layer21  = tf.layers.conv2d(inputs=max_pooling_layer1,filters=self.filter_sizes[1],kernel_size=self.kernel_size,padding="same",activation=tf.nn.relu,name="conv21",kernel_initializer=tf.contrib.layers.xavier_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),use_bias=True,bias_initializer=initializer,bias_regularizer=regularizer)
		convolutional_layer22  = tf.layers.conv2d(inputs=convolutional_layer21,filters=self.filter_sizes[1],kernel_size=self.kernel_size,padding="same",activation=tf.nn.relu,name="conv22",kernel_initializer=tf.contrib.layers.xavier_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),use_bias=True,bias_initializer=initializer,bias_regularizer=regularizer)
		max_pooling_layer2    = tf.layers.max_pooling2d(inputs=convolutional_layer22,pool_size=self.pool_size,strides=self.pool_stride,name="pool2")
		
		print max_pooling_layer2.shape
		
		convolutional_layer31  = tf.layers.conv2d(inputs=max_pooling_layer2,filters=self.filter_sizes[2],kernel_size=self.kernel_size,padding="same",activation=tf.nn.relu,name="conv31",kernel_initializer=tf.contrib.layers.xavier_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),use_bias=True,bias_initializer=initializer,bias_regularizer=regularizer)
		convolutional_layer32  = tf.layers.conv2d(inputs=convolutional_layer31,filters=self.filter_sizes[2],kernel_size=self.kernel_size,padding="same",activation=tf.nn.relu,name="conv32",kernel_initializer=tf.contrib.layers.xavier_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),use_bias=True,bias_initializer=initializer,bias_regularizer=regularizer)
		max_pooling_layer3    = tf.layers.max_pooling2d(inputs=convolutional_layer32,pool_size=self.pool_size,strides=self.pool_stride,name="pool3")
	
		print max_pooling_layer3.shape
	
		convolutional_layer41  = tf.layers.conv2d(inputs=max_pooling_layer3,filters=self.filter_sizes[3],kernel_size=self.kernel_size,padding="same",activation=tf.nn.relu,name="conv41",kernel_initializer=tf.contrib.layers.xavier_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),use_bias=True,bias_initializer=initializer,bias_regularizer=regularizer)
		convolutional_layer42  = tf.layers.conv2d(inputs=convolutional_layer41,filters=self.filter_sizes[3],kernel_size=self.kernel_size,padding="same",activation=tf.nn.relu,name="conv42",kernel_initializer=tf.contrib.layers.xavier_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),use_bias=True,bias_initializer=initializer,bias_regularizer=regularizer)
		max_pooling_layer4    = tf.layers.max_pooling2d(inputs=convolutional_layer42,pool_size=self.pool_size,strides=self.pool_stride,name="pool4")

		print max_pooling_layer4.shape
		
		flatten=tf.layers.Flatten()(max_pooling_layer4)
		#flatten = tf.reshape(max_pooling_layer4, [-1, 256])
		print "flatten values:"
		print flatten.shape[1]
		
		fully_connected_layer1 = tf.layers.dense(inputs=flatten,units=flatten.shape[1],activation=tf.nn.relu,name="fc1",kernel_initializer=tf.random_normal_initializer(stddev=0.01),kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
		dropout1               = tf.layers.dropout(inputs=fully_connected_layer1,rate=self.keep_prob,training=True,name="dropout1")
		fully_connected_layer2 = tf.layers.dense(inputs=dropout1,units=256,activation=tf.nn.relu,name="fc2",kernel_initializer=tf.random_normal_initializer(stddev=0.01),kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
		dropout2               = tf.layers.dropout(inputs=fully_connected_layer2,rate=self.keep_prob,training=True,name="dropout2")
		
		output_layer          = tf.layers.dense(inputs=dropout2,units=2,name="output_layer")
		
		tf.add_to_collection('vars',output_layer)
		return output_layer
		