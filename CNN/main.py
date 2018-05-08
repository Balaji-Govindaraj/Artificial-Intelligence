import tensorflow as tf
import sys
import numpy as np
import time
from BGNet import BGNet
from sklearn.model_selection import train_test_split

class Main:
	
	def train(self,bgnet,train_data,train_labels):
		X=tf.placeholder(tf.float32,[None,bgnet.img_height,bgnet.img_width,3],name="input")
		Y=tf.placeholder(tf.float32,[None,2],name="target")
		loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,logits=bgnet.layers(X,Y)))
		tf.summary.scalar('loss', loss)
		optimizer=tf.train.AdamOptimizer(bgnet.learning_rate).minimize(loss)
		init=tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init)
			train_writer = tf.summary.FileWriter('./logs', sess.graph)
			saver=tf.train.Saver()
			counter = 0
			for epoch in range(bgnet.epoch):
				print "\nEpoch "+str(epoch+1)+":"
				for step in range(bgnet.steps(train_data)):
					counter=counter+1
					print "\n     Step "+str(step+1)+":"
					merged = tf.summary.merge_all()
					train_data_batch,train_labels_batch=bgnet.batch_data(step,train_data,train_labels)
					summary,optimizer1,loss1=sess.run(fetches=[merged,optimizer,loss],feed_dict={X:train_data_batch,Y:train_labels_batch})
					
					train_writer.add_summary(summary, counter)
					print "         loss is "+str(round(loss1,3))
					
				if((epoch+1)%1==0):
					saver.save(sess,"./model/model/mymodel")
					print "Saved at Iteration "+str(epoch+1)
				#bgnet.learning_rate=bgnet.learning_rate*5
					
	def test(self,bgnet,test_data,test_labels):
		meta_file='./model/mymodel.meta'
		saver=tf.train.import_meta_graph(meta_file)
		with tf.Session() as sess:
			saver.restore(sess,tf.train.latest_checkpoint('./model/'))
			X = tf.get_default_graph().get_tensor_by_name("input:0")
			output=tf.get_collection('vars')
			probabilities=tf.nn.softmax(output,name="softmax")
			#For two class only i have implemented the evaluation category
			accuracy=0
			start_time=time.time()
			for id,test in enumerate(test_data):
				image=np.array(test).reshape(bgnet.input)
				p=sess.run(fetches=[probabilities],feed_dict={X:image})	
				classes=[0,1]
				if p[0][0][0][0]>p[0][0][0][1]:
					classes=[1,0]
				if classes[0]==test_labels[id][0]:
					accuracy=accuracy+1
				print "Image "+str(id+1)+" - Dog:"+str(round(p[0][0][0][0],3))+" Cat:"+str(round(p[0][0][0][1],3))+" Predicted class:"+str(classes)+" Correct classs:"+str(test_labels[id])
			print '     Testing data calculation time: {} seconds\n'.format(round(time.time()-start_time, 2))
			acc=float(accuracy)/len(test_labels)
			acc=acc*100
			print "\nACCURACY IS "+str(acc)+"\n"
			
			
			
if __name__=='__main__':
	mode=(sys.argv[1]).upper()
	if mode=='TRAIN':
		bgnet=BGNet()
		main=Main()
		train_data_without_shuffle,train_labels_without_shuffle=bgnet.load_train()
		train_data,test_data,train_labels,test_labels=train_test_split(train_data_without_shuffle,train_labels_without_shuffle,test_size=0.3,random_state=42)
		main.train(bgnet,train_data,train_labels)
	elif mode=='TEST':
		bgnet=BGNet()
		main=Main()
		train_data_without_shuffle,train_labels_without_shuffle=bgnet.load_train()
		train_data,test_data,train_labels,test_labels=train_test_split(train_data_without_shuffle,train_labels_without_shuffle,test_size=0.3,random_state=42)
		main.test(bgnet,test_data,test_labels)
	else:
		print "Only TRAIN and TEST options available"