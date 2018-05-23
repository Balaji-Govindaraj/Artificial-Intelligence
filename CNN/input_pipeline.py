import tensorflow as tf
import os
import glob
import time
from BGNet import BGNet

def load_train():
	X=[]
	Y=[]
	folders=['dogs','cats']
	for folder in folders:
		index=folders.index(folder)
		path=os.path.join('train_batch/',folder,'*g')
		for image_path in glob.glob(path):
			X.append(image_path)
			Y.append(index)			
	return X,Y

def parse_function(filenames,label):
	one_hot = tf.one_hot(label,2)
	image_string = tf.read_file(filenames)
	image_decoded = tf.image.decode_jpeg(image_string, channels=3)
	img_resized = tf.image.resize_images(image_decoded, [64, 64])
	image = tf.cast(img_resized, tf.float32)
	return image, one_hot


batch_size=32
f,l=load_train()
filenames=tf.constant(f)

labels=tf.constant(l)
train_dataset = tf.data.Dataset.from_tensor_slices((filenames,labels))
train_dataset = train_dataset.map(parse_function)
train_dataset = train_dataset.shuffle(buffer_size=1000)
#train_dataset = train_dataset.repeat(2)
train_dataset = train_dataset.batch(batch_size)	
iterator = tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
next_element= iterator.get_next()
training_init_op = iterator.make_initializer(train_dataset)
	
bgnet=BGNet()
X=tf.placeholder(tf.float32,[None,bgnet.img_height,bgnet.img_width,3],name="input")
Y=tf.placeholder(tf.float32,[None,2],name="target")
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,logits=bgnet.layers(X,Y),name="softmax"))
tf.summary.scalar('loss', loss)
optimizer=tf.train.AdamOptimizer(bgnet.learning_rate).minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())	
	for epoch in range(2):
		print "\nEpoch "+str(epoch+1)+":"
		sess.run(training_init_op)
		for step in range((len(f)/batch_size)+1):
			print "\n     Step "+str(step+1)+":"
			start_time=time.time()
			element=sess.run(next_element)
			print '     Testing data calculation time: {} seconds\n'.format(round(time.time()-start_time, 2))
			optimizer1,loss1=sess.run([optimizer,loss],feed_dict={X:element[0],Y:element[1]})	
			print "         loss is "+str(round(loss1,3))
					
				
		

