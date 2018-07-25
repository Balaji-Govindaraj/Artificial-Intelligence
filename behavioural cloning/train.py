import cv2
import sys
import time
import numpy as np
import pandas as pd
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten, Bidirectional, TimeDistributed, Conv2D, merge
from keras.models import Model, Sequential
from keras.preprocessing.image import img_to_array
train=pd.read_csv("training_data.csv",delimiter=',')
[X_train_left_path,X_train_center_path,X_train_right_path]=[train.left,train.center,train.right]
[Y_linear_x,Y_linear_y,Y_linear_z]=[train.linear_x,train.linear_y,train.linear_z]
[Y_angular_x,Y_angular_y,Y_angular_z]=[train.angular_x,train.angular_y,train.angular_z]
left_image=[]
center_image=[]
right_image=[]
total=len(X_train_left_path)
for i in range(total):
	li=cv2.imread(X_train_left_path[i])
	li_resize=cv2.resize(li,(64,64),interpolation=cv2.INTER_LINEAR)
	li_data=img_to_array(li_resize)
	left_image.append(li_data)
	ce=cv2.imread(X_train_center_path[i])
	ce_resize=cv2.resize(ce,(64,64),interpolation=cv2.INTER_LINEAR)
	ce_data=img_to_array(ce_resize)
	center_image.append(ce_data)
	ri=cv2.imread(X_train_right_path[i])
	ri_resize=cv2.resize(ri,(64,64),interpolation=cv2.INTER_LINEAR)
	ri_data=img_to_array(ri_resize)
	right_image.append(ri_data)	
	print "loading",(i*100)/total,"% training data\r",
left_image=np.asarray(left_image)/255.0
center_image=np.asarray(center_image)/255.0
right_image=np.asarray(right_image)/255.0
label_linear_x=np.asarray(Y_linear_x)
label_linear_y=np.asarray(Y_linear_y)
label_linear_z=np.asarray(Y_linear_z)
label_angular_x=np.asarray(Y_angular_x)
label_angular_y=np.asarray(Y_angular_y)
label_angular_z=np.asarray(Y_angular_z)
input_left_image=Input(shape=(64, 64, 3))
input_center_image=Input(shape=(64, 64, 3))
input_right_image=Input(shape=(64, 64, 3))
conv_left_image=Conv2D(64, (1, 1), padding='same', activation='relu')(input_left_image)
conv_left_image=Conv2D(64, (3, 3), padding='same', activation='relu')(conv_left_image)
conv_center_image=Conv2D(64, (1, 1), padding='same', activation='relu')(input_center_image)
conv_center_image=Conv2D(64, (3, 3), padding='same', activation='relu')(conv_center_image)
conv_right_image=Conv2D(64, (1, 1), padding='same', activation='relu')(input_right_image)
conv_right_image=Conv2D(64, (3, 3), padding='same', activation='relu')(conv_right_image)
merged=merge([conv_left_image,conv_center_image,conv_right_image],mode='sum',concat_axis=1)
merged=Flatten()(merged)
out=Dense(100, activation='relu')(merged)
linear_x=Dense(1)(out)
linear_y=Dense(1)(out)
linear_z=Dense(1)(out)
angular_x=Dense(1)(out)
angular_y=Dense(1)(out)
angular_z=Dense(1)(out)
model = Model([input_left_image,input_center_image,input_right_image],[linear_x,linear_y,linear_z,angular_x,angular_y,angular_z])
model.summary()
model.compile(optimizer='adam',loss=['mse','mse','mse','mse','mse','mse'],metrics=['acc'])
model.fit([left_image,center_image,right_image],[label_linear_x,label_linear_y,label_linear_z,label_angular_x,label_angular_y,label_angular_z], epochs=10,batch_size=32, validation_split=0.2,shuffle=True)
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
