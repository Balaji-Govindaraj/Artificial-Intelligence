import cv2
import sys
import time
import pandas as pd
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
test=pd.read_csv("training_data.csv",delimiter=',')
[X_test_left_path,X_test_center_path,X_test_right_path]=[test.left,test.center,test.right]
left_image=[]
center_image=[]
right_image=[]
total=len(X_test_left_path)
for i in range(total):
	li=cv2.imread(X_test_left_path[i])
	li_resize=cv2.resize(li,(64,64),interpolation=cv2.INTER_LINEAR)
	li_data=img_to_array(li_resize)
	left_image.append(li_data)
	ce=cv2.imread(X_test_center_path[i])
	ce_resize=cv2.resize(ce,(64,64),interpolation=cv2.INTER_LINEAR)
	ce_data=img_to_array(ce_resize)
	center_image.append(ce_data)
	ri=cv2.imread(X_test_right_path[i])
	ri_resize=cv2.resize(ri,(64,64),interpolation=cv2.INTER_LINEAR)
	ri_data=img_to_array(ri_resize)
	right_image.append(ri_data)	
	print "loading",(i*100)/total,"% testing data\r",
left_image=np.asarray(left_image)/255.0
center_image=np.asarray(center_image)/255.0
right_image=np.asarray(right_image)/255.0
model_name='model'
json_file = open(model_name+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(model_name+".h5")
loaded_model.compile(optimizer='adam',loss=['mse','mse','mse','mse','mse','mse'],metrics=['acc'])
output = loaded_model.predict([left_image,center_image,right_image])
for i in range(len(output[0])):
	print str(i)
	print "''"
	print "''"
	print "-------Linear X : "+str(output[0][i][0])+", ",
	print "Linear Y : "+str(output[1][i][0])+", ",
	print "Linear Z : "+str(output[2][i][0])
	print "-------Angular X : "+str(output[3][i][0])+", ",
	print "Angular Y : "+str(output[4][i][0])+", ",
	print "Angular Z : "+str(output[5][i][0])
	print "\n"
