import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def cost(predicted_y,correct_y):
	m=len(predicted_y)
	loss=predicted_y-correct_y
	MSE=np.sum(np.power(loss,2))/2*m
	return MSE
	
def closedForm(theta,train_data,y,iterations):
	predicted_train_y=0
	m=len(train_data.iloc[:,0])
	final_MSE=np.inf
	x=train_data.iloc[:,:3].values
	predicted_train_y=theta[0]*train_data.iloc[:,0]+theta[1]*train_data.iloc[:,1].values+theta[2]*train_data.iloc[:,2].values
	theta=np.linalg.inv(x.T.dot(x)).dot(x.T).dot(predicted_train_y)
	return theta,predicted_train_y
	
def batchGradientDescent(learning_rate,theta,train_data,y,iterations):
	predicted_train_y=0
	m=len(train_data.iloc[:,0])
	for i in range(iterations):
		predicted_train_y=theta[0]*train_data.iloc[:,0]+theta[1]*train_data.iloc[:,1].values+theta[2]*train_data.iloc[:,2].values
		loss=np.subtract(predicted_train_y,y)
		print "Cost: "+str(cost(predicted_train_y,y))
		for j in range(len(theta)):
			theta_derivative=(learning_rate/m)*loss.T.dot(train_data.iloc[:,j].values)
			theta[j]=theta[j]-theta_derivative
	return theta,predicted_train_y

def plotTrainingSet(x,correct_y,predicted_train_y):
	plt.plot(x, correct_y, 'ro',x,predicted_train_y)
	plt.axis([-15, 15, -15, 15])
	plt.xlabel("x feature")
	plt.ylabel("Predicted output")
	plt.title("Training")
	plt.show()

def predictTestData(theta,test_data):
	y=theta[0]+theta[1]*test_data.iloc[:,0].values+theta[2]*np.power(test_data.iloc[:,0].values,2)
	print "Result: "+str(y)
	return y

def trainTestSplit():
	train_data=pd.read_csv("training_pr.csv")
	test_data=pd.read_csv("testing_pr.csv")
	train_data_new=pd.DataFrame()
	train_data_new[1]=train_data.iloc[:,0]
	x=train_data_new.iloc[:,0].values
	x2=np.power(x,2)
	train_data_new[2]=x2
	train_data_new.insert(0,0,1)
	train_data_new[3]=train_data.iloc[:,1]
	'''x3=np.power(x,3)
	train_data_new[4]=x3
	x4=np.power(x,4)
	train_data_new[5]=x4
	x5=np.power(x,5)
	train_data_new[6]=x5'''
	correct_y=train_data_new[3].values
	return train_data_new,test_data

theta=[0.1,0.2,0.2]
learning_rate=0.01
iterations=1000
train_data,test_data=trainTestSplit()	
#theta,predicted_y=closedForm(theta,train_data,train_data.iloc[:,3],iterations) 							#theta=[0.1,0.2,0.2]
theta,predicted_y=batchGradientDescent(learning_rate,theta,train_data,train_data.iloc[:,3],iterations)
predictTestData(theta,test_data)
plotTrainingSet(train_data.iloc[:,1],train_data.iloc[:,3],predicted_y)
