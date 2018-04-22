import numpy as np
import matplotlib.pyplot as plt
x=np.matrix('1 1;1 2;1 9;1 4;1 7;1 6;1 11;1 8;1 19;1 10;1 21;1 12;1 13;1 14;1 5;1 16')
y=np.matrix('0;1;0;1;0;1;0;1;0;1;0;1;0;1;0;1')
test_x=np.matrix('12;3')
theta=np.matrix('0.01;0.01')
learning_rate=0.001
iterations=100
m=len(x)
for i in range(iterations):
	t=theta[0,0]+theta[1,0]*x[:,1]
	p=1/(1+np.exp(-t))
	print(p)
	cost=(-1/m)*(y.T.dot(np.log(p))+(1-y).T.dot(np.log(1-p)))
	#if(i%10000==0):
	
	print(cost[0,0])
	loss=p-y
	for j in range(len(theta)):
		theta_derivative=learning_rate*loss.T.dot(x[:,j])/m
		theta[j,0]=theta[j,0]-theta_derivative
        #print theta[0]

test_t=theta[0,0]+theta[1,0]*test_x[:,0]
test_p=1/(1+np.exp(-test_t))
print(test_p)
#print(metrics.classification_report(yy,y_proba))
plt.plot(x[:,1],y,'ro',x[:,1],p,'bo')
plt.axis([0, 30, -0.2, 1.2])
plt.xlabel("x feature")
plt.ylabel("Predicted output")
plt.title("Training")
plt.show()