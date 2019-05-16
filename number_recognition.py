import numpy as np
from sklearn.model_selection import train_test_split

def read_data(filename):
    with open(filename) as input:
        s=[1]+[float(i) for i in input.readline().split()]
        X,Y=np.array([s[:-1]],dtype=np.float64),np.array([s[-1]],dtype=np.int64)
        Y=Y%10
        
        for line in input:
            s=[1]+[float(i) for i in line.split()]
            X,Y=np.append(X,[s[:-1]],axis=0),np.append(Y,[s[-1]],axis=0)
            Y=Y%10
    return (X,Y)

def hypothesis(X,theta):
    return 1/(1+np.exp(-X.dot(theta)))

def gradient_descent(X,Y,alpha,iter,function):
    O,m=np.ones(X[0].size),Y.size
    
    while iter>0:
        O-=X.T.dot(function(X,O)-Y)*alpha/m
        iter-=1
        
    return O

X,Y=read_data('data.txt')
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=12)
n=X_train[0].size; theta=np.zeros((n,10))


for i in range(10):
    theta[:,i]=gradient_descent(X_train,Y_train==i,0.8,3200,hypothesis)

prediction=np.argmax(X_test.dot(theta).T,axis=0)

print('\nAccuracy:\n',np.mean(prediction == Y_test) * 100)