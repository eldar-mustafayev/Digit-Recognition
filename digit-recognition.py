import gzip
import numpy as np

from math import sqrt
from numpy import ones,zeros,concatenate
from numpy.random import rand,RandomState

def g(z):
    return 1 / (1 + np.exp(-z))


def cost(prediction, Y):
    m, J = Y.shape[0], 0
    for i in range(m):
        J += Y[ : ,i].T.dot(np.log(prediction[ : ,i])) + (1 - Y[ : ,i]).T.dot(np.log(1 - prediction[ : ,i]))
    return -J / m


def read_data(filename):
    with gzip.open(filename) as input:
        s = [1] + [float(i) for i in input.readline().split()]
        X, Y = np.array([s[:-1]],dtype=np.float64), np.array([s[-1]],dtype=np.int64)
        Y = Y % 10
        
        for line in input:
            s = [1] + [float(i) for i in line.split()]
            X, Y = np.append(X, [s[ :-1]], axis=0), np.append(Y,[s[-1]], axis=0)
            Y = Y % 10
    return (X, Y)


def initialTheta(unit_numbers):
    theta = []
    n = len(unit_numbers)
    for i in range(n-1):
        epsilon = sqrt(6) / sqrt(unit_numbers[i] + unit_numbers[i+1])
        theta.append(rand(unit_numbers[i+1], unit_numbers[i] + 1) * 2 * epsilon - epsilon)
    return theta


def gradient_descent(X, Y, alpha, lmbd, max_iter, *unit_numbers):
    theta = initialTheta(unit_numbers)
    m = X.shape[0]
    E = []
    for i in theta:
        E.append(ones(i.shape) * lmbd)
        E[-1][ : ,0] = 0

    L = len(unit_numbers)
    a = [0] * L; delta = [0] * L
    a[0] = concatenate((ones((m, 1)), X), axis=1).T
    while max_iter>0:
        for i in range(1,L-1):
            a[i] = concatenate((ones((1, m)),g(theta[i-1].dot(a[i-1]))))
        a[-1] = g(theta[-1].dot(a[-2]))
        
        delta[-1] = Y - a[-1]
        for i in range(L - 2, 0, -1):
            delta[i] = theta[i].T.dot(delta[i+1]) * a[i] * (1 - a[i])
            delta[i] = delta[i][1: ,:]
        
        for i in range(L - 1):
            theta[i] += alpha * (delta[i+1].dot(a[i].T) + E[i] * theta[i]) / m

        print('Cost:', cost(a[-1], Y))
        max_iter -= 1
    return theta


RandomState(12)

X, Y = read_data('data.gz')
m, n = X.shape
Y = concatenate((Y.reshape(m, 1), zeros((m, 9))), axis=1)
for i in range(m):
    t = int(Y[i,0])
    Y[i,0] = 0
    Y[i,t]  =1
Y = Y.T

alpha=1; lmbd=1; max_iter=1000
theta=gradient_descent(X, Y, alpha, lmbd, max_iter, 400,25,10)
