import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
from scipy.io import loadmat
from sklearn.preprocessing import PolynomialFeatures

data = loadmat('ex3data1.mat')
#se pueden consultar las claves con data.keys( )
y = data ['y']
X = data ['X']
#almacena los datos leídos en X, y

m = np.shape(X)[0]
n = np.shape(X)[1]

X = np.hstack([np.ones([m, 1]), X])

weights = loadmat('ex3weights.mat')
theta1, theta2 = weights['Theta1'], weights['Theta2']
#Theta1 es de dimensión 25x401
#Theta2 es de dimensión 10x26

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def calculate_probability(H, Y):
    index_max = np.argmax(H, axis = 1)
    index_max = index_max.reshape((len(index_max), 1))
    error = np.sum(Y - 1 == index_max) / Y.shape[0]
    print(str(error * 100) + '%')

def neuro_network(X, Y, theta1, theta2):
    Z1 = np.dot(X, theta1.T)
    A1 = sigmoid(Z1)
    A1 = np.hstack([np.ones([A1.shape[0], 1]), A1])
    Z2 = np.dot(A1, theta2.T)
    A2 = sigmoid(Z2)
    calculate_probability(A2, Y)

neuro_network(X, y, theta1, theta2)