import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import optimize as opt
from scipy import ndimage
from utils import load_images_from_folder
from utils import calculate_probability
from sklearn.preprocessing import PolynomialFeatures

Y_train = [] # labels are created during execution time
num_px = 64 # during execution images are resized to 64x64x3. This way we lose their quality but we save computational time. 
C = 3 # number of classes to detect

images_train = load_images_from_folder(Y_train, folder="images_dev")
X_train = images_train / 255 # normalize dataset
Y_train = np.array(Y_train) # convert the labels Y list into a numpy array

X_train = X_train.T
X_train = np.hstack([np.ones([len(X_train), 1]), X_train])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, X, Y, lambd):
    theta = theta.reshape((len(theta), 1))  
    A = sigmoid(np.matmul(X, theta))
    reg = (lambd / (2 * len(X))) * np.sum(theta ** 2)
    return (- 1 / (len(X))) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A  + 1e-6)) + reg

def gradient(theta, X, Y, lambd):
    theta = theta.reshape((len(theta), 1))
    A = sigmoid(np.matmul(X, theta))
    identity = np.identity(theta.shape[0])
    identity[0][0] = 0
    return (np.dot(X.T, (A - Y)) / len(Y) + (lambd / len(X)) * np.dot(identity, theta)).ravel()

def model(x, Y, lambd):
    theta = np.zeros(X.shape[1])
    result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args = (X, Y, lambd), maxfun=50)
    theta = result[0]
    return theta

def oneVsAll(X, y, n_labels, reg):
    """
    oneVsAll entrena varios clasificadores por regresión logística con término
    de regularización 'reg' y devuelve el resultado en una matriz, donde
    la fila i-ésima corresponde al clasificador de la etiqueta i-ésima
    """
    theta = []
    for i in range(n_labels):
        theta.append(model(X, Y[:, i].reshape((len(Y), 1)), reg))
    theta = np.array(theta)
    return theta

    


thetas = oneVsAll(X_train, Y_train, C, 100)
calculate_probability(X_train, Y_train, thetas) 

#theta = np.zeros(X.shape[1])
#print(theta.shape)
#print(gradient(theta, X, Y[:, 0].reshape((len(Y), 1)), 0.1).shape)
#print(cost(theta, X, Y, 0.1))
