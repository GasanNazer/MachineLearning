import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import optimize as opt
from scipy import ndimage
from utils import load_images_from_folder
from sklearn.preprocessing import PolynomialFeatures

Y_train = [] # labels are created during execution time
Y_dev = []
Y_test = []
num_px = 64 # during execution images are resized to 64x64x3. This way we lose their quality but we save computational time. 
C = 3 # number of classes to detect

images_train = load_images_from_folder(Y_train, folder="images_train")
X_train = images_train / 255 # normalize dataset
Y_train = np.array(Y_train) # convert the labels Y list into a numpy array

images_dev = load_images_from_folder(Y_dev, folder="images_dev")
X_dev = images_dev / 255 # normalize dataset
Y_dev = np.array(Y_dev) # convert the labels Y list into a numpy array

images_test = load_images_from_folder(Y_test, folder="images_test")
X_test = images_test / 255 # normalize dataset
Y_test = np.array(Y_test) # convert the labels Y list into a numpy array

X_train = X_train.T
X_train = np.hstack([np.ones([len(X_train), 1]), X_train])
X_dev = X_dev.T
X_dev = np.hstack([np.ones([len(X_dev), 1]), X_dev])
X_test = X_test.T
X_test = np.hstack([np.ones([len(X_test), 1]), X_test])

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

def model(X, Y, lambd):
    theta = np.zeros(X.shape[1])
    result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args = (X, Y, lambd), maxfun=50)
    theta = result[0]
    return theta

def oneVsAll(X, Y, n_labels, reg):
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

def choose_lambda(X, y, X_val, y_val, C):
    lambd = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100])
    error_dots = np.zeros(len(lambd))
    error_val = np.zeros(len(lambd))
    for i in range(len(lambd)):
        theta = oneVsAll(X, y, C, lambd[i])
        for c in range(C):
            error_dots[i] += gradient(theta[c, :], X, y, lambd[i])[0]
            error_val[i] += gradient(theta[c, :], X_val, y_val, lambd[i])[0]
        error_dots[i] /= C
        error_val[i] /= C
    plt.plot(lambd, error_dots)
    plt.plot(lambd, error_val)
    print(error_dots)
    print(error_val)
    plt.show()

def choose_lambda_acc(X, Y, X_val, Y_val, C=3):
    lambd = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000])
    accuracy_dots = np.zeros(len(lambd))
    accuracy_val = np.zeros(len(lambd))

    for i in range(len(lambd)):
        theta = oneVsAll(X, Y, C, lambd[i])
        prediction = np.dot(X, theta.T)
        index_max = np.argmax(prediction, axis = 1)
        index_max = index_max.reshape((len(index_max), 1))
        accuracy_dots[i] = np.sum(np.argmax(Y, axis = 1).reshape((len(Y), 1)) == index_max) / Y.shape[0]

        predict_val = np.dot(X_val, theta.T)
        index_max = np.argmax(predict_val, axis = 1)
        index_max = index_max.reshape((len(index_max), 1))
        accuracy_val[i] = np.sum(np.argmax(Y_val, axis = 1).reshape((len(Y_val), 1)) == index_max) / Y_val.shape[0]
    plt.plot(lambd, accuracy_dots)
    plt.plot(lambd, accuracy_val)
    print(accuracy_dots)
    print(accuracy_val)
    plt.show()

def calculate_probability(X, Y, theta, C=3):
    prediction = np.dot(X, theta.T)
    index_max = np.argmax(prediction, axis = 1)
    index_max = index_max.reshape((len(index_max), 1))
    accuracy = np.sum(np.argmax(Y, axis = 1).reshape((len(Y), 1)) == index_max) / Y.shape[0]
    print("Accuracy " + str(accuracy * 100) + '%')
    
    precision = np.zeros(C)
    recall = np.zeros(C)
    for c in range(C):
        precision[c] = np.sum((Y[:, c] == 1) * (index_max == c).ravel()) / np.sum(index_max == c)
        recall[c] = np.sum((Y[:, c] == 1) * (index_max == c).ravel()) / np.sum(Y[:, c] == 1)
    print("precision dogs: " + str(precision[0]))
    print("precision cats: " + str(precision[1]))
    print("precision elephants: " + str(precision[2]))

    print("recall dogs: " + str(recall[0]))
    print("recall cats: " + str(recall[1]))
    print("recall elephants: " + str(recall[2]))

    print("f1 score dogs: " + str(2 * precision[0] * recall[0] / (precision[0] + recall[0])))
    print("f1 score cats: " + str(2 * precision[1] * recall[1] / (precision[1] + recall[1])))
    print("f1 score elephants: " + str(2 * precision[2] * recall[2] / (precision[2] + recall[2])))

    print("average dogs: " + str((precision[0] + recall[0]) / 2))
    print("average cats: " + str((precision[1] + recall[1]) / 2))
    print("average elephants: " + str((precision[1] + recall[1]) / 2))


    

#choose_lambda(X_train, Y_train, X_dev, Y_dev, 3)
#choose_lambda_acc(X_train, Y_train, X_dev, Y_dev, 3)
thetas = oneVsAll(X_train, Y_train, C, 300)
print("Accuracy training set:")
calculate_probability(X_train, Y_train, thetas) 
print("Accuracy validation set:")
calculate_probability(X_dev, Y_dev, thetas) 
print("Accuracy test set:")
calculate_probability(X_test, Y_test, thetas)

#theta = np.zeros(X.shape[1])
#print(theta.shape)
#print(gradient(theta, X, Y[:, 0].reshape((len(Y), 1)), 0.1).shape)
#print(cost(theta, X, Y, 0.1))
