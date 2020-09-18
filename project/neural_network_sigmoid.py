import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
from scipy.io import loadmat
from sklearn.preprocessing import PolynomialFeatures
import scipy.optimize as opt
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

def derivade_sigmoid(dA):
    return dA * (1 - dA)

def pesosAleatorios(L_in, L_out, epsilon = 0.12):
    #devolverá una matriz de dimensión (L_out, 1 + L_in)
    return np.random.rand(L_out, 1 + L_in) * (epsilon + epsilon) - epsilon

def linear_activation_forward(A_prev, theta):
    Z = np.dot(A_prev, theta.T)
    A = sigmoid(Z)
    return A

def L_model_forward(X, parameters):
    A = X
    cache = {}
    L = len(parameters)
    for l in range(L):
        A_prev = A
        A_prev = np.hstack([np.ones([A_prev.shape[0], 1]), A_prev]) 
        cache["A" + str(l + 1)] = A_prev
        A = linear_activation_forward(A_prev, parameters['theta' + str(l + 1)])
    cache["A" + str(L + 1)] = A
    return (A, cache)

def cost(parameters, A, Y, lambd, m):
    reg = (lambd / (2 * m)) * (np.sum(parameters["theta1"][:, 1:] ** 2) + np.sum(parameters["theta2"][:, 1:] ** 2))
    coste = (Y * np.log(A)) + ((1 - Y) * np.log( 1 - A) )
    return (- 1 / m) * coste.sum() + reg

def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    parameters = {}
    grads = {}
    grads["dT1"] = 0
    grads["dT2"] = 0
    params_rn = params_rn.reshape(len(params_rn), 1)
    theta1 = np.reshape(params_rn[: num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
    theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1) :], (num_etiquetas, (num_ocultas + 1)))
    
    parameters['theta1'] = theta1
    parameters['theta2'] = theta2
    AL, cache = L_model_forward(X, parameters)
    coste = cost(parameters, AL, y, reg, len(X))
    grads["dA3"] = AL - y
    grads["dA2"] = np.dot(grads["dA3"], parameters['theta2']) * derivade_sigmoid(cache["A2"])

    grads["dT1"] += (np.dot(grads["dA2"][:, 1:].T, cache["A1"]) / len(X)) 
    grads["dT1"][:, 1:] += theta1[:, 1:] * reg / len(X)

    grads["dT2"] += (np.dot(grads["dA3"].T, cache["A2"]) / len(X)) 
    grads["dT2"][:, 1:] += theta2[:, 1:] * reg / len(X)

    theta_grads = np.concatenate((grads["dT1"].ravel(), grads["dT2"].ravel()))

    return (coste, theta_grads)


def modelo(input_size, num_labels, X, Y, reg, iterations):
    parameters = {}
    inner_layer = 40
    theta = []
    params_rn = np.concatenate((pesosAleatorios(input_size, inner_layer).ravel(),pesosAleatorios(inner_layer, num_labels).ravel()))
    
    min = opt.minimize(backprop, params_rn, args=(input_size, inner_layer, num_labels, X, Y, reg), method='TNC', options={'maxiter': iterations}, jac=True)
    params_rn = min.x
    params_rn = params_rn.reshape(len(params_rn), 1)
    theta1 = np.reshape(params_rn[: inner_layer * (input_size + 1)], (inner_layer, (input_size + 1)))
    theta2 = np.reshape(params_rn[inner_layer * (input_size + 1) :], (num_labels, (inner_layer + 1)))
    parameters['theta1'] = theta1
    parameters['theta2'] = theta2
    return parameters


def choose_lambda_acc(X, Y, X_val, Y_val, C=3):
    lambd = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 300])
    accuracy_dots = np.zeros(len(lambd))
    accuracy_val = np.zeros(len(lambd))

    for i in range(len(lambd)):
        theta = modelo(X.shape[1], 3, X, Y, lambd[i], 100) #300
        
        prediction, _ = L_model_forward(X, theta)
        index_max = np.argmax(prediction, axis = 1)
        index_max = index_max.reshape((len(index_max), 1))
        accuracy_dots[i] = np.sum(np.argmax(Y, axis = 1).reshape((len(Y), 1)) == index_max) / Y.shape[0]

        predict_val, _ = L_model_forward(X_val, theta)
        index_max = np.argmax(predict_val, axis = 1)
        index_max = index_max.reshape((len(index_max), 1))
        accuracy_val[i] = np.sum(np.argmax(Y_val, axis = 1).reshape((len(Y_val), 1)) == index_max) / Y_val.shape[0]
    plt.plot(lambd, accuracy_dots)
    plt.plot(lambd, accuracy_val)
    plt.xlabel("lambda")
    plt.ylabel("accuracy")
    plt.title("Searching optimal lambda")
    print(accuracy_dots)
    print(accuracy_val)
    plt.show()


def calculate_probability(X, Y, theta, C=3):
    AL, _ = L_model_forward(X, theta)
    indexes = np.argmax(AL, axis=1)
    index_max = indexes.reshape((len(indexes), 1))
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

#choose_lambda_acc(X_train, Y_train, X_dev, Y_dev, C=3)

params = modelo(X_train.shape[1], 3, X_train, Y_train, 30, 300)
print("Accuracy training set:")
calculate_probability(X_train, Y_train, params, C=3) 
print("Accuracy validation set:")
calculate_probability(X_dev, Y_dev, params, C=3) 
print("Accuracy test set:")
calculate_probability(X_test, Y_test, params, C=3)
