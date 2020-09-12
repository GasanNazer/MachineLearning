import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
from scipy.io import loadmat
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures

data = loadmat('ex5data1.mat')
#se pueden consultar las claves con data.keys( )
y = data ['y']
X = data ['X']
X_val = data ['Xval']
y_val = data ['yval']
X_test = data ['Xtest']
y_test = data ['ytest']

#almacena los datos leídos en X, y

m = np.shape(X)[0]
m_val = np.shape(X_val)[0]
n = np.shape(X)[1]
X = np.hstack([np.ones([len(X), 1]), X])
X_val = np.hstack([np.ones([len(X_val), 1]), X_val])
#lineal regression

def init_theta(X):
    return np.ones((X.shape[1], ))

def init_theta_zeros(X):
    return np.zeros((X.shape[1], ))

def hypothesis(X, Theta):
    return np.dot(X, Theta)

def cost(Theta,X, Y, reg):
    aux = (reg / (2 * m)) * np.sum(Theta[1:] ** 2)
    return 1/(2*len(X)) * (np.sum(np.square(hypothesis(X, Theta)-Y))) + aux

def gradient(Theta, X, Y, reg):
    Y = Y.ravel()
    coste = cost(Theta, X, Y, reg)
    grad = 1 / len(X) * np.dot((hypothesis(X, Theta) - Y), X)
    grad[1:] += (reg / len(X)) * Theta[1:]
    return (coste, grad.ravel())

#print(gradient(init_theta(X), X, y, 1))

def model_lineal_regression(X, y, reg):
    y = y.ravel()
    theta = init_theta(X)
    min = opt.minimize(gradient, theta, args=(X, y, reg), method='TNC', jac=True)
    theta = min.x
    #plt.scatter(X[:, 1:], y, c = 'red', marker = 'x')
    #plt.xlabel("Change in water level (x)")
    #plt.ylabel("Water flowing out of the dam (y)")
    #plt.plot(X[:, 1], hypothesis(X, theta))
    return theta

#model_lineal_regression(X, y, 0)

##################################

#curvas de aprendizaje

def curvas_aprendizaje(X, y, X_val, y_val, reg):
    dots = np.arange(m)
    error_dots = np.zeros(m)
    error_val = np.zeros(m)
    for i in range(1, m + 1):
        theta = model_lineal_regression(X[0 : i], y[0 : i], reg)
        error_dots[i - 1] = gradient(theta, X[0:i], y[0:i], reg)[0]
        error_val[i - 1] = gradient(theta, X_val, y_val, reg)[0]
    plt.plot(dots, error_dots, label='Train')
    plt.plot(dots, error_val,  label='Cross Validation')
    plt.xlabel("Number of training examples")
    plt.ylabel("Error")
    plt.legend(loc = 'upper right')

#curvas_aprendizaje(X, y, X_val, y_val, 2)

##########################################

#polinomial regression

def modify_x(X, p):
    return PolynomialFeatures(p, include_bias = False).fit_transform(X)

def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return ((X - mean) / std, mean, std)

def model_polinomial_regression(X, X_pol, norm, y, reg): #X[:, 1:] expected
    y = y.ravel()
    theta = init_theta_zeros(X_pol)
    min = opt.minimize(gradient, theta, args=(X_pol, y, reg), method='TNC', jac=True)
    theta = min.x
    '''
    dots  = np.arange(X.min() - 5, X.max() + 5, 0.05)
    dots_pol = modify_x(dots.reshape(-1,1), 8)
    dots_pol = (dots_pol-norm[1])/norm[2]
    dots_pol = np.hstack([np.ones([len(dots_pol), 1]), dots_pol])
    
    plt.scatter(X, y, c = 'red', marker = 'x')
    plt.plot(dots, hypothesis(dots_pol, theta))
    '''
    return theta

X_pol = modify_x(X[:, 1:], 8)
norm = normalize(X_pol)
X_pol = norm[0]
X_pol = np.hstack([np.ones([len(X), 1]), X_pol])

X_val_pol = modify_x(X_val[:, 1:], 8)
X_val_pol = (X_val_pol - norm[1]) / norm[2]
X_val_pol = np.hstack([np.ones([len(X_val), 1]), X_val_pol])

#model_polinomial_regression(X[:, 1:], X_pol, norm, y, 0)

def graphic_pol(X, y, X_val, y_val, reg):
    dots = np.arange(m)
    error_dots = np.zeros(m)
    error_val = np.zeros(m)
    for i in range(1, m + 1):
        theta = model_polinomial_regression(None, X[0 : i], None, y[0 : i], reg)
        error_dots[i - 1] = gradient(theta, X[0:i], y[0:i], reg)[0]
        error_val[i - 1] = gradient(theta, X_val, y_val, reg)[0]
    plt.plot(dots, error_dots)
    plt.plot(dots, error_val)


#graphic_pol(X_pol, y, X_val_pol, y_val, 1)


#########################

#modify lambda

def choose_lambda(X, y, X_val, y_val):
    lambd = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    error_dots = np.zeros(len(lambd))
    error_val = np.zeros(len(lambd))
    for i in range(len(lambd)):
        theta = model_polinomial_regression(None, X, None, y, lambd[i])
        error_dots[i] = gradient(theta, X, y, lambd[i])[0]
        error_val[i] = gradient(theta, X_val, y_val, lambd[i])[0]
    plt.plot(lambd, error_dots)
    plt.plot(lambd, error_val)

#choose_lambda(X_pol, y, X_val_pol, y_val)


X_test_pol = modify_x(X_test, 8)
X_test_pol = (X_test_pol - norm[1]) / norm[2]

X_test_pol = np.hstack([np.ones([len(X_test), 1]), X_test_pol])

theta = model_polinomial_regression(None, X_pol, None, y, 3)
#print(gradient(theta, X_test_pol, y_test, 0)[0])

plt.show()