import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv

def carga_csv(file_name):
    """carga el fichero csv especificado y lo
 devuelve en un array de numpy
    """
    valores = read_csv(file_name, header=None).values
    # suponemos que siempre trabajaremos con float
    return valores.astype(float)

datos = carga_csv('ex1data2.csv')
X = datos[:, :-1] # (47, 2)
Y = datos[:, -1] # (47,)

m = np.shape(X)[0]
n = np.shape(X)[1]

def normalizar(X):
    return ((X - np.mean(X, axis = 0)) / np.std(X, axis = 0), np.mean(X, axis = 0), np.std(X, axis = 0))
    
# añadimos una columna de 1's a la X_norm
X_norm, mean, std = normalizar(X)
X_norm = np.hstack([np.ones([m, 1]), X_norm])
X = np.hstack([np.ones([m, 1]), X])

def init_theta(X):
    return np.zeros((X.shape[1], 1))

def hypothesis(X, Theta):
    return np.dot(X, Theta)


def cost(X, Theta, Y):
    return np.dot((hypothesis(X, Theta).flatten() - Y).T, hypothesis(X, Theta).flatten() - Y)  / (2 * m)

def descenso_gradiente(X, Y, alpha):
    Theta = init_theta(X)
    costes = []
    for i in range(10000):
        for j in range(X.shape[1]):
            Theta[j] -= (alpha / m) * np.sum((hypothesis(X, Theta).flatten() - Y) * X[:, j])
        costes.append(cost(X, Theta, Y))
    return (Theta, costes)

def eq_normal(X, Y):
    return np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X), rcond=1e-15), X.T), Y.reshape((Y.shape[0], 1)))


alpha = 0.001
Thetas, costes = descenso_gradiente(X_norm, Y, alpha)
Thetas_qe_normal = eq_normal(X, Y)

#test

test = np.array([[1650, 3]])
test_norm = (test - mean) / std
test = np.hstack([np.ones([np.shape(test)[0], 1]), test])
test_norm = np.hstack([np.ones([np.shape(test)[0], 1]), test_norm])
print("gradient" + str(hypothesis(test_norm, Thetas)))
print("norm" + str(hypothesis(test, Thetas_qe_normal)))

#cost function graphic
#plt.plot(np.linspace(0, 9999, 10000), np.array(costes))

#graphic with data

plt.show()

