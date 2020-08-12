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

datos = carga_csv('ex1data1.csv')
X = datos[:, :-1] # (97, 1)
Y = datos[:, -1] # (97,)

m = np.shape(X)[0]
n = np.shape(X)[1]
    
# añadimos una columna de 1's a la X
X = np.hstack([np.ones([m, 1]), X])

def init_theta(X):
    return np.zeros((X.shape[1], 1))

def hypothesis(X, Theta):
    return np.dot(X, Theta)


def cost(X, Theta, Y):
    return np.sum((hypothesis(X, Theta).flatten() - Y) ** 2) / (2 * m)

def descenso_gradiente(X, Y, alpha):
    Theta = init_theta(X)
    costes = []
    for i in range(1500):
        for j in range(X.shape[1]):
            Theta[j] -= (alpha / m) * np.sum((hypothesis(X, Theta).flatten() - Y) * X[:, j])
        costes.append(cost(X, Theta, Y))
    return (Theta, costes)



alpha = 0.01
Thetas, costes = descenso_gradiente(X, Y, alpha)

#cost function graphic
plt.plot(np.linspace(0, 1499, 1500), np.array(costes))

#graphic with data
#plt.scatter(X[:, 1:], Y, c = 'red', marker = 'x')
#plt.plot(X[:, 1], hypothesis(X, Thetas))
plt.show()

