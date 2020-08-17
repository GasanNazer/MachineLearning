import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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

def make_data(X, Y):
    step = 0.1
    T0 = np.arange(-10, 10, step)
    T1 = np.arange(-1, 4, step)
    T0, T1 = np.meshgrid(T0, T1)

    Coste = np.empty_like(T0)
    for ix, iy in np.ndindex(T0.shape):
        Coste[ix, iy] = cost(X, [T0[ix, iy], T1[ix, iy]], Y)

    return (T0, T1, Coste)

def graphic_3D(X, Y):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    T0, T1, Z = make_data(X, Y)
    surf = ax.plot_surface(T0, T1, Z, cmap = cm.coolwarm, linewidth = 1, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)


def contour(X, Y):
    T0, T1, Z = make_data(X, Y)
    plt.contour(T0, T1, Z, np.logspace(-2, 3, 20), colors='blue')

alpha = 0.01
Thetas, costes = descenso_gradiente(X, Y, alpha)

#cost function graphics:

#3D graphic
#graphic_3D(X, Y)

#contour
contour(X, Y)
plt.scatter(Thetas[0], Thetas[1], c = 'red', marker = 'x')

#graphic with data
#plt.scatter(X[:, 1:], Y, c = 'red', marker = 'x')
#plt.plot(X[:, 1], hypothesis(X, Thetas))
plt.show()

