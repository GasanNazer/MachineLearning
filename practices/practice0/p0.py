import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import time

def fun(x):
    return -1 * x ** 2 + 2 * x + 3 

def integra_mc(fun, a, b, num_puntos=10000):
    X = np.random.rand(num_puntos) * (b - a) + a
    M = max(fun(X))
    Y = np.random.rand(num_puntos) * M
    

    line = np.linspace(a, b, num_puntos)
    plt.scatter(X[:200], Y[:200], c = 'red', marker = 'x')
    plt.plot(line, fun(line))
    sum_nonVect(X, Y)
    N_debajo = sum_Vectorized(X, Y)

    plt.show()
    return (N_debajo / num_puntos) * (b - a) * M

def sum_nonVect(X, Y):
    tic = time.process_time()
    N_debajo = 0
    points = fun(X)
    for i in range(len(Y)):
        if(Y[i] < points[i]):
            N_debajo += 1
    toc = time.process_time()
    print("Non-vectorized sum: " + str(N_debajo) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

#difference with 1000000, with less it's 0ms
def sum_Vectorized(X, Y):
    tic = time.process_time()
    N_debajo = np.sum(Y < fun(X))
    toc = time.process_time()
    print("Vectorized sum: " + str(N_debajo) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
    return N_debajo



our_res = integra_mc(fun, 0, 3)
true_res = scipy.integrate.quad(fun, 0, 3)

print("Result obtained by us: " + str(our_res))
print("Result obtained by the scipy.integrate.quad: " + str(true_res[0]))

