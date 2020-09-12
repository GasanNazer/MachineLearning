from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm

data1 = loadmat("ex6data1.mat")
data2 = loadmat("ex6data2.mat")
data3 = loadmat("ex6data3.mat")

y1 = data1['y']
X1 = data1['X']

y2 = data2['y']
X2 = data2['X']

y3 = data3['y']
X3 = data3['X']

X_val = data3['Xval']
y_val = data3['yval']


def draw_graphic(X, y):
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()

    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='black')
    plt.scatter(X[neg, 0], X[neg, 1], c='yellow', edgecolors='black')

    

def draw_line(X, y, c):
    clf = svm.SVC(kernel='linear', C=c)
    clf.fit(X, y.ravel())

    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x1, x2 = np.meshgrid(x1, x2)
    
    y = clf.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)
    plt.contour(x1, x2, y)

#draw_graphic(X1, y1)

#draw_line(X1, y1, 1)
#draw_line(X1, y1, 100)

################################################

def gaussian_kernel(X, y, c, sigma):
    clf = svm.SVC(kernel='rbf', C=c, gamma=1 / (2 * sigma ** 2))
    clf.fit(X, y.ravel())

    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x1, x2 = np.meshgrid(x1, x2)
    y = clf.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape)
    plt.contour(x1, x2, y)

#draw_graphic(X2, y2)
#gaussian_kernel(X2, y2, 1, 0.1)

##################################################

def choose_params(X, y, X_val, y_val):
    C_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    scores = np.zeros((len(C_vec), len(sigma_vec)))
    best_score = [0, 0]

    for c in range(len(C_vec)):
        for s in range(len(sigma_vec)):
            clf = svm.SVC(kernel='rbf', C=C_vec[c], gamma= 1 / ( 2 * sigma_vec[s] ** 2))
            clf.fit(X, y)
            scores[c][s] = clf.score(X_val, y_val)

            if scores[c][s] > scores[best_score[0]][best_score[1]]:
                best_score[0] = c
                best_score[1] = s
    #print(f"Best accuracy: {scores[best_score[0]][best_score[1]] * 100} %")
    return (C_vec[best_score[0]], sigma_vec[best_score[1]])

#draw_graphic(X3, y3)
#c, sigma = choose_params(X3, y3, X_val, y_val)
#gaussian_kernel(X3, y3, c, sigma)

#plt.show()
