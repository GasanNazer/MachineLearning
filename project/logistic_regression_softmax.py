import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from utils import load_images_from_folder, print_images

Y_train = [] # labels are created during execution time
Y_dev = []
Y_test = []
num_px = 64 # during execution images are resized to 64x64x3. This way we lose their quality but we save computational time. 
C = 3 # number of classes to detect


folder_train = "images_train"
folder_dev = "images_dev"
folder_test = "images_test"

images_train = load_images_from_folder(Y_train, folder= folder_train)
X_train = images_train / 255 # normalize dataset

images_dev = load_images_from_folder(Y_dev, folder= folder_dev)
X_dev = images_dev / 255 # normalize dataset

images_test = load_images_from_folder(Y_test, folder= folder_test)
X_test = images_test / 255 # normalize dataset


Y_train = np.array(Y_train) # convert the labels Y list into a numpy array
Y_dev = np.array(Y_dev)
Y_test = np.array(Y_test)

### activation function - softmax
def softmax(z):
    t = np.exp(z)
    return t / np.sum(t, axis= 0)


def initialize(dim):
    
    w = np.random.randn(dim, C) * 0.01
    b = 0
    
    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]
    
    A = softmax(np.dot(w.T, X) + b)
    cost = np.sum(Y * np.log(A.T) + 1e-8) / -m
    dw = np.dot(X, np.transpose(A - Y.T)) / m
    db = np.sum((A - Y.T)) / m

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost
    

w, b = initialize(X_train.shape[0])
grads, cost = propagate(w, b, X_train, Y_train)


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False): # clean the code  
    costs = []

    print(f"Learning rate {learning_rate}")
    
    for i in range(num_iterations):
        # Compute gradients and cost
        grads, cost = propagate(w, b, X, Y)

        # Gradients
        dw = grads["dw"]
        db = grads["db"]
        
        # Update weight and bias
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

def one_hot_reverse(Y):
    H = np.argmax(Y, axis=1)
    return H.reshape((len(H), 1))

def predict(w, b, X, Y):
    A = softmax(np.dot(w.T, X) + b)
    
    index_max = np.argmax(A, axis = 0)
    index_max = index_max.reshape((len(index_max), 1))
    accuracy = np.sum(one_hot_reverse(Y) % C == index_max) / Y.shape[0]
    print("Accuracy: " + str(accuracy * 100) + '%')

def predict_one_example(index):
    softmax_picture = softmax(np.dot(X[:,index], params["w"]) + params["b"])
    type_animal = np.argmax(softmax_picture)
    message = ""
    if type_animal == 0:
        message += "I am a dog "
    elif type_animal == 1:
        message += "I am a cat"
    elif type_animal == 2:
        message += "I am an elephant"
    message += "with probability of " + "{:.2f}".format(softmax_picture[type_animal] * 100) + "%."
    plt.figure()
    plt.xlabel(message)
    plt.imshow(images[:,index].reshape((num_px, num_px, 3)))

#predict_one_example(2001)

plt.show()

# Precision/Recall


def choose_rate(X, y, X_val, y_val):
    lr = np.array([0, 0.001, 0.003, 0.01]) #0.03, 0.1, 0.3
    costs_X = np.zeros(len(lr))
    costs_X_val = np.zeros(len(lr))
    for rate in range(len(lr)):
            _, _, cost_X = optimize(w, b, X, y, num_iterations= 500, learning_rate = lr[rate], print_cost = True)
            costs_X[rate] = cost_X[-1]
            _, _, cost_X_val = optimize(w, b, X_val, y_val, num_iterations= 500, learning_rate = lr[rate], print_cost = True)
            costs_X_val[rate] = cost_X_val[-1]
    print(costs_X)
    print(costs_X_val)
    plt.plot(lr, costs_X)
    plt.xlabel("learning rate")
    plt.plot(lr, costs_X_val)
    plt.ylabel("cost")
    plt.title("Choosing correct learning rate")
    plt.show()

#choose_rate(X_train, Y_train, X_dev, Y_dev)


def choose_iterations(X, y, X_val, y_val):
    lr = np.array([100, 200, 400, 500, 800, 1000, 2000])
    costs_X = np.zeros(len(lr))
    costs_X_val = np.zeros(len(lr))
    for rate in range(len(lr)):
            _, _, cost_X = optimize(w, b, X, y, num_iterations= lr[rate], learning_rate = 0.001, print_cost = True)
            costs_X[rate] = cost_X[-1]
            _, _, cost_X_val = optimize(w, b, X_val, y_val, num_iterations= lr[rate], learning_rate = 0.001, print_cost = True)
            costs_X_val[rate] = cost_X_val[-1]
    print(costs_X)
    print(costs_X_val)
    plt.plot(lr, costs_X)
    plt.xlabel("iterations")
    plt.plot(lr, costs_X_val)
    plt.ylabel("cost")
    plt.show()

#choose_iterations(X_train, Y_train, X_dev, Y_dev)

def calculate_probability(w, b, X, Y, C=3):
    A = softmax(np.dot(w.T, X) + b)
    
    index_max = np.argmax(A, axis = 0)
    index_max = index_max.reshape((len(index_max), 1))
    accuracy = np.sum(one_hot_reverse(Y) % C == index_max) / Y.shape[0]
    print("Accuracy: " + str(accuracy * 100) + '%')
    
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

    print("average dogs: " + str((precision[0] + recall[0]) / 2))
    print("average cats: " + str((precision[1] + recall[1]) / 2))
    print("average elephants: " + str((precision[2] + recall[2]) / 2))

    f1_score_dogs = 2 * precision[0] * recall[0] / (precision[0] + recall[0])
    print("f1 score dogs: " + str(f1_score_dogs))
    f1_score_cats = 2 * precision[1] * recall[1] / (precision[1] + recall[1])
    print("f1 score cats: " + str(f1_score_cats))
    f1_score_elephants = 2 * precision[2] * recall[2] / (precision[2] + recall[2])
    print("f1 score elephants: " + str(f1_score_elephants))

# Running the model
params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations= 400, learning_rate = 0.001, print_cost = True)

print("Accuracy training set:")
calculate_probability(params["w"], params["b"], X_train, Y_train)
print("Accuracy validation set:")
calculate_probability(params["w"], params["b"], X_dev, Y_dev)
print("Accuracy test set:")
calculate_probability(params["w"], params["b"], X_test, Y_test)