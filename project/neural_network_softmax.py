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


def softmax(z):
    t = np.exp(z)
    A = t / np.sum(t, axis= 0)
    cache = z 
    return A, cache

def relu(Z):
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache

def relu_backward(dA, cache):
    
    Z = cache
    dZ = np.array(dA, copy=True)
    
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)
    
    W1 = np.random.randn(n_h, n_x)*0.01 
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters 

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
     
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    elif activation == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters, C):
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network
    
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev,parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A,parameters["W" + str(L)], parameters["b" + str(L)], "softmax")
    caches.append(cache)
    assert(AL.shape == (C, X.shape[1]))
            
    return AL, caches

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims = True) / m
    dA_prev = np.dot(W.T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
      
    if activation == "relu":
        dZ = relu_backward(dA, cache[1])
        dA_prev, dW, db = linear_backward(dZ, cache[0])
    if activation == "softmax":
        dZ = dA
        dA_prev, dW, db = linear_backward(dZ, cache[0])
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    #Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    dAL = AL - Y.T
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = linear_activation_backward(dAL, caches[L - 1], "softmax")
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = current_cache
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        current_cache = linear_activation_backward(grads["dA" + str(l + 1)], caches[l], "relu")
        dA_prev_temp, dW_temp, db_temp = current_cache
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    return parameters

def compute_cost(X, AL, Y):
    m = X.shape[1]
    cost = np.sum(Y * np.log(AL.T) + 1e-8) / -m
    return cost

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False, C = 3):
    np.random.seed(1)
    costs = []                         # keep track of cost
    
    parameters = initialize_parameters_deep(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters, C)

        cost = compute_cost(X, AL, Y)
        grads = L_model_backward(AL, Y, caches)
 
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    '''
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    '''
    return parameters, costs

def one_hot_reverse(Y):
    H = np.argmax(Y, axis=1)
    return H.reshape((len(H), 1))

def predict_nn(X, Y, parameters, C=3):
    probas, caches = L_model_forward(X, parameters, C)
    
    index_max = np.argmax(probas, axis = 0)
    index_max = index_max.reshape((len(index_max), 1))
    accuracy = np.sum(one_hot_reverse(Y) % C == index_max) / Y.shape[0]
    print("Accuracy: " + str(accuracy * 100) + '%')

def predict_one_example_nn(index, X, parameters, C=3):
    probas, caches = L_model_forward(X, parameters, C)
    type_animal = np.argmax(probas, axis = 0)
    if type_animal[index] == 0:
        print("I am a dog", end =" ")
    elif type_animal[index] == 1:
        print("I am a cat", end =" ")
    elif type_animal[index] == 2:
        print("I am an elephant", end =" ")
    probabilty = probas[:,index][np.argmax(probas[:,index])]
    print("with probability of " + "{:.2f}".format(probabilty * 100) + "%.")
    plt.figure()
    #plt.imshow(images[:,index].reshape((num_px, num_px, 3)))

#predict_one_example_nn(2291, X, parameters)


C = 3
#layers_dims = [12288, 20, 7, 5, C] #  4-layer model
#layers_dims = [12288, 20, C] #  2-layer model learning_rate = 0.005, 10000 iterations accuracy = 79.5
layers_dims = [12288, 40, C] # 2 layers with learning_rate = 0.0075, num_iterations = 20000 accuracy=100
#layers_dims = [12288, 20, 7, C] #  3-layer model learning_rate = 0.009, 10000 iterations accuracy = 68
#params, _ = L_layer_model(X_test, Y_test, layers_dims, learning_rate = 0.0075, num_iterations = 100, print_cost = True, C = C)


#predict_nn(X_test, Y_test, params)

def calculate_probability(parameters, X, Y, C=3):
    probas, _ = L_model_forward(X, parameters, C)
    index_max = np.argmax(probas, axis = 0)
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

def choose_rate(X, y, X_val, y_val):
    lr = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 1])
    costs_X = np.zeros(len(lr))
    costs_X_val = np.zeros(len(lr))
    for rate in range(len(lr)):
            _ , cost_X = L_layer_model(X, y, layers_dims, learning_rate = lr[rate], num_iterations = 500, print_cost = True)
            costs_X[rate] = cost_X[-1]
            _, cost_X_val = L_layer_model(X_val, y_val, layers_dims, learning_rate = lr[rate], num_iterations = 500, print_cost = True)
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
    lr = np.array([100, 200, 400, 500, 800, 1000])
    costs_X = np.zeros(len(lr))
    costs_X_val = np.zeros(len(lr))
    for rate in range(len(lr)):
            _ , cost_X = L_layer_model(X, y, layers_dims, learning_rate = 0.03, num_iterations = lr[rate], print_cost = True)
            costs_X[rate] = cost_X[-1]
            _, cost_X_val = L_layer_model(X_val, y_val, layers_dims, learning_rate = 0.03, num_iterations = lr[rate], print_cost = True)
            costs_X_val[rate] = cost_X_val[-1]
    print(costs_X)
    print(costs_X_val)
    plt.plot(lr, costs_X)
    plt.xlabel("iterations")
    plt.plot(lr, costs_X_val)
    plt.ylabel("cost")
    plt.title("Choosing correct #iteration")
    plt.show()

#choose_iterations(X_train, Y_train, X_dev, Y_dev)

params, _ = L_layer_model(X_test, Y_test, layers_dims, learning_rate = 0.03, num_iterations = 800, print_cost = True, C = C)
#params, _ = L_layer_model(X_test, Y_test, layers_dims, learning_rate = 0.03, num_iterations = 100, print_cost = True, C = C)

print("Accuracy training set:")
calculate_probability(params, X_train, Y_train)
print("Accuracy validation set:")
calculate_probability(params, X_dev, Y_dev)
print("Accuracy test set:")
calculate_probability(params, X_test, Y_test)

Y = []

test_hq = load_images_from_folder(Y, num_px = 256 ,folder = folder_test)
test_hq = test_hq/256

def predict_one_example_nn(index, X, parameters, C=3):
    probas, caches = L_model_forward(X, parameters, C)
    type_animal = np.argmax(probas, axis = 0)
    message = ""
    if type_animal[index] == 0:
        message += "I am a dog "
    elif type_animal[index] == 1:
        message += "I am a cat"
    elif type_animal[index] == 2:
        message += "I am an elephant"
    probability = probas[:,index][np.argmax(probas[:,index])]
    message += " with probability of " + "{:.2f}".format(probability * 100) + "%."
    plt.figure()
    plt.xlabel(message)
    plt.imshow(X[:,index].reshape((num_px, num_px, 3)))
    plt.show()

    plt.figure()
    plt.xlabel(message)
    plt.imshow(test_hq[:,index].reshape((256, 256, 3)))
    plt.show()

    


predict_one_example_nn(2, X_test, params)
predict_one_example_nn(20, X_test, params)
predict_one_example_nn(100, X_test, params)
predict_one_example_nn(110, X_test, params)

predict_one_example_nn(150, X_test, params)
predict_one_example_nn(160, X_test, params)
predict_one_example_nn(151, X_test, params)
predict_one_example_nn(161, X_test, params)
predict_one_example_nn(152, X_test, params)
predict_one_example_nn(162, X_test, params)

predict_one_example_nn(290, X_test, params)
predict_one_example_nn(320, X_test, params)

predict_one_example_nn(330, X_test, params)
predict_one_example_nn(340, X_test, params)

