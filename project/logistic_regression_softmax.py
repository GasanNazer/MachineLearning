import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from utils import load_images_from_folder, print_images

Y = [] # labels are created during execution time
num_px = 64 # during execution images are resized to 64x64x3. This way we lose their quality but we save computational time. 
C = 3 # number of classes to detect


#images = load_images_from_folder(Y)
#plt.imshow(images[:,0].reshape((num_px, num_px, 3))) # showing the first example
#X = images / 255 # normalize dataset

folder_train = "images_train"
folder_dev = "images_dev"
folder_test = "images_test"

images_train = load_images_from_folder(Y, folder= folder_train)
X_train = images_train / 255 # normalize dataset

images_dev = load_images_from_folder(Y, folder= folder_dev)
X_dev = images_dev / 255 # normalize dataset

images_test = load_images_from_folder(Y, folder= folder_test)
X_test = images_test / 255 # normalize dataset


print(images_train.shape)
print(images_dev.shape)
print(images_test.shape)





'''

Y = np.array(Y) # convert the labels Y list into a numpy array

### activation funtion - softmax
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
    

w, b = initialize(X.shape[0])
grads, cost = propagate(w, b, X, Y)


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False): # clean the code  
    costs = []
    
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

# Running the model
params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.05, print_cost = True)

print("Final weights, bias and gradients: ")
print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))


def one_hot_reverse():
    H = np.argmax(Y, axis=1)
    return H.reshape((len(H), 1))

def predict(w, b, X):
    A = softmax(np.dot(w.T, X) + b)
    
    index_max = np.argmax(A, axis = 0)
    index_max = index_max.reshape((len(index_max), 1))
    accuracy = np.sum(one_hot_reverse() % C == index_max) / Y.shape[0]
    print("Accuracy: " + str(accuracy * 100) + '%')
    
predict(params["w"], params["b"], X)

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

predict_one_example(2001)

plt.show()


# include dev, test sets
# include learning curves
# Precision/Recall

'''