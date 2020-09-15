import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage


Y = []

def create_label(folder):
    if folder == "dog":
        val = [1, 0, 0]
        Y.append(val)
    elif folder == "cat":
        val = [0, 1, 0]
        Y.append(val)
    elif folder == "elephant":
        val = [0, 0, 1]
        Y.append(val)



import os

folder = "images"

def convert_png_image_to_jpg(path):
    im1 = Image.open(path)
    plt.imshow(im1)
    
    if im1.mode in ("RGBA", "P"):
        im1 = im1.convert("RGB")
    
    im1.save(path.replace(".png", ".jpg"))
    if os.path.exists(path):
        os.remove(path)

for subfolder in os.listdir(folder):
        subfolder_complete_path = os.path.join(folder, subfolder)
        for filename in os.listdir(subfolder_complete_path):
            if ".png" in filename:
                print(filename)
                convert_png_image_to_jpg(os.path.join(subfolder_complete_path, filename))



import os
import imageio

num_px = 64
C = 3 # number of classes

def load_images_from_folder(folder="images"):
    images = []
    for subfolder in os.listdir(folder):
        subfolder_complete_path = os.path.join(folder, subfolder)
        count = 0
        for filename in os.listdir(subfolder_complete_path):
            create_label(subfolder)
            img = np.array(imageio.imread(os.path.join(subfolder_complete_path, filename)))
            if img is not None:
                img = np.array(Image.fromarray(img).resize(size=(num_px, num_px))) # resize the image
                print(filename)
                print(img.shape)
                img = np.reshape(img, (1, num_px*num_px*3)).T
                images.append(img)
                count = count + 1
            if count == 1000:
                break
    images = np.array(images)
    return np.reshape(images, (images.shape[0], images.shape[1])).T

images = load_images_from_folder()
plt.imshow(images[:,0].reshape((num_px, num_px, 3)))
X = images / 255


def print_images():
    for index in range(images.shape[1]):
        plt.figure()
        plt.imshow(images[:,index].reshape((num_px, num_px, 3)))


Y = np.array(Y)
Labels = Y.T
print(Labels)
print(Labels.shape)


def softmax(z):
    t = np.exp(z)
    A = t / np.sum(t, axis= 0)
    cache = z 
    return A, cache

softmax(np.array([[2, 3], [2, 3]]))



def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ


