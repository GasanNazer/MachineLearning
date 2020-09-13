import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import os
import imageio

def create_label(Y, folder):
    if folder == "dog":
        val = [1, 0, 0]
        Y.append(val)
    elif folder == "cat":
        val = [0, 1, 0]
        Y.append(val)
    elif folder == "elephant":
        val = [0, 0, 1]
        Y.append(val)


def convert_png_image_to_jpg(path):
    im1 = Image.open(path)
    plt.imshow(im1)
    
    if im1.mode in ("RGBA", "P"):
        im1 = im1.convert("RGB")
    
    im1.save(path.replace(".png", ".jpg"))
    if os.path.exists(path):
        os.remove(path)


def covert_all_png_images_to_jpg():
    folder = "images"
    for subfolder in os.listdir(folder):
            subfolder_complete_path = os.path.join(folder, subfolder)
            for filename in os.listdir(subfolder_complete_path):
                if ".png" in filename:
                    print(filename)
                    convert_png_image_to_jpg(os.path.join(subfolder_complete_path, filename))


def load_images_from_folder(Y, num_px = 64, folder="images"):
    images = []
    for subfolder in os.listdir(folder):
        subfolder_complete_path = os.path.join(folder, subfolder)
        for filename in os.listdir(subfolder_complete_path):
            create_label(Y, subfolder)
            img = np.array(imageio.imread(os.path.join(subfolder_complete_path, filename)))
            if img is not None:
                img = np.array(Image.fromarray(img).resize(size=(num_px, num_px))) # resize the image
                print(filename)
                print(img.shape)
                img = np.reshape(img, (1, num_px*num_px*3)).T
                images.append(img)
    images = np.array(images)
    return np.reshape(images, (images.shape[0], images.shape[1])).T


def print_images(images):
    for index in range(images.shape[1]):
        plt.figure()
        plt.imshow(images[:,index].reshape((num_px, num_px, 3)))

