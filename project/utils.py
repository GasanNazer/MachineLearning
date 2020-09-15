import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import os
import imageio
import shutil


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
                #print(filename)
                #print(img.shape)
                if img.shape != (num_px, num_px, 3): # delete all images with icorrect shapes
                    os.remove(os.path.join(subfolder_complete_path, filename))
                    #print(os.path.join(subfolder_complete_path, filename))
                else:
                    img = np.reshape(img, (1, num_px*num_px*3)).T
                    images.append(img)
    images = np.array(images)
    return np.reshape(images, (images.shape[0], images.shape[1])).T


def print_images(images, num_px = 64):
    for index in range(images.shape[1]):
        plt.figure()
        plt.imshow(images[:,index].reshape((num_px, num_px, 3)))


def separate_dataset(X):
    print("a")
    print(X.shape)
    np.random.shuffle(X.T)
    print(X)
    print(np.random.shuffle(X.T))
    print(X.shape)

    #print_images(X)

    X_splitted = X[:]
    X_train = X[:,:5]
    X_dev = X_splitted[:,5:8]
    X_test = X_splitted[:,8:9]
    print("shapes")
    print(X_train.shape)
    print(X_dev.shape)
    print(X_test.shape)

    print_images(X_train)
    #print_images(X_dev)
    #print_images(X_test)

    plt.show()


def separate_dataset_folders(max_examples = 1400):
    # max_examples is the number of files found for every 
    # class(in this case as for class-elephant, there are only 1440 
    # although for dog and cats there are more this the maximum number
    # of images that our model is going to use)

    train = max_examples * 80 / 100
    dev = max_examples * 10 / 100
    test = max_examples * 10 / 100

    folder = "images"
    folder_train = "images_train"
    folder_dev = "images_dev"
    folder_test = "images_test"

    for subfolder in os.listdir(folder):
            subfolder_complete_path = os.path.join(folder, subfolder)
            subfolder_complete_path_train = os.path.join(folder_train, subfolder)
            subfolder_complete_path_dev = os.path.join(folder_dev, subfolder)
            subfolder_complete_path_test = os.path.join(folder_test, subfolder)

            for i in [subfolder_complete_path_train, subfolder_complete_path_dev, subfolder_complete_path_test]:
                if not os.path.exists(i):
                    os.makedirs(i)

            count_train = 0
            count_dev = 0
            count_test = 0
            files_read = 0

            for filename in os.listdir(subfolder_complete_path):
                print(filename)
                if max_examples > files_read:
                    if count_train < train:
                        os.replace(os.path.join(subfolder_complete_path, filename), os.path.join(subfolder_complete_path_train, filename))
                        count_train += 1
                    elif count_dev < dev:
                        os.replace(os.path.join(subfolder_complete_path, filename), os.path.join(subfolder_complete_path_dev, filename))
                        count_dev += 1
                    elif count_test < test:
                        os.replace(os.path.join(subfolder_complete_path, filename), os.path.join(subfolder_complete_path_test, filename))
                        count_test += 1
                    files_read += 1
                else:
                    break

    shutil.rmtree(folder)


def delete_not_translated_folders(dog_folder, cat_folder, elephant_folder, images_path):
    for directory in os.listdir(images_path):
                directory = os.path.join(images_path, directory)
                if directory not in [dog_folder, elephant_folder, cat_folder]:
                     shutil.rmtree(directory)


def change_folders_names():
    folder_name = "59760_840806_bundle_archive"
    import zipfile
    
    with zipfile.ZipFile(folder_name + ".zip", 'r') as zip_ref:
        zip_ref.extractall(folder_name)

    
    dog_folder = "cane"
    elephant_folder = "elefante"
    cat_folder = "gatto"
    if os.path.exists(folder_name):
        images_path = os.path.join(folder_name, "raw-img")
        if os.path.exists(images_path):
            if os.path.exists("images"):
                os.rmdir("images")

            os.rename(images_path, "images")
            images_path = os.path.join("images", "")

            if os.path.exists(folder_name):
                shutil.rmtree(folder_name)

            dog_folder = os.path.join(images_path, dog_folder)

            if os.path.exists(dog_folder):
                 os.rename(dog_folder, os.path.join(images_path, "dog"))
                 dog_folder = os.path.join(images_path, "dog")

            elephant_folder = os.path.join(images_path, elephant_folder)

            if os.path.exists(elephant_folder):
                 os.rename(elephant_folder, os.path.join(images_path, "elephant"))
                 elephant_folder = os.path.join(images_path, "elephant")
            
            cat_folder = os.path.join(images_path, cat_folder)

            if os.path.exists(cat_folder):
                 os.rename(cat_folder, os.path.join(images_path, "cat"))
                 cat_folder = os.path.join(images_path, "cat")
            
            delete_not_translated_folders(dog_folder, cat_folder, elephant_folder, images_path)


def prepare_dataset():
    change_folders_names()
    covert_all_png_images_to_jpg()
    separate_dataset_folders()

