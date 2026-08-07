#!/bin/python3
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# load the MNIST images from the IDX file format
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        # Read magic number and dimensions from the header
        magic, num, rows, cols = np.fromfile(f, dtype='>i4', count=4)
        # Load the rest of the data as unsigned bytes
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return images

images = load_mnist_images('train-images.idx3-ubyte')

# load the MNIST labels from the IDX file format
def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        # Read magic number and number of labels from the header
        magic, num = np.fromfile(f, dtype='>i4', count=2)
        # Load the rest of the data as unsigned bytes
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

labels = load_mnist_labels('train-labels.idx1-ubyte')

# Here i create the array of images of zeros, ones, twos and so on
def create_image_arrays(images, labels):
    image_arrays = {i: [] for i in range(10)}  # Create a dictionary to hold arrays for each digit
    for img, lbl in zip(images, labels):
        image_arrays[lbl].append(img)  # Append the image to the corresponding label's list
    return image_arrays

image_arrays = create_image_arrays(images, labels)

# Here i save the images to dir traindata/0-9/image_XXXXX.png
def save_images(image_arrays):
    for digit, imgs in image_arrays.items():
        dir_path = f'traindata/{digit}'
        os.makedirs(dir_path, exist_ok=True)  # Create directory if it doesn't exist
        for idx, img in enumerate(imgs):
            img_pil = Image.fromarray(img)  # Convert numpy array to PIL Image
            img_pil.save(f'{dir_path}/image_{idx:05d}.png')  # Save image with zero-padded index

save_images(image_arrays)
