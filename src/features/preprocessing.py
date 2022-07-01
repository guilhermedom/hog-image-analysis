# Guilherme Domingos Faria Silva
# Wallace Alves Esteves Manzano

import shutil
from os.path import join
from os import listdir, path
import os
import random
import math
import numpy as np

from matplotlib import pyplot as plt

from skimage import transform
from skimage import exposure
from skimage.io import imread
from skimage.util import random_noise
from skimage.feature import hog

from scipy import ndimage

# path variables
orlfacesdir = '../../data/raw/OrlFaces20'
pessoalicmcdir = '../../data/raw/PessoasICMC'
preprocessed = '../../data/processed/'
pre_orl = preprocessed + 'orl/'
pre_icmc = preprocessed + 'icmc/'
pre_icmc_images = pre_icmc + 'images'
pre_orl_images = pre_orl + 'images'
pre_icmc_hogs = pre_icmc + 'hog'
pre_orl_hogs = pre_orl + 'hog'

# plot or not the generated images
PLOT = False

# data augmentation for the ICMC dataset
def gen_data(base_image, quantity, label):
    data = []
    labels = []
	# for all input images
    for _ in range(0, quantity):
        pepper_seed = random.randint(10, 20)
        rotate_degrees = random.uniform(-10, 10)
		
		# resize image
        img = transform.resize(base_image, (112, 92), anti_aliasing=True)
		
		# add random pepper effect as noise to the image
        img = random_noise(img, mode='pepper', seed=pepper_seed, clip=True)
		
		# randomly rotate image by a maximum of 10 degrees
        img = transform.rotate(img, rotate_degrees, resize=False, center=None, order=1, mode='constant', cval=1,
                               clip=True, preserve_range=True)
        # add blur effect to the image
        img = ndimage.gaussian_filter(img, sigma=2)

        data.append(img)
        labels.append(label)
		
    return data, labels

# load all images in input path
def load_images(path):
    images = []
    labels = []
    dictionary = {}
	
	# run through all subdirectories and collect their images
    for _, d in enumerate(listdir(path)):
        if d in dictionary.values():
            l = dictionary[d]
        else:
            l = len(dictionary.keys()) + 1
            dictionary[l] = d
        d = join(path, d)
        for i in listdir(d):
            image = imread(join(d, i), as_gray=True)
            images.append(image)
            labels.append(l)
    return images, labels, dictionary

# plot images in input array. Images are put together in a matrix
def plot_images(array, rows, columns):
    fig = plt.figure(figsize=(columns * 2, rows * 2))
    for i, j in enumerate(array):
        ax = fig.add_subplot(rows, columns, i + 1)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.grid(False)
        plt.imshow(j, cmap=plt.cm.gray)

    plt.show()

# generate HOG feature descriptors and their visualization
def hog_images(imgs):
    feature_descriptors_list, hog_imgs = [], []
    for i, image in enumerate(imgs):
		# generate HOG features using a specific configuration that creates them fast 
        # without causing performance loss in classifiers
        feature_descriptors, hog_image = hog(image, orientations=9, pixels_per_cell=(4, 4),
                            cells_per_block=(1, 1), visualize=True, multichannel=False,
                            transform_sqrt=True, block_norm="L1")
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        feature_descriptors_list.append(feature_descriptors)
        hog_imgs.append(hog_image_rescaled)
        print('\r\tHOG %d/%d generated' % (i + 1, len(imgs)), end='')
    print()
    return feature_descriptors_list, hog_imgs

def main():
    print('Loading base dataset icmc')
    images, labels_a, dictionary_icmc = load_images(pessoalicmcdir)
    # plot images if PLOT constant is set to true
    if PLOT:
        plot_images(images, len(images) / 4, 4)
        print(labels_a)
    print('Dataset loaded')

    print('Generating dataset icmc')
    dataset_icmc, labels_icmc = [], []
    for im, l in zip(images, labels_a):
        a, b = gen_data(im, 10, l)
        dataset_icmc += a
        labels_icmc += b
    if PLOT:
        plot_images(dataset_icmc, math.ceil(len(dataset_icmc) / 4), 4)
        print(labels_icmc)
    print('Dataset generated')

    print('Loading dataset orl')
    dataset_orl, labels_orl, dictionary_orl = load_images(orlfacesdir)
    if PLOT:
        plot_images(dataset_orl, math.ceil(len(dataset_orl) / 4), 4)
        print(labels_orl)
    print('Dataset loaded')


    print('Generating HOG icmc')
    feature_descriptors_icmc, hog_icmc = hog_images(dataset_icmc)
    if PLOT:
        plot_images(hog_icmc, math.ceil(len(hog_icmc) / 4), 4)
    print('HOG generated')


    print('Generating HOG orl')
    feature_descriptors_orl, hog_orl = hog_images(dataset_orl)
    if PLOT:
        plot_images(hog_orl, math.ceil(len(hog_orl) / 4), 4)
    print('HOG generated')

    # create directories, in case they do not exist, to save images
    print('Creating output directories')
    if path.exists(preprocessed):
        shutil.rmtree(preprocessed)
    os.mkdir(preprocessed)
    os.mkdir(pre_icmc)
    os.mkdir(pre_orl)
    os.mkdir(pre_icmc_images)
    os.mkdir(pre_orl_images)
    os.mkdir(pre_icmc_hogs)
    os.mkdir(pre_orl_hogs)
    print('Directories created')

    print('Saving icmc images')
    for i, im, lb in zip(range(len(dataset_icmc)), dataset_icmc, labels_icmc):
        if not path.exists(pre_icmc_images + '/p' + str(lb)):
            os.mkdir(pre_icmc_images + '/p' + str(lb))
        plt.imsave(pre_icmc_images + '/p' + str(lb) + '/' + str(i) + '.png', im, cmap=plt.cm.gray)
    print('Images saved')

    print('Saving orl images')
    for i, im, lb in zip(range(len(dataset_orl)), dataset_orl, labels_orl):
        if not path.exists(pre_orl_images + '/p' + str(lb)):
            os.mkdir(pre_orl_images + '/p' + str(lb))
        plt.imsave(pre_orl_images + '/p' + str(lb) + '/' + str(i) + '.png', im, cmap=plt.cm.gray)
    print('Images saved')


    print('Saving icmc HOG')
    np.save(pre_icmc_hogs + '/fd', np.array(feature_descriptors_icmc))
    np.save(pre_icmc_hogs + '/labels', np.array(labels_icmc))
    print('HOG features and images saved')

    print('Saving orl HOG')
    np.save(pre_orl_hogs + '/fd', np.array(feature_descriptors_orl))
    np.save(pre_orl_hogs + '/labels', np.array(labels_orl))
    print('HOG features and images saved')


if __name__ == "__main__":
    main()
