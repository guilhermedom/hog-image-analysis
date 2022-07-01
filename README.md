# HOG image features analysis with KNN and MLP
A data science analysis using KNN and MLP algorithms on HOG image feature descriptors. In this study, we use two collections of images as datasets. Firstly, the HOG descriptors are extracted from each image in a preprocessing step. Then, they are passed to a hyperparameter search grid for KNN and MLP models. Lastly, the impact of dimensionality reduction on the images, using PCA, is analyzed.


## Experiments
The objective is to use simple classifiers (KNN and MLP) to experiment with HOG features, check the resulting performance in terms of accuracy and precision. Another point of this exploratory analysis is to check the robustness of the HOG features after their dimensionality is reducted, by running the same classifiers on the reduced feature space.

To achieve the results, two image datasets, that have each 10 images of 20 different persons, are classified into 20 classes. Each class belongs to one, and only one, person. Hence, the objective is to recognize a couple of different individuals.

### Datasets
The first dataset is "icmc", a collection of 20 images from 20 different professors and employees from University of Sao Paulo (USP). The images are publicly available and were obtained from the [university website]. Since this datasets has originally only 20 images, data augmentations techniques were applied to transform the single image from each person into 10 different images.

The second dataset is "orl", a collection of 400 images from 40 different persons publicly available on the web. In this case, no data augmentation was necessary, as there was already 10 different images of each individual. Only 20 persons were selected for this experiments, making 200 images.

Both raw and processed datasets are available under the "data" directory of this repository.

### Grid search


### PCA
To take our dimensionality reduction analysis to the extreme, we use PCA so that only 50% of the original variance is explained. Even with such strong reductions, both classifiers performed well in their tasks, indicating the robustness of the HOG feature descriptors:


## What is HOG?
[Histogram of Oriented Gradients] (HOG) is a feature extraction technique for images. The general idea is that images can be divided into cells that are small connected portions of the image. For the pixels of each cell, a histogram of gradient directions is constructed and the final feature descriptor is the concetenation of these histograms. Overall, the appearance of image objects and their shapes are described by the distribution of intensity gradients, i.e., edge directions and sizes.

The HOG features can be visualized as "vectors" inside cells of the image:

![hog](https://user-images.githubusercontent.com/33037020/176957511-c6e82268-89c3-4548-822e-d3d86b74b182.png)

HOG has been known to be particularly useful to detect humans in images as pointed out in the paper "Dalal, N., & Triggs, B. (2005, June). Histograms of oriented gradients for human detection. In 2005 IEEE computer society conference on computer vision and pattern recognition (CVPR'05) (Vol. 1, pp. 886-893). Ieee.".

[//]: #
[Histogram of Oriented Gradients]: https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
[university website]: https://icmc.usp.br/pessoas
: https://www.kaggle.com/datasets/tavarez/the-orl-database-for-training-and-testing
