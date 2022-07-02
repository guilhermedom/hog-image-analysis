# HOG image features analysis with KNN and MLP
A data science analysis using KNN and MLP algorithms on HOG image feature descriptors. In this study, we use two collections of images as datasets. Firstly, the HOG descriptors are extracted from each image in a preprocessing step. Then, they are passed to a hyperparameter search grid for KNN and MLP models. Lastly, the impact of dimensionality reduction on the images, using PCA, is analyzed.


## Experiments
The objective is to use simple classifiers (KNN and MLP) to experiment with HOG features, check the resulting performance in terms of accuracy and precision. Another point of this exploratory analysis is to check the robustness of the HOG features after their dimensionality is reducted, by running the same classifiers on the reduced feature space.

To achieve the results, two image datasets, that have each 10 images of 20 different persons, are classified into 20 classes. Each class belongs to one, and only one, person. Hence, the objective is to recognize a couple of different individuals.

### Datasets
The first dataset is "icmc", a collection of 20 images from 20 different professors and employees from University of Sao Paulo (USP). The images are publicly available and were obtained from the [university website]. Since this datasets has originally only 20 images, data augmentations techniques were applied to transform the single image from each person into 10 different images.

The second dataset is "orl" ([ORL Database of Faces]), a collection of 400 images from 40 different persons publicly available on the web. In this case, no data augmentation was necessary, as there was already 10 different images of each individual. Only 20 persons were selected for this experiments, making 200 images.

Both raw and processed datasets are available under the "data" directory of this repository.

### Grid search
We performed a stratified cross-validation with 10 folds to evaluate each of the following hyperparameter configurations:

<center>

| KNN k | MLP neurons per layer | MLP learning rate | MLP momentum |
|:---:|:---------------------:|:-----------------:|:------------:|
| 3 | (10,) | 0.1 | 0.1 |
| 5 | (20,) | 0.5 | 0.5 |
| 7 | (10, 5) | 1.0 | 0.9 |
| | (15, 10) | |

</center>

After taking the average of the 10 folds for each hyperparameter configuration, these accuracy results were achieved:

| KNN k | Accuracy on ICMC | Accuracy on ORL |
|:-----:|:----------------:|:---------------:|
| 3 | 0.995 | 0.945 |
| 5 | 0.985 | 0.895 |
| 7 | 0.960 | 0.870 |

| MLP neurons per layer | MLP learning rate | MLP momentum | Accuracy on ICMC | Accuracy on ORL |
|:---------------------:|:-----------------:|:------------:|:----------------:|:---------------:|
| (10,) | 0.1 | 0.1 | 0.990 | 0.930 |
| (10,) | 0.1 | 0.5 | 0.990 | 0.940 |
| (10,) | 0.1 | 0.9 | 0.990 | 0.925 |
| (10,) | 0.5 | 0.1 | 0.160 | 0.060 |
| (10,) | 0.5 | 0.5 | 0.090 | 0.050 |
| (10,) | 0.5 | 0.9 | 0.095 | 0.050 |
| (10,) | 1.0 | 0.1 | 0.060 | 0.050 |
| (10,) | 1.0 | 0.5 | 0.060 | 0.050 |
| (10,) | 1.0 | 0.9 | 0.075 | 0.050 |
| (20,) | 0.1 | 0.1 | 1.0 | 0.955 |
| (20,) | 0.1 | 0.5 | 1.0 | 0.960 |
| (20,) | 0.1 | 0.9 | 1.0 | 0.960 |
| (20,) | 0.5 | 0.1 | 0.570 | 0.050 |
| (20,) | 0.5 | 0.5 | 0.320 | 0.050 |
| (20,) | 0.5 | 0.9 | 0.250 | 0.060 |
| (20,) | 1.0 | 0.1 | 0.060 | 0.050 |
| (20,) | 1.0 | 0.5 | 0.075 | 0.050 |
| (20,) | 1.0 | 0.9 | 0.075 | 0.055 |
| (10, 5) | 0.1 | 0.1 | 0.845 | 0.225 |
| (10, 5) | 0.1 | 0.5 | 0.770 | 0.215 |
| (10, 5) | 0.1 | 0.9 | 0.850 | 0.095 |
| (10, 5) | 0.5 | 0.1 | 0.105 | 0.050 |
| (10, 5) | 0.5 | 0.5 | 0.070 | 0.050 |
| (10, 5) | 0.5 | 0.9 | 0.060 | 0.050 |
| (10, 5) | 1.0 | 0.1 | 0.055 | 0.060 |
| (10, 5) | 1.0 | 0.5 | 0.060 | 0.050 |
| (10, 5) | 1.0 | 0.9 | 0.050 | 0.050 |
| (15, 10) | 0.1 | 0.1 | 0.990 | 0.880 |
| (15, 10) | 0.1 | 0.5 | 0.990 | 0.835 |
| (15, 10) | 0.1 | 0.9 | 0.975 | 0.310 |
| (15, 10) | 0.5 | 0.1 | 0.250 | 0.095 |
| (15, 10) | 0.5 | 0.5 | 0.115 | 0.050 |
| (15, 10) | 0.5 | 0.9 | 0.075 | 0.055 |
| (15, 10) | 1.0 | 0.1 | 0.070 | 0.050 |
| (15, 10) | 1.0 | 0.5 | 0.055 | 0.050 |
| (15, 10) | 1.0 | 0.9 | 0.050 | 0.050 |

With best KNN having k = 3 (0.995 accuracy on ICMC and 0.945 on ORL) and best MLP having 20 neurons and only one hidden layer, learning rate 0.1 and momentum 0.1 (1.0 accuracy on ICMC and 0.960 on ORL).

### PCA
To take our dimensionality reduction analysis to the extreme, we use PCA so that only 50% of the original variance is explained. Even with such strong reductions, both classifiers performed well in their tasks, indicating the robustness of the HOG feature descriptors:


## What is HOG?
[Histogram of Oriented Gradients] (HOG) is a feature extraction technique for images. The general idea is that images can be divided into cells that are small connected portions of the image. For the pixels of each cell, a histogram of gradient directions is constructed and the final feature descriptor is the concetenation of these histograms. Overall, the appearance of image objects and their shapes are described by the distribution of intensity gradients, i.e., edge directions and sizes.

The HOG features can be visualized as "vectors" inside cells of the image, before they are concatenated:

![hog](https://user-images.githubusercontent.com/33037020/176983585-9c7efeb6-cd62-4c86-b533-9334f01972e9.png)

HOG has been known to be particularly useful to detect humans in images as pointed out in the paper "Dalal, N., & Triggs, B. (2005, June). Histograms of oriented gradients for human detection. In 2005 IEEE computer society conference on computer vision and pattern recognition (CVPR'05) (Vol. 1, pp. 886-893). Ieee.".

[//]: #
[Histogram of Oriented Gradients]: https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
[university website]: https://icmc.usp.br/pessoas
[ORL Database of Faces]: https://www.kaggle.com/datasets/tavarez/the-orl-database-for-training-and-testing
