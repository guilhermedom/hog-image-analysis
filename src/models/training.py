# Guilherme Domingos Faria Silva
# Wallace Alves Esteves Manzano

import warnings

from statistics import mean

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

import numpy as np

warnings.filterwarnings("ignore")

# path variables
preprocessed = '../../data/processed/'
pre_orl = preprocessed + 'orl/'
pre_icmc = preprocessed + 'icmc/'
pre_icmc_hogs = pre_icmc + 'hog/'
pre_orl_hogs = pre_orl + 'hog/'

# read numpy arrays from preprocessed images
fd_icmc = np.load(pre_icmc_hogs + 'fd.npy')
fd_orl = np.load(pre_orl_hogs + 'fd.npy')
labels_icmc = np.load(pre_icmc_hogs + 'labels.npy')
labels_orl = np.load(pre_orl_hogs + 'labels.npy')

# PCA configuration that keep explained variance at around 50%, a percentage that does not disturb classifiers
g_icmc = PCA(n_components=25)
g_orl = PCA(n_components=50)
pca_icmc = g_icmc.fit_transform(fd_icmc)
pca_orl = g_orl.fit_transform(fd_orl)

# test input models in a cross-validation manner
def test(model, data, target):
    # accuracies list for each fold
    accuracies = []

    # instantiate 10-fold method
    skf_10 = StratifiedKFold(n_splits=10)
    i = 1
    # 20x20 matrix to save one confusion matrix for each one of the 20 classes from input
    confusion_matrix_list = np.zeros(shape=(20, 20))

    # goes through each fold
    for train_ind, test_ind in skf_10.split(data, target):
        print('\r\tFold %d' % i, end='')
        i += 1
		
        x_train, x_test = data[train_ind], data[test_ind]
        y_train, y_test = target[train_ind], target[test_ind]

        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        acc_score = accuracy_score(y_test, y_pred)
        confusion_matrix_list += confusion_matrix(y_test, y_pred)
        accuracies.append(acc_score)
    print('\r', end='')
	
    return {'acc': accuracies, 'cm': confusion_matrix_list}

# search for best models using a small grid search
def test_dataset(dataset_name, data, data_reduced_dim, target, pca):
    print('*************** TESTING IN DATASET %s ***************' % dataset_name)
    knn_results = {}

    # *********************** KNN ***********************
	# instantiate optimal k, having the best accuracy
    kmax = None
    for i, k in enumerate([3, 5, 7]):
        print('%d-NN (%d/3)' % (k, i + 1))

		# call cross-validation method
        knn_results[k] = test(KNeighborsClassifier(n_neighbors=k), data, target)

		# get the average accuracy for all folds
        acc = mean(knn_results[k]['acc'])
        print('\tAverage accuracy: %f' % acc)

        if kmax is None or mean(knn_results[kmax]['acc']) < acc:
            kmax = k
        print()

    print('Best KNN has k = %d' % kmax)
    print('Average accuracy: %f' % mean(knn_results[kmax]['acc']))
    cm = knn_results[kmax]['cm']
	
	# show precision for all classes
    for i in range(20):
		# precision = True Positives / True Positives + False Positives
        print('Precision for class %d: %f' % (i+1, cm[i, i] / sum(cm[:, i])))
    print('Confusion matrix')
    print(cm)

    # PCA
	# run cross-validation using KNN in the dataset with reduced dimensionality
    d = test(KNeighborsClassifier(n_neighbors=kmax), data_reduced_dim, target)

    print('KNN with PCA and k = %d | # components = %d | explained variance = %f' % (kmax, pca.n_components_,
                                                                               sum(pca.explained_variance_ratio_)))
    print('Average accuracy: %f' % mean(d['acc']))
    cm = d['cm']
	# show precision for all classes (with PCA dimensionality reduction)
    for i in range(20):
		# precision = True Positives / True Positives + False Positives
        print('Precision for class %d: %f' % (i + 1, cm[i, i] / sum(cm[:, i])))
    print('Confusion matrix')
    print(cm)

    # *********************** MLP ***********************
	# instantiate best number of neurons per hidden layer, having the best accuracy
    hl_max = None
    mlp_results = {}
    it = 0
	# start grid search changing number of hidden layers, number of neurons, momentum and learning rate
    for neurons_per_layer in [(10,), (20,), (10, 5), (15, 10)]:
        for learning_rate in [0.1, 0.5, 1]:
            for momentum in [0.1, 0.5, 0.9]:
                ind = '%s | learning_rate = %f | momentum = %f' % (str(neurons_per_layer), learning_rate, momentum)
                it += 1
                print('MLP hidden layers %s  (%d/36)' % (ind, it))

				# call cross-validation method
                mlp_results[ind] = test(MLPClassifier(solver='sgd', hidden_layer_sizes=neurons_per_layer, learning_rate_init=learning_rate,
                                                      momentum=momentum, random_state=1, max_iter=100, tol=1e-3), data, target)
                mlp_results[ind]['neurons_per_layer'] = neurons_per_layer
                mlp_results[ind]['learning_rate'] = learning_rate
                mlp_results[ind]['momentum'] = momentum

                # get the average accuracy for all folds
                acc = mean(mlp_results[ind]['acc'])
                print('Average accuracy: %f' % acc)

                if hl_max is None or mean(mlp_results[hl_max]['acc']) < acc:
                    hl_max = ind
                print()

    print('Best MLP has hidden layers = %s' % str(hl_max))
    print('Average accuracy: %f' % mean(mlp_results[hl_max]['acc']))
    cm = mlp_results[hl_max]['cm']
	
	# show precision for all classes
    for i in range(20):
		# precision = True Positives / True Positives + False Positives
        print('Precision for class %d: %f' % (i + 1, cm[i, i] / sum(cm[:, i])))
    print('Confusion matrix')
    print(cm)


    # PCA
	# run cross-validation using KNN in the dataset with reduced dimensionality
    d = test(MLPClassifier(solver='sgd', hidden_layer_sizes=mlp_results[hl_max]['neurons_per_layer'],
                                          learning_rate_init=mlp_results[hl_max]['learning_rate'], momentum=mlp_results[hl_max]['momentum'],
                                          random_state=1, max_iter=100, tol=1e-3), data_reduced_dim, target)

    print('MLP with PCA and hidden layers = %s | # components = %d | explained variance = %f' %
          (hl_max, pca.n_components_, sum(pca.explained_variance_ratio_)))
    print('Average accuracy: %f' % mean(d['acc']))
    cm = d['cm']
	# show precision for all classes (with PCA dimensionality reduction)
    for i in range(20):
		# precision = True Positives / True Positives + False Positives
        print('Precision for class %d: %f' % (i+1, cm[i, i] / sum(cm[:, i])))
    print('Confusion matrix')
    print(cm)

def main():
    # run starting functions for both datasets
    test_dataset('ICMC', fd_icmc, pca_icmc, labels_icmc, g_icmc)
    test_dataset('ORL', fd_orl, pca_orl, labels_orl, g_orl)

if __name__ == "__main__":
    main()