from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
# Guilherme Domingos Faria Silva NUSP: 9361094
# Wallace Alves Esteves Manzano NUSP: 9790840

import numpy as np
from statistics import mean
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

# diretorios usados no código
preprocessed = 'preprocessed_data/'
pre_orl = preprocessed + 'orl'
pre_icmc = preprocessed + 'icmc'
pre_icmc_hogs = pre_icmc + '/hog'
pre_orl_hogs = pre_orl + '/hog'

# numpy arrays usados no código
fd_icmc = np.load(pre_icmc_hogs + '/fd.npy')
fd_orl = np.load(pre_orl_hogs + '/fd.npy')
labels_icmc = np.load(pre_icmc_hogs + '/labels.npy')
labels_orl = np.load(pre_orl_hogs + '/labels.npy')

# configurações do PCA que mantém a variância por volta de 50%
g_icmc = PCA(n_components=25)
g_orl = PCA(n_components=50)
pca_icmc = g_icmc.fit_transform(fd_icmc)
pca_orl = g_orl.fit_transform(fd_orl)

# função usada para testar os modelos recebidos como entrada no cross-validation, mais detalhes no relatório
def test(model, data, target):
    # lista de acuracias para cada fold
    acuracias = []

    # instanciação do método com 10-fold
    skf_10 = StratifiedKFold(n_splits=10)
    i = 1
    # percurso pelas 10 execuções
    cm_t = np.zeros(shape=(20, 20))
    for train_ind, test_ind in skf_10.split(data, target):
        print('\r\tFold %d' % i, end='')
        i += 1
		
		# divisão do dataset de treino e de teste e dos rótulos
        x_train, x_test = data[train_ind], data[test_ind]
        y_train, y_test = target[train_ind], target[test_ind]

        # treina o modelo na repartição de treino do dataset
        model.fit(x_train, y_train)

		# realiza a predição usando o modelo treinado acima
        y_pred = model.predict(x_test)

		# calcula a acurácia e a matriz de confusão
        acc_score = accuracy_score(y_test, y_pred)
        cm_t += confusion_matrix(y_test, y_pred)
        acuracias.append(acc_score)
    print('\r', end='')
	
	# retorna a lista de acurácias e uma matriz contendo todas as matrizes de confusão obtidas
    return {'acc': acuracias, 'cm': cm_t}

# função para instanciação dos modelos com base nos melhores parâmetros obtidos nos testes
# name = nome do dataset, data = dados de entrada, data_pcd = dados de dimensão reduzida, target = rótulos, pca = configuração do PCA
def test_dataset(name, data, data_pcd, target, pca):
    print('*************** TEST NO DATASET %s ***************' % name)
    knn_results = {}
    # *********************** KNN ***********************
	# inicializa o número de K ótimo, de maior acurácia
    kmax = None
	# percorre os 3 valores para K recomendados
    for i, k in enumerate([3, 5, 7]):
        print('%d-NN (%d/3)' % (k, i + 1))
		# chama a função test para executar o cross-validation no classificador e dados passados como argumento
        knn_results[k] = test(KNeighborsClassifier(n_neighbors=k), data, target)
		# calcula média das acurácias retornadas pela função test
        acc = mean(knn_results[k]['acc'])
        print('\tAcurácia média: %f' % acc)
		# compara a acurácia dos classificadores para decidir o melhor K
        if kmax is None or mean(knn_results[kmax]['acc']) < acc:
            kmax = k
        print()

    print('KNN com mair desempenho é com k = %d' % kmax)
    print('Acurácia média: %f' % mean(knn_results[kmax]['acc']))
    cm = knn_results[kmax]['cm']
	
	# percorre todas as classes exibindo a precisão para cada uma
    for i in range(20):
		# a precisão é = TP / TP + FP
        print('Precisão classe %d: %f' % (i+1, cm[i, i] / sum(cm[:, i])))
    print('Matriz de confusão')
    print(cm)

    # PCA
	# chama a função test para fazer o cross-validation em determinado classificador, porém com o PCA aplicado no dataset
    d = test(KNeighborsClassifier(n_neighbors=kmax), data_pcd, target)

    print('KNN com PCA k = %d | # componentes = %d | variância coberta = %f' % (kmax, pca.n_components_,
                                                                               sum(pca.explained_variance_ratio_)))
    print('Acurácia média: %f' % mean(d['acc']))
    cm = d['cm']
	# percorre todas as classes exibindo a precisão para cada uma (modo PCA)
    for i in range(20):
		# a precisão é = TP / TP + FP
        print('Precisão classe %d: %f' % (i+1, cm[i, i] / sum(cm[:, i])))
    print('Matriz de confusão')
    print(cm)

    # *********************** MLP ***********************
	# inicializa o número de camadas ótimo, de maior acurácia
    cmax = None
    mlp_results = {}
    it = 0
	# executa o MLP para as 4 configurações de camada abaixo, consideradas as melhores segundo os testes
    for c in [(10,), (20,), (10, 5), (15, 10)]:
		# executa o MLP para as 3 taxas de aprendizado consideradas as melhores segundo os testes
        for n in [0.1, 0.5, 10]:
			# executa o MLP para os 3 melhores momentums obtidos nos testes
            for a in [0.1, 0.5, 0.9]:
                ind = '%s | n = %f | a = %f' % (str(c), n, a)
                it += 1
                print('MLP camada escondida %s  (%d/36)' % (ind, it))
				# executa o cross-validation do MLP através da função test
                mlp_results[ind] = test(MLPClassifier(solver='sgd', hidden_layer_sizes=c, learning_rate_init=n,
                                                      momentum=a, random_state=1, max_iter=100, tol=1e-3), data, target)
                mlp_results[ind]['c'] = c
                mlp_results[ind]['n'] = n
                mlp_results[ind]['a'] = a
                acc = mean(mlp_results[ind]['acc'])
                print('\tAcurácia média: %f' % acc)
				# determina a melhor configuração do MLP de acordo com a acuracia
                if cmax is None or mean(mlp_results[cmax]['acc']) < acc:
                    cmax = ind
                print()

    print('MLP com mair desempenho é com as camadas = %s' % str(cmax))
    print('Acurácia média: %f' % mean(mlp_results[cmax]['acc']))
    cm = mlp_results[cmax]['cm']
	
	# obtém a precisão para cada uma das 20 classes
    for i in range(20):
		# a precisão é = TP / TP + FP
        print('Precisão classe %d: %f' % (i+1, cm[i, i] / sum(cm[:, i])))
    print('Matriz de confusão')
    print(cm)


    # PCA
	# executa o cross-validation do MLP através da função test, em modo PCA
    d = test(MLPClassifier(solver='sgd', hidden_layer_sizes=mlp_results[cmax]['c'],
                                          learning_rate_init=mlp_results[cmax]['n'], momentum=mlp_results[cmax]['a'],
                                          random_state=1, max_iter=100, tol=1e-3), data_pcd, target)

    print('MLP com PCA camadas = %s | # componentes = %d | variância coberta = %f' %
          (cmax, pca.n_components_, sum(pca.explained_variance_ratio_)))
    print('Acurácia média: %f' % mean(d['acc']))
    cm = d['cm']
	# obtém a precisão para cada uma das 20 classes (modo PCA)
    for i in range(20):
		# a precisão é = TP / TP + FP
        print('Precisão classe %d: %f' % (i+1, cm[i, i] / sum(cm[:, i])))
    print('Matriz de confusão')
    print(cm)


# chama as funções iniciais para os dois datasets de imagens
test_dataset('ICMC', fd_icmc, pca_icmc, labels_icmc, g_icmc)
test_dataset('ORL', fd_orl, pca_orl, labels_orl, g_orl)
