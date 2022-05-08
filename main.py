"""
Luyckx Marco
Projet d'année de 3eme bachelier : Analyse de la précision et des performances pour MNIST
Inspiré de : https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
"""
import datetime
import numpy

from numpy import mean
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.optimizers import SGD


def prepareData(reduceDataSetSize=False):
    """
    Chargements des données en mémoire
    :return: les données formatées correctement
    """
    (trainX, trainY), (testX, testY) = mnist.load_data()
    if reduceDataSetSize:
        n = 1

        trainX, trainY = shuffle(numpy.array(trainX), numpy.array(trainY))
        trainX = trainX[0:int(len(trainX) / n)]
        trainY = trainY[0:int(len(trainY) / n)]

    # Reformatage des données
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))

    # convertion en float
    trainX = trainX.astype('float32')
    testX = testX.astype('float32')

    # redimensionne les valeurs pour qu'elles soient entre 0 et 1
    trainX = trainX / 255.0
    testX = testX / 255.0

    # la sortie est un chiffre de 0 à 9 donc on besoin de 10 classes. Sur ces 10 classes, une seule valeur
    # sera égale à 1 et les 9 autres seront égales à 0
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    return trainX, trainY, testX, testY


def defineModel():
    """
    Fonction qui va définir notre modèle.
    Cette fonction est composée de 2 couches importantes : les couches de convolution et les couches de pooling.

    Description des paramètres pour les couches de convolution :
        32 : nombre de filtre
        (3, 3) : taille du filtre
        activation='relu' : j'utilise une Rectified Linear Activation Function. De plus, c'est une bonne pratique.
        kernel_initializer='he_uniform' : He = schéma d'initialisation du poids
        input_shape=(28, 28, 1) : les images seront de taille 28x28
    Description des paramètres pour les couches de pooling :
        (2, 2) : choisit la valeur maximale dans une matrice de taille (2*2)

    :return: le modèle
    """
    model = Sequential()  # définition vide du modèle
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())  # permet de standardiser les ouptputs.
    model.add(MaxPooling2D((2, 2)))

    # pour les tests, je fais varier le nombre de couches ci-dessous
    # ---
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    # ---

    model.add(Flatten())  # aplatit l'output reçu des couches précédentes
    model.add(Dense(100, activation='relu',
                    kernel_initializer='he_uniform'))  # permet d'interpreter les caractéristiques avec 100 noeuds
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))  # les 10 noeuds de sortie correspond aux chiffres de 0 à 9

    # compile le modèle
    opt = SGD(learning_rate=0.01, momentum=0.9)  # signifie Stochastic Gradient Descent
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def evaluateModel(dataX, dataY, nbEpoch, n=5):
    """
    Permet d'évaluer le modèle avec un "n fold cross-validation" qui est une technique pour évaluer un modèle.


    :param dataX: le jeu de données
    :param dataY: le jeu de données
    :param nbEpoch: epoch
    :param n: l'ensemble des données seront divisés en n groupes (5 par défaut, ce qui veut dire qu'on
    divise 60 000 par 5 = 12 000)

    :return: données pour les graphiques
    """
    scores = []
    histories = []
    kfold = KFold(n, shuffle=True, random_state=1)  # on mélange le jeu de données avant de le découper n fois

    for train_ix, test_ix in kfold.split(dataX):
        model = defineModel()

        # selectionne les lignes pour train et test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]

        # entraine le modèle
        history = model.fit(trainX, trainY, epochs=nbEpoch, batch_size=32, validation_data=(testX, testY),
                            verbose=0)

        # évalue la précision du modèle
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('La précision de notre modèle est de :  %.3f' % (acc * 100.0))

        # pour les graphiques
        scores.append(acc)
        histories.append(history)

    return scores, histories


# plot diagnostic learning curves
def summarize_diagnostics(histories, nbEpoch):
    for i in range(len(histories)):
        # plot accuracy
        plt.rcParams['font.size'] = '16'
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='red', label='test')
        plt.xlabel("Nombre d'epoch")
        plt.ylabel("Taux de précision")
        plt.xlim(0, nbEpoch - 1)
        plt.ylim(0.95, 1)    # (0.965, 1) => epoch
    print("Date de fin : " + str(datetime.datetime.now()))
    plt.show()


def main():
    print("Date de début : " + str(datetime.datetime.now()))
    trainX, trainY, testX, testY = prepareData()  # mettre True pour tester le dataSet
    nbEpoch = 10
    scores, histories = evaluateModel(trainX, trainY, nbEpoch)
    print('Précision moyenne : %.3f' % (mean(scores) * 100))
    summarize_diagnostics(histories, nbEpoch)


main()
