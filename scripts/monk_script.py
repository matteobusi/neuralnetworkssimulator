import math
import random

import numpy as np

from activationfunction import ActivationFunction
from batchbackpropagation import BatchBackPropagation
from layer import DenseLayer
from model import Model
from onlinebackpropagation import OnlineBackPropagation


def activationSigmoid():
    s = lambda x: 1.0 / (1.0 + math.exp(-x))
    ds = lambda x: s(x) * (1 - s(x))

    def delta(a, b):
        if (a <= 0.5 and b == 0) or (a > 0.5 and b == 1):
            return 1
        else:
            return 0

    return ActivationFunction(s, ds, delta)


def activationTanh():
    ds = lambda x: 1 - math.tanh(x)**2

    def delta(a, b):
        if (a <= 0 and b == 0) or (a > 0 and b == 1):
            return 1
        else:
            return 0

    return ActivationFunction(math.tanh, ds, delta)


def loadPatternList(fileName):
    f = open(fileName)
    patternList = []
    for line in f:
        splitted = line.strip().split(' ')[0:-1]
        splitted = [int(c) for c in splitted]
        patternList.append((splitted[1:7], splitted[0]))
    return patternList


def encodePatternList(patternList):
    def mapOnOne(x):
        if x == 1:
            return [1]
        else:
            return [0]

    def mapOnTwo(x):
        if x == 1:
            return [0, 1]
        else:
            return [1, 0]

    def mapOnThree(x):
        if x == 1:
            return [0, 0, 1]
        elif x == 2:
            return [0, 1, 0]
        else:
            return [1, 0, 0]

    def mapOnFour(x):
        if x == 1:
            return [0, 0, 0, 1]
        elif x == 2:
            return [0, 0, 1, 0]
        elif x == 3:
            return [0, 1, 0, 0]
        else:
            return [1, 0, 0, 0]

    encodedList = []
    for p in patternList:
        first = [mapOnThree(p[0][0]), mapOnThree(p[0][1]), mapOnTwo(p[0][2]), mapOnThree(p[0][3]),
                 mapOnFour(p[0][4]), mapOnTwo(p[0][5])]
        flattened_first = np.array([y for x in first for y in x])[:, None]

        second = [mapOnOne(p[1])]
        flattened_second = np.array([y for x in second for y in x])[:, None]

        encodedList.append((flattened_first, flattened_second))

    return encodedList


def printConfusionMatrix(model, testPatternList):
    conf = [[0, 0], [0,0]]

    for x, d in testPatternList:
        o = model.layers[-1].layerActivation.performance(model.output(x), d)
        conf[o][d] += 1

    print(conf)


def monkTrain(hiddenUnits, trainFile, testFile, graphFile, maxEpochs, minAvgDesc, eta, lmbda, trialNum=1, mode="b", alpha=0, minibatchSize=-1):
    """
    Helper function for monk's training.
    Produces plots and everything from a trainFile and a testFile.
    Graph file is the path of the graph.
    Mode decides whether we should use batch (b), online (o). Default is b.

    Some of the hyperparameters are not used in some mode. e.g. alpha and minibatchSize are ignored in mode "o"


    """
    # Load train data
    trainPatternList = loadPatternList(trainFile)
    testPatternList = loadPatternList(testFile)

    # Preprocessing the input
    encodedTrainPatternList = encodePatternList(trainPatternList)
    encodedTestPatternList = encodePatternList(testPatternList)

    # mean = (np.mean([x[0] for x in encodedTrainPatternList]), np.mean([x[1] for x in encodedTrainPatternList]))
    # std = (np.std([x[0] for x in encodedTrainPatternList]), np.std([x[1] for x in encodedTrainPatternList]))
    #
    # encodedTrainPatternList = [(x - mean[0], d) for x, d in encodedTrainPatternList]
    # encodedTrainPatternList = [(x / std[0], d) for x, d in encodedTrainPatternList]
    #
    # encodedTestPatternList = [(x - mean[0], d) for x, d in encodedTestPatternList]
    # encodedTestPatternList = [(x / std[0], d) for x, d in encodedTestPatternList]

    # Set hyp pars for the nn
    inputSize = 17
    hiddenLayerSize = hiddenUnits
    outputSize = 1

    # Force-set some of the hyper parameters.
    if (mode == "b"):
        minibatchSize = len(encodedTrainPatternList)

    mseTRCSV = open(graphFile.format("mse_train"), "w")
    accTRCSV = open(graphFile.format("acc_train"), "w")
    mseTSCSV = open(graphFile.format("mse_test"), "w")
    accTSCSV = open(graphFile.format("acc_test"), "w")

    for trial in range(trialNum):
        random.shuffle(encodedTrainPatternList)

        print("###### Trial {}/{} #####".format(trial+1, trialNum))
        # Create the model, with the input layer
        m = Model(inputSize)

        # Add the hidden layer
        initialWeights = np.zeros([hiddenLayerSize, inputSize])
        initialBias = np.zeros((hiddenLayerSize, 1))
        m.addLayer(DenseLayer(inputSize, hiddenLayerSize, initialWeights, initialBias, activationSigmoid()))

        # Add the output layer
        initialWeights = np.zeros([outputSize, hiddenLayerSize])
        initialBias = np.zeros((outputSize, 1))
        m.addLayer(DenseLayer(hiddenLayerSize, outputSize, initialWeights, initialBias, activationSigmoid()))

        if mode == "o":
            t = OnlineBackPropagation(maxEpochs, minAvgDesc)
            trainMSE, trainAccuracy, testMSE, testAccuracy = t.train(m, encodedTrainPatternList, eta, lmbda, False,
                                                                     encodedTestPatternList)
        else:
            t = BatchBackPropagation(maxEpochs, minAvgDesc)
            # Now, train!
            trainMSE, trainAccuracy, testMSE, testAccuracy = t.train(m, encodedTrainPatternList, minibatchSize, eta,
                                                                     lmbda, alpha, False,
                                                                     encodedTestPatternList)
        # Write the plot data
        for epoch in range(maxEpochs):
            mseTRCSV.write("{}, {}, {}\n".format(trial, epoch, trainMSE[epoch]))
            accTRCSV.write("{}, {}, {}\n".format(trial, epoch, trainAccuracy[epoch]))
            mseTSCSV.write("{}, {}, {}\n".format(trial, epoch, testMSE[epoch]))
            accTSCSV.write("{}, {}, {}\n".format(trial, epoch, testAccuracy[epoch]))

        printConfusionMatrix(m, encodedTestPatternList)