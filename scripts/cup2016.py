import itertools
import random
import math
import copy
from collections import OrderedDict

import dill as pickle

import numpy as np
import time

from activationfunction import ActivationFunction
from batchbackpropagation import BatchBackPropagation
from layer import DenseLayer
from model import Model


def activationSoft():
    def s(x):
        if x < -45:
            return 0
        elif x > 45:
            return 1;
        else:
            return 1 / (1 + math.exp(-x))

    ds = lambda x: s(x) * (1 - s(x))

    def delta(a, b):
        if np.linalg.norm(a - b, 2) <= 0.1:
            return 1
        else:
            return 0

    return ActivationFunction(s, ds, delta)


def loadPatternList(fileName):
    f = open(fileName)
    patternList = []

    for line in f.readlines():
        if line != "\n" and not line.startswith("#"):
            # This is a pattern
            splitted = line.strip().split(',')[1:] # Ignore the ID
            splitted = [float(c) for c in splitted]

            patternList.append((np.array(splitted[0:10]).reshape(10, 1), np.array(splitted[10:]).reshape(2, 1)))
    f.close()
    return patternList

def loadBlindTest(fileName):
    f = open(fileName)
    patternDictionary = {}

    for line in f.readlines():
        if line != "\n" and not line.startswith("#"):
            # This is a pattern
            splitted = line.strip().split(',')

            id = int(splitted[0])
            input = [float(c) for c in splitted[1:]]

            patternDictionary[id] = np.array(input).reshape(10, 1)
    f.close()
    return patternDictionary

def writeBlindTestResults(fileName, dictionary):
    f = open(fileName, "w")

    f.write("# Matteo Busi\n")
    f.write("# mcaos\n")
    f.write("# (LOC-OSM2) - AA1 2016 CUP v1\n")
    f.write("# {}\n".format(time.strftime("%d %b %Y")))

    for id in dictionary:
        f.write("{}, {}, {}\n".format(id, dictionary[id][0][0], dictionary[id][1][0]))  # Transform it to a scalar. It's a numpy array

    f.close()

def createModel():
    # For the model we know that it should have at least
    #   - an input layer with 10 units
    #   - an output layer with 2 units, with a linear activation function
    inputSize = 10
    outputSize = 2
    nHiddenUnits = 20
    linearActivation = ActivationFunction(lambda x: x, lambda x: 1, lambda x, d: np.linalg.norm(x - d, 2))

    # Build the model:
    model = Model(inputSize)
    initialWeights = np.zeros([nHiddenUnits, inputSize])
    initialBias = np.zeros((nHiddenUnits, 1))
    model.addLayer(DenseLayer(inputSize, nHiddenUnits, initialWeights, initialBias, activationSoft()))

    initialWeights = np.zeros([outputSize, nHiddenUnits])
    initialBias = np.zeros((outputSize, 1))
    model.addLayer(DenseLayer(nHiddenUnits, outputSize, initialWeights, initialBias, linearActivation))

    return model

def gridSearch(model, nEpochs, minAvgDesc, trainPatternList, spaceGenerator, spaceSize, kFolds):
    trainPatternListLen = len(trainPatternList)
    # We use a k-fold-cross validation within train-set
    foldSize = trainPatternListLen // kFolds
    trainPatternListCross = [trainPatternList[i:i + foldSize] for i in range(0, trainPatternListLen, foldSize)]

    t = BatchBackPropagation(nEpochs, minAvgDesc)

    # Perform cross-validated model selection
    mseValidationMin = float("+inf")
    minimalHyper = ()
    minimalModel = None

    # Keep track of the results
    mseTrain = np.zeros((spaceSize, kFolds, nEpochs))
    meeTrain = np.zeros((spaceSize, kFolds, nEpochs))

    mseValidation = np.zeros((spaceSize, kFolds, nEpochs))
    meeValidation = np.zeros((spaceSize, kFolds, nEpochs))

    msNum = 0

    mseValidationAvg = 0

    for lmbda, eta, alpha in spaceGenerator:
        print("({}, {}, {}) - Model selection: {}/{}".format(lmbda, eta, alpha, msNum + 1, spaceSize))

        innerMinimalMSE = float("+inf")
        innerMinimalModel = None

        for k in range(kFolds):
            print("Current fold: {}/{}".format(k + 1, kFolds))
            # Properly divide the set
            currTrain = []
            for i in range(k):
                currTrain += trainPatternListCross[i]
            for i in range(k + 1, kFolds):
                currTrain += trainPatternListCross[i]

            currValidation = trainPatternListCross[k]

            # Shuffle the samples.
            random.shuffle(currTrain)

            # Train and get results
            mseTrain[msNum][k], meeTrain[msNum][k], mseValidation[msNum][k], meeValidation[msNum][k] = \
                t.train(model, currTrain, len(currTrain), eta, lmbda, alpha, True, currValidation)

            # Log validation mse to estimate the risk over the training set
            mseValidationAvg += mseValidation[msNum][k][-1]

            # Keep track of the best one
            if mseValidation[msNum][k][-1] < innerMinimalMSE:
                innerMinimalMSE = mseValidation[msNum][k][-1]
                innerMinimalModel = copy.deepcopy(model)

        mseValidationAvg /= kFolds # This is the avg of mseValidation, an estimation of risk over the whole training set

        # if the last mseValidationAvg is minimal thus far choose this parameter tuple
        if mseValidationAvg < mseValidationMin:
            mseValidationMin = mseValidationAvg
            minimalHyper = (lmbda, eta, alpha)
            minimalModel = copy.deepcopy(innerMinimalModel)
        msNum += 1

    # ok, here we've minimalHyper containing the result of parameter selection
    # retrain the model over the whole training set.
    print("Chosen hypers: lambda={}, eta={}, alpha={}".format(minimalHyper[0], minimalHyper[1], minimalHyper[2]))

    return minimalModel, minimalHyper, mseTrain, meeTrain, mseValidation, meeValidation


def searchAndPlot(model, trainPatternList, basename, nEpochs, kFolds, lmbdaRange, etaRange, alphaRange):
    # Generator for the parameter space
    spaceGenerator = itertools.product(lmbdaRange, etaRange, alphaRange)
    spaceSize = len(lmbdaRange) * len(etaRange) * len(alphaRange)

    # Perform the grid search, and produce the graph of results
    resultingModel, resultingMinimalHyper, mseTrain, meeTrain, mseValidation, meeValidation = \
        gridSearch(model, nEpochs, minAvgDesc, trainPatternList, spaceGenerator, spaceSize, kFolds)

    i = 0
    spaceGenerator = itertools.product(lmbdaRange, etaRange, alphaRange)

    for lmbda, eta, alpha in spaceGenerator:
        mseTCSV = open("plots/cup/{}_mse_train_{}.csv".format(basename, i), "w")
        meeTCSV = open("plots/cup/{}_mee_train_{}.csv".format(basename, i), "w")
        mseVCSV = open("plots/cup/{}_mse_val_{}.csv".format(basename, i), "w")
        meeVCSV = open("plots/cup/{}_mee_val_{}.csv".format(basename, i), "w")


        print("Producing: {} {} {}".format(lmbda, eta, alpha))

        mseTrainAvg = np.zeros(nEpochs)
        meeTrainAvg = np.zeros(nEpochs)

        mseValidationAvg = np.zeros(nEpochs)
        meeValidationAvg = np.zeros(nEpochs)

        for k in range(kFolds):
            # Write the value of err/perf for each fold/value

            for e in range(nEpochs):
                mseTCSV.write("{}, {}, {}\n".format(k, e, mseTrain[i][k][e]))
                meeTCSV.write("{}, {}, {}\n".format(k, e, meeTrain[i][k][e]))
                mseVCSV.write("{}, {}, {}\n".format(k, e, mseValidation[i][k][e]))
                meeVCSV.write("{}, {}, {}\n".format(k, e, meeValidation[i][k][e]))

            for e in range(nEpochs):
                mseTrainAvg[e] += mseTrain[i][k][e]
                meeTrainAvg[e] += meeTrain[i][k][e]

                mseValidationAvg[e] += mseValidation[i][k][e]
                meeValidationAvg[e] += meeValidation[i][k][e]


        # Compute the averages
        for e in range(nEpochs):
            mseTrainAvg[e] /= kFolds
            meeTrainAvg[e] /= kFolds

            mseValidationAvg[e] /= kFolds
            meeValidationAvg[e] /= kFolds

        print("Chosen avg.: MSE Train= {}, MEE Train={}, MSE Test={}, MEE Test={}"\
              .format(mseTrainAvg[-1], meeTrainAvg[-1], mseValidationAvg[-1], meeValidationAvg[-1]))

        i+=1


    return resultingModel, resultingMinimalHyper


if __name__ == "__main__":
    # Load existing models/parameters?
    loadFirst = True
    loadFiner = True
    loadFinal = False

    # Parameters
    trainPatternListRatio = 0.75
    minAvgDesc = 0.0

    # Read the input file
    patternList = loadPatternList("datasets/cup/LOC-OSM2-TR.csv")

    # Split the dataset.
    trainPatternListLen = int(len(patternList) * trainPatternListRatio)

    trainPatternList = patternList[0:trainPatternListLen]
    testPatternList = patternList[trainPatternListLen:]

    # Perform some pre-processing.
    # BEWARE: use the same mean/avg value both on train and testing set
    mean = (np.mean([x[0] for x in trainPatternList]), np.mean([x[1] for x in trainPatternList]))
    std = (np.std([x[0] for x in trainPatternList]), np.std([x[1] for x in trainPatternList]))

    trainPatternList = [(x - mean[0], d) for x, d in trainPatternList]
    trainPatternList = [(x / std[0], d) for x, d in trainPatternList]

    testPatternList = [(x - mean[0], d) for x, d in testPatternList]
    testPatternList = [(x / std[0], d) for x, d in testPatternList]

    # Do the first grid search. Coarser over a few fold
    nEpochs = 20
    kFolds = 3

    lmbdaRange = np.array([0, 0.001, 0.01])
    etaRange = np.array([0.01, 0.1, 0.3])
    alphaRange = np.array([0, 0.25, 0.5])

    if not loadFirst:
        model = createModel()
        firstModel, firstHyper = searchAndPlot(model, trainPatternList, "grid_coarse", nEpochs, kFolds, lmbdaRange, etaRange, alphaRange)

        # Save the first model.
        pickle.dump(firstModel, open("models/first.model", "wb"), pickle.HIGHEST_PROTOCOL)
        pickle.dump(firstHyper, open("models/first.hyper", "wb"), pickle.HIGHEST_PROTOCOL)

    else:
        print("Loading first model...")
        firstModel = pickle.load(open("models/first.model", "rb"))
        firstHyper = pickle.load(open("models/first.hyper", "rb"))

        print("Loaded hypers: lambda: {}, eta: {}, alpha: {}".format(firstHyper[0], firstHyper[1], firstHyper[2]))

    # Do a second grid search. More epochs and a restricted set of hypers
    nEpochs = 40
    kFolds = 5

    lmbdaEps = firstHyper[0]/2 + 0.0001
    etaEps = firstHyper[1] / 2 + 0.03
    alphaEps = firstHyper[2] / 2 + 0.025

    lmbdaRange = np.linspace(max(firstHyper[0]-lmbdaEps, 0), firstHyper[0] + lmbdaEps, 3)
    etaRange = np.linspace(max(firstHyper[1] - etaEps, 0.01), firstHyper[1] + etaEps, 5)
    alphaRange = np.linspace(max(firstHyper[2] - alphaEps, 0), firstHyper[2] + alphaEps, 3)

    if not loadFiner:
        model = createModel()
        finerModel, finerHyper = searchAndPlot(model, trainPatternList, "grid_finer", nEpochs, kFolds, lmbdaRange, etaRange,
                                               alphaRange)

        # Save the finer model.
        pickle.dump(finerModel, open("models/finer.model", "wb"), pickle.HIGHEST_PROTOCOL)
        pickle.dump(finerHyper, open("models/finer.hyper", "wb"), pickle.HIGHEST_PROTOCOL)

    else:
        print("Loading finer model...")

        finerModel = pickle.load(open("models/finer.model", "rb"))
        finerHyper = pickle.load(open("models/finer.hyper", "rb"))

        print("(Finer) Loaded hypers: lambda: {}, eta: {}, alpha: {}".format(finerHyper[0], finerHyper[1], finerHyper[2]))

    lmbda, eta, alpha = finerHyper[0], finerHyper[1], finerHyper[2]

    # Finally train it over the full training set
    nEpochs = 1000
    trialNum = 5
    minAvgDesc = 0.0
    t = BatchBackPropagation(nEpochs, minAvgDesc) # use a non 0 minAvgDsc to speed-up training

    # Train, get and write results
    if not loadFinal:
        mseTCSV = open("plots/cup/final_mse_train.csv", "w")
        meeTCSV = open("plots/cup/final_mee_train.csv", "w")
        mseVCSV = open("plots/cup/final_mse_val.csv", "w")
        meeVCSV = open("plots/cup/final_mee_val.csv", "w")

        finalModel = createModel()

        mseTrainAvg = np.zeros(nEpochs)
        meeTrainAvg = np.zeros(nEpochs)

        mseTestAvg = np.zeros(nEpochs)
        meeTestAvg = np.zeros(nEpochs)

        minimalMSETrain = float("+inf")
        minimalModel = None

        for trial in range(trialNum):
            random.shuffle(trainPatternList)
            mseTrain, meeTrain, mseTest, meeTest = \
                t.train(finalModel, trainPatternList, len(trainPatternList), eta, lmbda, alpha, False, testPatternList)

            print("Trial {} : MSE Train= {}, MEE Train={}, MSE Test={}, MEE Test={}".format(trial, mseTrain[-1], meeTrain[-1], mseTest[-1], meeTest[-1]))

            for epoch in range(nEpochs):
                mseTCSV.write("{}, {}, {}\n".format(trial, epoch, mseTrain[epoch]))
                meeTCSV.write("{}, {}, {}\n".format(trial, epoch, meeTrain[epoch]))
                mseVCSV.write("{}, {}, {}\n".format(trial, epoch, mseTest[epoch]))
                meeVCSV.write("{}, {}, {}\n".format(trial, epoch, meeTest[epoch]))

            mseTrainAvg += mseTrain
            meeTrainAvg += meeTrain
            mseTestAvg += mseTest
            meeTestAvg += meeTest

            if mseTrain[-1] < minimalMSETrain:
                minimalMSETrain = mseTrain[-1]
                minimalModel = copy.deepcopy(finalModel)

        # Compute the averages and plot them too
        mseTrainAvg /= trialNum
        meeTrainAvg /= trialNum

        mseTestAvg /= trialNum
        meeTestAvg /= trialNum

        print("Final training avg.: MSE Train= {}, MEE Train={}, MSE Test={}, MEE Test={}".format(mseTrainAvg[-1], meeTrainAvg[-1],
                                                                                            mseTestAvg[-1], meeTestAvg[-1]))

        # Save the final model.
        pickle.dump(finalModel, open("models/final.model", "wb"), pickle.HIGHEST_PROTOCOL)

    else:
        print("Loading final model...")
        finalModel = pickle.load(open("models/final.model", "rb"))

    # Ok, now run the blind test
    blindTestPatternDictionary = loadBlindTest("datasets/cup/LOC-OSM2-TS.csv") # This is a dictionary

    print("Blind test over {} samples.".format(len(blindTestPatternDictionary)))

    results = OrderedDict([]) # Associates id to result.

    for id in sorted(blindTestPatternDictionary):
        results[id] = finalModel.output(blindTestPatternDictionary[id])

    writeBlindTestResults("results/cup/mcaos_LOC-OSM2-TS.csv", results)

    # Done!

