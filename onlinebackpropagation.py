# Standard library imports
import random

# Numpy related imports
import numpy as np

# Local imports
import time

from costfunction import SECost
from trainer import Trainer

class OnlineBackPropagation(Trainer):
    """
    A trainer.
    This is an online backpropagation trainer.
    """

    def __init__(self, maxEpochs, minAvgDesc):
        super(OnlineBackPropagation, self).__init__\
            (maxEpochs, minAvgDesc, SECost())

    def train(self, model, trainPatternList, eta, lmbda=0, suppressPrint=False, testPatternList=None):
        """
        Trains the given model on the pattern list.
        Callbacks the function plt passing to it the pair mean training error and epoch number.

        Warning (If testPatternList is given): the so-called "test set" (actually it may be a validation set) is used
        **ONLY** to produce MSE and performance values for intermediate models (for logging purposes),
        this set has no effect on train!

        :param model: Model to be trained
        :type model: Model
        :param trainPatternList: The train pattern list
        :type trainPatternList: [(np.array, np.array)]
        :param eta: The learning rate
        :type eta: float
        :param lmbda: The regularization hyper-parameter
        :type lmbda: float
        :param suppressPrint: Should the function avoid prints?
        :type suppressPrint: bool
        :param testPatternList: Should the function use this second set and compute error and performace?
        :type testPatternList: [(np.array, np.array)]
        :return: A tuple containing error and performance measure
        :rtype: Tuple
        """
        if not suppressPrint:
            print("##### Online training #####")
            if not (testPatternList is None):
                print("Train patterns #{}; Test patterns #{}".format(len(trainPatternList), len(testPatternList)))
            else:
                print("Train patterns #{}".format(len(trainPatternList)))

            print(
                "Hyper-parameters: eta (learning rate): {} - lambda (regularization): {}".format(
                    eta, lmbda))
            print("Max #epochs: {}".format(self._maxEpochs))
            print("##### Training started at: {} #####".format(time.strftime("%c")))

        numberOfPatterns = len(trainPatternList)

        trainMSE = []
        trainAccuracy = []

        testMSE = []
        testAccuracy = []

        epoch = 0

        start = time.time()

        self.initModel(model)

        gradAvg = 0  # Rolling average of 2-norm of descent over epochs

        while epoch == 0 or not (epoch >= self._maxEpochs or gradAvg < self._minAvgDesc):
            random.shuffle(trainPatternList)

            for pattern in trainPatternList:
                deltaW, deltaB = self.backpropagation(model, pattern)

                for l in range(1, model.layerCount):
                    model.layers[l].weights = (1-2*lmbda/numberOfPatterns)*model.layers[l].weights - eta*deltaW[l]
                    model.layers[l].bias = model.layers[l].bias - eta*deltaB[l]
                    gradAvg += np.linalg.norm(deltaW[l], 2)

                gradAvg /= model.layerCount

            trainMSE.append(self.meanerror(model, trainPatternList))
            trainAccuracy.append(self.performance(model, trainPatternList))

            if not (testPatternList is None):
                testMSE.append(self.meanerror(model, testPatternList))
                testAccuracy.append(self.performance(model, testPatternList))

            if not suppressPrint and not (testPatternList is None):
                print("Epoch {}/{}: Train MSE: {}, Train accuracy: {}; Test MSE: {}, Test accuracy: {}"
                      .format(epoch + 1, self._maxEpochs, trainMSE[-1], trainAccuracy[-1], testMSE[-1],
                              testAccuracy[-1]))
            elif not suppressPrint:
                print("Epoch {}/{}: Train MSE: {}, Train accuracy: {}"
                      .format(epoch + 1, self._maxEpochs, trainMSE[-1], trainAccuracy[-1]))

            epoch += 1

        end = time.time()

        if not suppressPrint:
            print("##### Training ended at: {} #####".format(time.strftime("%c")))
            print("##### Took {0:.2f} seconds. #####".format(end - start))

        return trainMSE, trainAccuracy, testMSE, testAccuracy
