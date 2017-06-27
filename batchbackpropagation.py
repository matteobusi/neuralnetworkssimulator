# Standard library imports
import random

# Numpy related imports
import numpy as np

# Local imports
import time

from costfunction import SECost
from trainer import Trainer

class BatchBackPropagation(Trainer):
    """
    A trainer.
    This is an SGD/Batch backpropagation trainer.
    """

    def __init__(self, maxEpochs, minAvgDesc):
        super(BatchBackPropagation, self).__init__(maxEpochs, minAvgDesc, SECost())

    def minibatchTrain(self, model, minibatch):
        """
        (Helper function)
        This function trains the given model over the specified minibatch with given hyper-parameters.

        :param model: Model to be trained
        :type model: Model
        :param minibatch: The minibatch
        :type minibatch: int
        :return: A pair, delta values for weights and bias for each layer. An approximation of the gradient
        :rtype: ([np.array()], [np.array()])
        """
        minibatchSize = len(minibatch)

        sDeltaB = [np.zeros(0)] + [np.zeros(l.bias.shape) for l in model.layers[1:]]
        sDeltaW = [np.zeros(0)] + [np.zeros(l.weights.shape) for l in model.layers[1:]]

        for pattern in minibatch:
            deltaW, deltaB = self.backpropagation(model, pattern)
            for l in range(model.layerCount):
                sDeltaW[l] += deltaW[l]
                sDeltaB[l] += deltaB[l]

        for l in range(model.layerCount):
             sDeltaW[l] /= minibatchSize
             sDeltaB[l] /= minibatchSize

        return sDeltaW, sDeltaB


    def train(self, model, trainPatternList, minibatchSize, eta, lmbda=0, alpha=0, suppressPrint=False, testPatternList=None):
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
        :param minibatchSize: The size of minibatches
        :type minibatchSize: int
        :param eta: The learning rate
        :type eta: float
        :param lmbda: The regularization hyper-parameter
        :type lmbda: float
        :param alpha: The momentum hyper-parameter
        :type alpha: float
        :param suppressPrint: Should the function avoid prints?
        :type suppressPrint: bool
        :param testPatternList: Should the function use this second set and compute error and performace?
        :type testPatternList: [(np.array, np.array)]
        :return: A tuple containing error and performance measure
        :rtype: Tuple
        """
        if not suppressPrint:
            print("##### Batch/SGD training #####")
            if not (testPatternList is None):
                print("Train patterns #{}; Test patterns #{}".format(len(trainPatternList), len(testPatternList)))
            else:
                print("Train patterns #{}".format(len(trainPatternList)))

            print("(Mini-)Batch size: {}".format(minibatchSize))
            print("Hyper-parameters: eta (learning rate): {} - lambda (regularization): {} - alpha (momentum): {}".format(eta, lmbda, alpha))
            print("Max #epochs: {}".format(self._maxEpochs))
            print("##### Training started at: {} #####".format(time.strftime("%c")))

        trainMSE = []
        trainAccuracy = []

        testMSE = []
        testAccuracy = []

        epoch = 0

        start = time.time()

        self.initModel(model)

        gradAvg = 0 # Rolling average of 2-norm of descent for each minibatch over epochs

        # As a simple stopping criteria use a rolling average over the gradient. If too small stop.
        while epoch == 0 or not (epoch >= self._maxEpochs or gradAvg < self._minAvgDesc):
            random.shuffle(trainPatternList)

            minibatches = [trainPatternList[k:k+minibatchSize] for k in range(0, len(trainPatternList), minibatchSize)]

            deltaW = [np.zeros(0)] + [np.zeros(l.weights.shape) for l in model.layers[1:]]

            for minibatch in minibatches:
                deltaWtmp, deltaB = self.minibatchTrain(model, minibatch)

                for l in range(model.layerCount):
                    deltaW[l] = eta*deltaWtmp[l] + alpha*deltaW[l] # apply momentum, use the old value of deltaW to compute the new one.

                for l in range(1, model.layerCount):
                    model.layers[l].weights = (1 - 2 * lmbda) * model.layers[l].weights - deltaW[l]
                    model.layers[l].bias = model.layers[l].bias - eta * deltaB[l]

            for l in range(model.layerCount):
                gradAvg += np.linalg.norm(deltaW[l]/len(minibatches), 2)  # avg delta for each minibatch considered

            gradAvg /= model.layerCount # so grad avg is the avg delta that each minibatch caused to each weight

            trainMSE.append(self.meanerror(model, trainPatternList))
            trainAccuracy.append(self.performance(model, trainPatternList))

            if not (testPatternList is None):
                testMSE.append(self.meanerror(model, testPatternList))
                testAccuracy.append(self.performance(model, testPatternList))

            if not suppressPrint and not (testPatternList is None):
                print("Epoch {}/{}: Train MSE: {}, Train accuracy: {}; Test MSE: {}, Test accuracy: {}"
                      .format(epoch+1, self._maxEpochs, trainMSE[-1], trainAccuracy[-1], testMSE[-1], testAccuracy[-1]))
            elif not suppressPrint:
                print("Epoch {}/{}: Train MSE: {}, Train accuracy: {}"
                      .format(epoch + 1, self._maxEpochs, trainMSE[-1], trainAccuracy[-1]))

            epoch += 1

        end = time.time()

        if not suppressPrint:
            print("##### Training ended at: {} #####".format(time.strftime("%c")))
            print("##### Took {0:.2f} seconds. #####".format(end-start))

        return trainMSE, trainAccuracy, testMSE, testAccuracy