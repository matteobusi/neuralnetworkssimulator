import abc

import numpy as np


class Trainer(object):
    """
    This is a learner using the simple online back propagation (basic gradient descent).
    Given a list of samples, a number of epochs, a maximum error and a learning rate it trains the given model.

    No regularization is applied.
    The update strategy is the basic back propagation.
    The cost function is the MSE.

    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, maxEpochs, minAvgDesc, costFunction):
        """
        Builds the trainer
        """
        self._costFunction = costFunction
        self._minAvgDesc = minAvgDesc
        self._maxEpochs = maxEpochs

    def initModel(self, model):
        for l in model.layers[1:]:
            l.weights = 2* np.random.randn(l.outputSize, l.inputSize) / l.inputSize
            l.bias = np.zeros([l.outputSize, 1])  # No big deal initializing biases to 0. The asymmetry breaking is provided by weights

        # last layer must be initialized w/o the use of fan-in, o.w. deltas will be too close to 0
        outputLayer = model.layers[-1]
        outputLayer.weights = np.random.randn(outputLayer.outputSize, outputLayer.inputSize)

    def meanerror(self, model, patternList):
        """
        (Helper method)
        This method computes the mean error over the given list.

        :param model: The model on which we want to evaluate the error
        :type model: Model
        :param patternList: The pattern list
        :type patternList: A list of pairs (input, target)
        :return: The value of the total error
        :rtype: float
        """

        return sum(self._costFunction.cost(model.output(x), d) for x, d in patternList) / len(patternList)


    def performance(self, model, patternList):
        """
        (Helper method)
        This method computes the accuracy of the model over the pattern list
        :param model:
        :type model:
        :param patternList:
        :type patternList:
        :return:
        :rtype:

        """
        delta = model.layers[-1].layerActivation.performance
        return sum(delta(model.output(x), d) for x, d in patternList) / len(patternList)
        #return sum(np.linalg.norm(model.output(x) - d) for x,d in patternList) / len(patternList)


    def backpropagation(self, model, pattern):
        """
        Given the model and pattern returns deltas for weights and bias as lists for each layer

        :param model: The model to train
        :type model: Model
        :param pattern: The single pattern to train on
        :type pattern: Pair
        :return: deltaW and deltaB
        :rtype: Pair
        """
        # Step 0: extract some useful informations
        inputLayer = model.getLayer(0)
        outputLayer = model.getLayer(model.layerCount - 1)

        x, d = pattern
        outputs = []
        nets = []
        deltaB = [np.zeros(0)] + [np.zeros(l.bias.shape) for l in model.layers[1:]]
        deltaW = [np.zeros(0)] + [np.zeros(l.weights.shape) for l in model.layers[1:]]

        # Step 1: compute the output of the input layer
        outputs.append(inputLayer.output(x))
        nets.append(inputLayer.net(x))

        # Step 2: feed-forward
        for l in range(1, model.layerCount):
            nets.append(model.getLayer(l).net(outputs[l - 1]))
            outputs.append(model.getLayer(l).layerActivation.function(nets[l]))

        # Step 3: compute output's error (delta)
        delta = self._costFunction.gradient(outputs[-1], d) * outputLayer.layerActivation.derivative(nets[-1])
        deltaW[-1] = np.dot(delta, outputs[-2].T)
        deltaB[-1] = delta

        # Step 4: back-propagate the error
        for l in range(model.layerCount - 2, 0, -1):
            delta = np.dot(model.getLayer(l + 1).weights.T, delta) * model.getLayer(l).layerActivation.derivative(
                nets[l])
            deltaW[l] = np.dot(delta, outputs[l - 1].T)
            deltaB[l] = delta

        return deltaW, deltaB
