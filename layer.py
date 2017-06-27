import numpy as np
import abc

class Layer (object):
    __metaclass__ = abc.ABCMeta
    pass

class InputLayer(Layer):
    def __init__(self, inputSize, outputSize):
        """
        Constructor for the input layer.

        :param inputSize: size of the input for each neuron of the layer
        :type inputSize: int
        :param outputSize: size of the output, i.e. number of neurons for this layer
        :type outputSize: int
        """
        assert (outputSize > 0)
        assert (inputSize > 0)

        self._inputSize = inputSize
        self._outputSize = outputSize

    @property
    def inputSize(self):
        """
        (Read-only) Property getter. Returns input size

        :return: The input size of the layer
        :rtype: int

        """
        return self._inputSize

    @property
    def outputSize(self):
        """
        (Read-only) Property getter. Returns output size

        :return: Output size of the layer
        :rtype: int

        """
        return self._outputSize

    def net(self, x):
        """
        Computes the net output of the layer, in the form of a vector.

        :param x: The input for the layer (e.g. the sample input or the previous layer output)
        :type x: np.array of (inputSize)
        :return: The net of the layer.
        :rtype: np.array of (outputSize)

        """
        assert (np.shape(x) == (self._inputSize, 1))
        return x  # row i is net(x) of the i-th neuron

    def output(self, x):
        """
        Computes the output of the layer, in the form of a vector.

        :param x: The input for the layer (e.g. the sample input or the previous layer output)
        :type x: np.array of (inputSize)
        :return: The output of the layer.
        :rtype: np.array of (outputSize)

        """
        return self.net(x)

class DenseLayer (Layer):
    def __init__(self, inputSize, outputSize, initialWeightMatrix, initialBias, layerActivation):
        """
        Constructor for the dense layer (i.e. fully connected layer).

        :param inputSize: size of the input for each neuron of the layer
        :type inputSize: int
        :param outputSize: size of the output, i.e. number of neurons for this layer
        :type outputSize: int
        :param initialWeightMatrix: Weight matrix.
                                    Entry (i, j) corresponds to the weight coming from j-th neuron of previous layer
                                    to the i-th neuron in the current layer.
        :type initialWeightMatrix: np.array of (outputSize x inputSize)
        :param initialBias: Bias vector. Entry i corresponds to the bias for the i-th neuron.
        :type initialBias: np.array of (outputSize)
        :param layerActivation: The activation function for the layer
        :type layerActivation: ActivationFunction

        """
        assert(outputSize > 0)
        assert(inputSize > 0)
        assert(not (layerActivation is None))
        assert(not (initialWeightMatrix is None))
        assert(np.shape(initialWeightMatrix) == (outputSize, inputSize))
        assert(np.shape(initialBias) == (outputSize, 1))

        self._weights = initialWeightMatrix
        self._bias = initialBias
        self._inputSize = inputSize
        self._outputSize = outputSize
        self._layerActivation = layerActivation

    @property
    def weights(self):
        """
        Property getter. Returns the weight matrix

        :return: The weight matrix
        :rtype: np.array of (outputSize x inputSize)

        """
        return self._weights

    @weights.setter
    def weights(self, w):
        """
        Property setter. Sets the weight matrix.

        :param w: The new weight matrix
        :type w: np.array of (outputSize x inputSize)

        """
        assert(np.shape(w) == np.shape(self._weights))
        self._weights = w

    @property
    def bias(self):
        """
        Property getter. Returns the bias vector

        :return: The bias vector
        :rtype:  np.array of (outputSize)

        """
        return self._bias

    @bias.setter
    def bias(self, b):
        """
        Property setter. Sets the bias vector.

        :param b: The new bias vector
        :type b: np.array of (outputSize)

        """
        assert (np.shape(b) == np.shape(self._bias))
        self._bias = b

    @property
    def inputSize(self):
        """
        (Read-only) Property getter. Returns input size

        :return: The input size of the layer
        :rtype: int

        """
        return self._inputSize

    @property
    def outputSize(self):
        """
        (Read-only) Property getter. Returns output size

        :return: Output size of the layer
        :rtype: int

        """
        return self._outputSize

    @property
    def layerActivation(self):
        """
        (Read-only) Property getter. Returns activation function of the layer

        :return: Activation function of the layer
        :rtype: ActivationFunction

        """
        return self._layerActivation

    def net(self, x):
        """
        Computes the net output of the layer, in the form of a vector.

        :param x: The input for the layer (e.g. the sample input or the previous layer output)
        :type x: np.array of (inputSize)
        :return: The net of the layer.
        :rtype: np.array of (outputSize)

        """
        assert(np.shape(x) == (self._inputSize, 1))
        return np.dot(self._weights, x) + self._bias  # row i is net(x) of the i-th neuron

    def output(self, x):
        """
        Computes the output of the layer, in the form of a vector.

        :param x: The input for the layer (e.g. the sample input or the previous layer output)
        :type x: np.array of (inputSize)
        :return: The output of the layer.
        :rtype: np.array of (outputSize)

        """
        return self._layerActivation.function(self.net(x))