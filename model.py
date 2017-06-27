import numpy as np

from layer import InputLayer


class Model:
    def __init__(self, inputSize):
        self._inputSize = inputSize
        self._outputSize = inputSize
        self._layers = [InputLayer(inputSize, inputSize)]  # Input layer!
        self._layerCount = 1

    def output(self, x):
        """
        Computes the output of the whole model.

        :param x: Input for the model
        :type x: np.array
        :return: Output of the model
        :rtype: np.array

        """
        assert(np.shape(x) == (self._inputSize, 1))

        for l in self._layers:
            x = l.output(x)

        return x

    @property
    def layers(self):
        """
        (Read-only) Property getter. Returns the layers list

        :return: The layers
        :rtype: List of Layer

        """
        return self._layers

    @property
    def layerCount(self):
        """
        (Read-only) Property getter. Returns the number of layers

        :return: The number of layers
        :rtype: int

        """
        return self._layerCount

    def addLayer(self, layer):
        """
        Adds a layer to the model

        :param layer: The layer to add
        :type layer: Layer

        """
        assert(layer.inputSize == self._outputSize)

        self._layers.append(layer)
        self._outputSize = layer.outputSize
        self._layerCount += 1

    def getLayer(self, i):
        """
        Returns the i-th (0 based) layer of the model.

        :param i: The index of the layer
        :type i: int
        :return: The chosen layer
        :rtype: Layer

        """
        assert(0 <= i < self._layerCount)

        return self._layers[i]

