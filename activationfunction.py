import numpy as np

class ActivationFunction:
    def __init__(self, function, derivative, performance):
        """
        Initializes the activation function

        :param function: The function
        :type function: float -> float
        :param derivative: The derivative
        :type derivative: float -> float
        :param performance: The performance evaluation function. Used only for output layers
        :type performance: float -> float
        """
        self._function = np.vectorize(function)
        self._derivative = np.vectorize(derivative)
        self._performance = performance

    @property
    def function(self):
        """
        Returns the activation function

        :return: The activation function
        :rtype: [float] -> [float]
        """
        return self._function

    @property
    def derivative(self):
        """
        Returns the derivative of the activation function

        :return: The function representing the derivative
        :rtype: [float] -> [float]
        """
        return self._derivative

    @property
    def performance(self):
        """
        Returns the performance evaluation function

        :return: The function representing evaluation function
        :rtype: float -> float
        """
        return self._performance
