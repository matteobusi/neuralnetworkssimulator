import abc
import numpy as np
import math

class CostFunction (object):
    """
    (Abstract) class representing possible cost functions.
    """
    __metaclass__ = abc.ABCMeta
    pass


class SECost (CostFunction):
    """
    Class representing the standard SE cost function
    """

    def cost(self, o, d):
        """
        Returns the cost of the given pair.
        """
        return 0.5 * np.linalg.norm(o-d, 2)**2 # multiplied by 1/2 so to get e more "elegant" gradient function

    def gradient(self, o, d):
        """
        Returns the gradient of the cost function over the given pair.
        """
        return o - d
