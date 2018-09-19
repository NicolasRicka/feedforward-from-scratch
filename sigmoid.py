""" Define the activation function sigmoid. the argument deriv is set to True if one wants to compute the derivative instead """

import numpy as np

def sigmoid(x, deriv = False):
    if (deriv == True):
        return (x*(1-x))
    
    return 1/(1 + np.exp(-x))