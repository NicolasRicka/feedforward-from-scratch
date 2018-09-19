''' This function creates a list of random matrices that correspond to the
shape specified for the neural network, and returns the shape if the parameter
return_shape is set to true.
'''

import numpy as np


def neural_network(shape, return_shape = True):
    
    #Initialization
    nn = []
    
    #Creating the matrices
    i  = shape[0]
    
    for j in shape[1:]:
        W = 2*np.random.random((j,i+1)) -1
        i = j
        nn.append(W)
        
    if return_shape == True:
        return [shape, nn]
    else:
        return nn