
import numpy as np
from sigmoid import sigmoid

def nn_feedforward(nn, X, all_layers = False):
    L = len(nn)
    a = [X]
        
    for j in range(L):
        z_j = sigmoid(np.dot(np.insert(a[-1],0,[1],1),(nn[j].T)))
        a.append(z_j)
    if all_layers == False:
        return a[-1]
    else:
        return a
