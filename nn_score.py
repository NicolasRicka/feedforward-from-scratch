import numpy as np

def nn_score(pred, t):
    m = len(pred)
    correct = 0
    for i in range(m):
        if pred[i].argmax() == t[i].argmax():
            correct += 1
    return correct / m