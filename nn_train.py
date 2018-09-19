''' This is the train function, that takes a neural network and a data set
and train it to fit .
nn should have its shape ( nn = [shape,neural network]).'''

import numpy as np
from sigmoid import sigmoid
from nn_feedforward import nn_feedforward
from nn_score import nn_score

def nn_train(nn, X,y,iterations, epsilon = 1,error = 0, Lambda = 0, X_test = None, y_test = None):
    #Initialization of the output
    nn_out = nn[1]
    shape = nn[0]
    error_log = []
    errortest_log = []
    
    for i in range(iterations):
        
        #Forward propagation
        #print('Forward propagation')
        a = [X]
        
        for j in range(len(shape)-1):
            z_j = sigmoid(np.dot(np.insert(a[-1],0,[1],1),(nn_out[j].T)))
            if j != len(shape)-2:
                a.append(z_j)
            else:
                a.append(z_j)
        #print('forward propagation done, a:',a)
        #Gradient computation. Note that delta is the reversed list of errors.
        #print('Gradient computation')
        delta = [a[-1]-y]
        for j in range(len(shape)-2):
            #print('trying to compute delta_' + str(len(shape)-j-1))
            delta_j = (np.dot((delta[-1]* sigmoid(a[-j-1], deriv = True)),nn_out[-j-1][:,1:]))
            delta.append(delta_j)
            #print('delta_', len(shape) - j-1, '=', delta_j)
            
        #Update weights
        for j in range(len(nn_out)):
            correction = np.zeros(nn_out[-j-1].shape)
            if j != (len(nn_out)-1):
                for k  in range(a[-j-2].shape[0]):
                    #print(k)
                    correction += np.dot((delta[j][k]).reshape(-1,1),(np.insert(a[-j-2][k],0,[1],0).reshape(1,-1)))
                nn_out[-j-1] -= epsilon*1/(a[-j-2].shape[0])*correction 
            else:
                for k  in range(a[-j-2].shape[0]):
                    #print(k)
                    correction += np.dot((delta[j][k]).reshape(-1,1),(np.insert(X[k],0,[1],0)).reshape(1,-1))
                nn_out[-j-1] -= epsilon*1/(a[-j-2].shape[0])*correction 
        if (error != 0):
            if (i % error) == 0:
                error_val = np.mean(np.abs(delta[0]))
                try:
                    pred_test = nn_feedforward(nn[1], X_test)
                    error_test = nn_score(pred_test, y_test)
                    print('Error at step ' + str(i) + ' = ' + str(error_val)[0:6] + ', correct on test set = ' + str((error_test*100))[0:6] + '%')
                    error_log.append(error_val)
                    errortest_log. append(error_test)
                except:
                    print('Error at step ' + str(i) + ' = ' + str(error_val)[0:6])
                    error_log.append(error_val)
    return [nn_out, shape, error_log, errortest_log]