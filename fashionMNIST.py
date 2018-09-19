import matplotlib.pyplot as plt
import numpy as np
import random

#from sigmoid import sigmoid
from neural_network import neural_network
from nn_train import nn_train
from nn_feedforward import nn_feedforward


''' fashion-mnist classification
classification follows the following convention:
0 T-shirt/top
1 Trouser
2 Pullover
3 Dress
4 Coat
5 Sandal
6 Shirt
7 Sneaker
8 Bag
9 Ankle boot 
'''

int_to_fashion = dict({0 : 'T-shirt/top', 1 : 'Trouser', 2: 'Pullover',3: 'Dress',4: 'Coat',
                       5: 'Sandal',6: 'Shirt',7: 'Sneaker',8: 'Bag',9: 'Ankle boot' })



## loading Data
import tensorflow as tf
mnist = tf.keras.datasets.mnist
import pandas as pd

dftrain = pd.read_csv('fashion-mnist/fashion-mnist_train.csv')
dftest = pd.read_csv('fashion-mnist/fashion-mnist_test.csv')

x_train = dftrain.drop(['label'], axis = 1).values
y_train = dftrain['label'].values

x_test = dftest.drop(['label'], axis = 1).values
y_test = dftest['label'].values


x_train, x_test = x_train / 255.0, x_test / 255.0

## Preprocessing

t_train = np.array(pd.get_dummies(y_train))
t_test = np.array(pd.get_dummies(y_test))

print('data loaded...')

z_train = x_train.reshape(60000,28*28, order = 'F')
z_test = x_test.reshape(10000,28*28, order = 'F')
number = z_train[1].reshape(28,28)
plt.pcolor(number, cmap = 'gray')
plt.colorbar()
plt.show()


## Training classifier

nn = neural_network([784,256,256,10])

print('network created...')


nnpluslog = nn_train(nn,z_train,t_train,250,epsilon = 0.1, error = 10, X_test = x_test, y_test = t_test)

print('training done...')

plt.plot(nnpluslog[-1])

nn = nnpluslog[0]

predictions = nn_feedforward(nn,z_test)

print('here are some examples of classification done by the neural network')

for i in range(5):
    j = random.randint(0,10000)
    number = z_test[j].reshape(28,28)
    plt.pcolor(number, cmap = 'gray')
    plt.colorbar()
    plt.title('This is predicted to be a ' + int_to_fashion[int(predictions[j].argmax())])
    plt.show()


from nn_score import nn_score

score = nn_score(predictions, t_test)
print('final score',score)