import matplotlib.pyplot as plt
import numpy as np

from sigmoid import sigmoid
from neural_network import neural_network
from nn_train import nn_train
from nn_feedforward import nn_feedforward


''' Let's try mnist classification'''

## loading Data
import tensorflow as tf
mnist = tf.keras.datasets.mnist
import pandas as pd

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

## Preprocessing

t_train = np.array(pd.get_dummies(y_train))
t_test = np.array(pd.get_dummies(y_test))

z_train = x_train.reshape(60000,28*28, order = 'F')
z_test = x_test.reshape(10000,28*28, order = 'F')
number = z_train[10].reshape(28,28)
plt.pcolor(number, cmap = 'gray')
plt.colorbar()
plt.show()


## Training classifier

nn = neural_network([784,56,28,10])

nnpluslog = nn_train(nn,z_train,t_train,1000,epsilon = 0.1, error = 25,X_test = x_test, y_test = t_test)

plt.plot(nnpluslog[-1])

nn = nnpluslog[0]

predictions = nn_feedforward(nn,z_test)


for i in range(5):
    number = z_test[i].reshape(28,28)
    plt.pcolor(number, cmap = 'gray')
    plt.colorbar()
    plt.title('This is predicted to be a ' + str(predictions[i].argmax()))
    plt.show()


from nn_score import nn_score

score = nn_score(predictions, t_test)
print(score)