# feedforward-from-scratch

This repository contains an implementation of a feedforward neural network in Python from scratch. The only library used in this implementation is Numpy.

**Structure of the feedforward neural network**

* **neuralnetwork.py** - creation of a neural network (sequence of matrices) with architecture prescribed by the user.
* **nn_feedforward.py** - feedforward function.
* **nn_score.py** - computes a score function between output and prediction.
* **nn_train.py** - training function for a nn. The optimizer implemented here is a simple gradient descent without regularization.
* **sigmoid.py** - activation function, here the sigmoid.


**Applications**
Additional libraries required for the applications: MathPlotLib and TensorFlow.

* **fashion-mnist.py** - Uses the neural network described above to classify fashion products. The dataset considered here is the fashion-MNIST dataset.
* **mnist.py** - Uses the neural network described above to classify handwritten digits. The dataset considered here is the MNIST dataset, imported from Tensorflow for simplicity.
