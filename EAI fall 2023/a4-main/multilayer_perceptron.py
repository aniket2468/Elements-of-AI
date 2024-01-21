# multilayer_perceptron.py: Machine learning implementation of a Multilayer Perceptron classifier from scratch.
#
# Submitted by: [Aniket Sharma - Shreya Parab - Vaishnavi Rai] -- [anikshar-shparab-vairai]
#
# Based on skeleton code by CSCI-B 551 Fall 2023 Course Staff

import numpy as np
from utils import identity, sigmoid, tanh, relu, softmax, cross_entropy, one_hot_encoding

class MultilayerPerceptron:
    def __init__(self, n_hidden=16, hidden_activation='sigmoid', n_iterations=1000, learning_rate=0.01):
        activation_functions = {'identity': identity, 'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu}

        if not isinstance(n_hidden, int) or hidden_activation not in activation_functions or \
                not isinstance(n_iterations, int) or not isinstance(learning_rate, float):
            raise ValueError('The provided class parameter arguments are not recognized.')

        self.n_hidden = n_hidden
        self.hidden_activation = activation_functions[hidden_activation]
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self._output_activation = softmax
        self._loss_function = cross_entropy
        self._loss_history = []
        self._X = None
        self._y = None
        self._h_weights = None
        self._h_bias = None
        self._o_weights = None
        self._o_bias = None

    def _initialize(self, X, y):
        self._X = X
        self._y = one_hot_encoding(y)

        np.random.seed(42)

        # Initialize weights and biases
        self._h_weights = np.random.rand(X.shape[1], self.n_hidden)
        self._h_bias = np.zeros((1, self.n_hidden))
        self._o_weights = np.random.rand(self.n_hidden, self._y.shape[1])
        self._o_bias = np.zeros((1, self._y.shape[1]))

    def fit(self, X, y):
        self._initialize(X, y)

        for iteration in range(self.n_iterations):
            # Forward propagation
            h_input = np.dot(self._X, self._h_weights) + self._h_bias
            h_output = self.hidden_activation(h_input)
            o_input = np.dot(h_output, self._o_weights) + self._o_bias
            o_output = self._output_activation(o_input)

            # Backward propagation
            o_error = self._y - o_output
            o_delta = o_error * self._output_activation(o_input, derivative=True)
            h_error = o_delta.dot(self._o_weights.T)
            h_delta = h_error * self.hidden_activation(h_input, derivative=True)

            # Update weights and biases
            self._o_weights += h_output.T.dot(o_delta) * self.learning_rate
            self._o_bias += np.sum(o_delta, axis=0, keepdims=True) * self.learning_rate
            self._h_weights += self._X.T.dot(h_delta) * self.learning_rate
            self._h_bias += np.sum(h_delta, axis=0, keepdims=True) * self.learning_rate

            # Compute and store loss every 20 iterations
            if iteration % 20 == 0:
                loss = self._loss_function(self._y, o_output)
                self._loss_history.append(loss)

    def predict(self, X):
        h_input = np.dot(X, self._h_weights) + self._h_bias
        h_output = self.hidden_activation(h_input)
        o_input = np.dot(h_output, self._o_weights) + self._o_bias
        o_output = self._output_activation(o_input)

        return np.argmax(o_output, axis=1)