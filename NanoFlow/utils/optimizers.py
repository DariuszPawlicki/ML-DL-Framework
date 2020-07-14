import numpy as np
from abc import ABC, abstractmethod


class Optimizer(ABC):

    def __init__(self, beta):
        self.beta = beta
        self.momentum_W = []
        self.momentum_b = None

    @abstractmethod
    def update_weights(self, weights, biases, dW, db, learning_rate, layers):
        pass


class SGD(Optimizer):
    def __init__(self, beta = 0.9):
        super(SGD, self).__init__(beta)

    def update_weights(self, weights, biases, dW, db, learning_rate, layers):
        if self.beta == 0:
            for layer_id, layer_weights in enumerate(weights):
                if layers[layer_id].trainable == True:
                    layer_weights -= learning_rate * dW[layer_id]
                    weights[layer_id] = layer_weights

            return weights, biases - learning_rate * db

        if self.momentum_W == [] and self.momentum_b == None:
            for weights_matrix in weights:
                self.momentum_W.append(np.zeros((weights_matrix.shape[0], 1)))

            self.momentum_b = np.zeros(biases.shape)

        self.momentum_W = np.array(self.momentum_W)
        self.momentum_b = np.array(self.momentum_b)

        self.momentum_W += self.momentum_W * self.beta + \
                          (1 - self.beta) * learning_rate * dW

        self.momentum_b += self.momentum_b * self.beta * \
                          (1 - self.beta) * learning_rate * db

        return weights - self.momentum_W, biases - self.momentum_b


class RMSprop(Optimizer):
    pass