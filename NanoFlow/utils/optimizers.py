import numpy as np
from abc import ABC, abstractmethod


class Optimizer(ABC):

    def __init__(self, beta):
        self.beta = beta
        self.__momentum_W = None
        self.__momentum_b = None

    @abstractmethod
    def update_weights(self, weights, biases, dW, db, learning_rate):
        pass


class SGD(Optimizer):
    def __init__(self, beta = 0.9):
        super(SGD, self).__init__(beta)

    def update_weights(self, weights, biases, dW, db, learning_rate):
        if self.__momentum_W == None and self.__momentum_b == None:
            self.__momentum_W = np.zeros(weights.shape)
            self.__momentum_b = np.zeros(biases.shape)

        self.__momentum_W = self.__momentum_W * self.beta + \
                            (1 - self.beta) * learning_rate * dW

        self.__momentum_b = self.__momentum_b * self.beta * \
                            (1 - self.beta) * learning_rate * db

        return weights - self.__momentum_W, biases - self.__momentum_b


class RMSprop(Optimizer):
    pass