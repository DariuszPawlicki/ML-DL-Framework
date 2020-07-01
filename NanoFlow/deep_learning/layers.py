from abc import ABC, abstractmethod
from utils.activations import pick_activation
from utils.cost_functions import compute_cost


class Layer(ABC):
    def __init__(self, output_shape: int, activation: str,
                 regularization: str, reg_strength: float):

        self.input_shape = None
        self.output_shape = output_shape
        self.activation = activation
        self.__name__ = type(self).__name__
        self.trainable = True
        self.regularization = regularization
        self.reg_strength = reg_strength

    @abstractmethod
    def activate(self, X):
        pass

class InputLayer(Layer):
    def __init__(self, input_shape, output_shape, activation,
                 reg_strength = 0.01, regularization = ""):

        super(InputLayer, self).__init__(output_shape= output_shape, activation = activation,
                                         reg_strength = reg_strength, regularization = regularization)

        self.input_shape = input_shape

    def activate(self, X, derivative = False):
        return pick_activation(activation = self.activation)(X, derivative)


class DenseLayer(Layer):
    def __init__(self, output_shape, activation, reg_strength = 0.01, regularization =""):

        super(DenseLayer, self).__init__(output_shape= output_shape, activation = activation,
                                         reg_strength = reg_strength, regularization = regularization)

    def activate(self, X, derivative = False):
        return pick_activation(activation = self.activation)(X, derivative)


class OutputLayer(Layer):
    def __init__(self, output_shape, activation, cost_function: str,
                 reg_strength = 0.01, regularization = ""):

        super(OutputLayer, self).__init__(output_shape = output_shape, activation = activation,
                                          reg_strength = reg_strength, regularization = regularization)

        self.cost_function = cost_function


    def activate(self, X):
        return pick_activation(activation = self.activation)(X)


    def cost(self, target_labels, predictions, derivative = False):
        if derivative == True:
            return compute_cost(target_labels, predictions, derivative = True,
                                cost_function = self.cost_function)

        return compute_cost(target_labels, predictions,
                            cost_function = self.cost_function)


class BatchNormalization(Layer):
    def __init__(self, output_shape):
        super(BatchNormalization, self).__init__(output_shape= output_shape, activation ="",
                                                 reg_strength = 0, regularization = "")
        self.activation = "normalization"
        self.input_shape = self.output_shape
        self.trainable = False

    def activate(self, X):
        return pick_activation(activation = self.activation)(X)


class Dropout(Layer):
    def __init__(self, output_shape, rate: float):
        super(Dropout, self).__init__(output_shape= output_shape, activation ="",
                                      reg_strength = 0, regularization = "")
        self.input_shape = self.output_shape
        self.trainable = False

        try:
            assert 0 <= rate <= 1
            self.drop_rate = rate

        except(AssertionError):
            print("Dropout rate must be in [0,1] range.")


    def activate(self, X):
        from numpy import random

        drop_mask = random.choice([0, 1], size = (X.shape[0], X.shape[1]),
                                  p = [self.rate, 1 - self.rate])

        return X * drop_mask