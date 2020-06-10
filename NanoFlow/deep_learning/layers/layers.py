from abc import ABC, abstractmethod
from utils.activations import pick_activation
from utils.cost_functions import compute_cost


class Layer(ABC):
    def __init__(self, output: int, activation: str):
        self.input_shape = None
        self.output = output
        self.activation = activation
        self.__name__ = type(self).__name__
        self.trainable = True


    @abstractmethod
    def activate(self, X):
        pass


class InputLayer(Layer):
    def __init__(self, input_shape, output, activation):
        super(InputLayer, self).__init__(output, activation)
        self.input_shape = input_shape

    def activate(self, X, derivative = False):
        return pick_activation(activation = self.activation)(X, derivative)


class DenseLayer(Layer):
    def __init__(self, output, activation):
        super(DenseLayer, self).__init__(output, activation)

    def activate(self, X, derivative = False):
        return pick_activation(activation = self.activation)(X, derivative)


class OutputLayer(Layer):
    def __init__(self, output, activation, cost_function: str):
        super(OutputLayer, self).__init__(output, activation)
        self.cost_function = cost_function

    def activate(self, X):
        return pick_activation(activation = self.activation)(X)

    def cost(self, y_labels, predictions, derivative = False):
        if derivative == True:
            return compute_cost(y_labels, predictions, derivative = True,
                                cost_function = self.cost_function)

        return compute_cost(y_labels, predictions,
                            cost_function = self.cost_function)


class BatchNormalization(Layer):
    def __init__(self, output):
        super(BatchNormalization, self).__init__(output, "")
        self.activation = "normalization"
        self.input_shape = self.output
        self.trainable = False

    def activate(self, X):
        return pick_activation(activation = self.activation)(X)


class Dropout(Layer):
    def __init__(self, output, rate: float):
        super(Dropout, self).__init__(output, "")
        self.input_shape = self.output
        self.trainable = False

        try:
            assert 0 <= rate <= 1
            self.rate = rate

        except(ValueError):
            print("Dropout rate must be in [0,1] range.")


    def activate(self, X):
        from numpy import random

        drop_mask = random.choice([0, 1], size = (X.shape[0], X.shape[1]),
                             p = [self.rate, 1 - self.rate])

        return X * drop_mask