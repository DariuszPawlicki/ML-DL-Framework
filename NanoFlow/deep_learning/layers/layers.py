from abc import ABC, abstractmethod
from utils.activations import pick_activation
from utils.cost_functions import compute_cost


class Layer(ABC):
    def __init__(self, output: int, activation: str):
        self.input_shape = None
        self.output = output
        self.activation = activation
        self.__name__ = type(self).__name__


    @abstractmethod
    def activate(self, X):
        pass


class InputLayer(Layer):
    def __init__(self, input_shape, output, activation):
        super(InputLayer, self).__init__(output, activation)
        self.input_shape = input_shape
        self.activation = activation

    def activate(self, X, derivative = False):
        return pick_activation(activation = self.activation)(X, derivative)


class DenseLayer(Layer):
    def __init__(self, output, activation):
        super(DenseLayer, self).__init__(output, activation)
        self.activation = activation

    def activate(self, X, derivative = False):
        return pick_activation(activation = self.activation)(X, derivative)


class OutputLayer(Layer):
    def __init__(self, output, activation, cost_function: str):
        super(OutputLayer, self).__init__(output, activation)
        self.activation = activation
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
        self.input_shape = output

    def activate(self, X):
        return pick_activation(activation = self.activation)(X)