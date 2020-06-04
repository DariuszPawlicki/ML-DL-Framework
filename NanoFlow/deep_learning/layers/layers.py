class Layer:
    def __init__(self, output: int, activation: str):
        self.input_shape = None
        self.output = output
        self.activation = activation


class InputLayer(Layer):
    def __init__(self, input_shape, output, activation):
        super(InputLayer, self).__init__(output, activation)
        self.input_shape = input_shape


class DenseLayer(Layer):
    def __init__(self, output, activation):
        super(DenseLayer, self).__init__(output, activation)