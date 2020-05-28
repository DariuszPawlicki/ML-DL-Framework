class Layer:
    def __init__(self, input_shape, output, activation):
        self.input_shape = input_shape
        self.output = output
        self.activation = activation

class DenseLayer(Layer):
    def __init__(self, input_shape, output, activation):
        super(DenseLayer, self).__init__(input_shape, output, activation)