import numpy as np
from layers import *


class NeuralNet:

    def __init__(self):
        self.layers = []
        self.weights = None
        self.biases = None

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def relu(self, X, derivative = False):
        if derivative == True:
            if X <= 0:
                return 0
            else:
                return 1

        return np.maximum(0, X)

    def sigmoid(self, X, derivative = False):
        if derivative == True:
            return self.sigmoid(X) * (1 - self.sigmoid(X))

        return 1 / (1 + np.exp(-X))

    def softmax(self, X):
        return np.exp(X) / np.sum(np.exp(X))

    def softmax_categorical_crossentropy(self, y_labels, output, derivative = False):
        if derivative == True: # Derivative of softmax crossentropy function with respect to softmax input "X"
            return output - y_labels

        return -np.sum(y_labels * np.log(output))

    def weights_init(self):
        weights = []

        for i in range(len(self.layers)):
            weights.append(np.random.randn(self.layers[i].input_shape, self.layers[i].output))

        self.weights = weights
        self.biases = np.ones((len(self.layers), 1))

    def forward_propagation(self, X):
        if X.shape[0] != 1:  # Reshaping X to row vector [1, n]
            X = X.T

        gradient_tape = []

        z = X

        gradient_tape.append("X")

        z_cache = []
        a_cache = []

        for i in range(len(self.layers)):
            layer_gradient = []

            a = self.relu(np.dot(z, self.weights[i]) + self.biases)

            layer_gradient.append(f"Z{i + 1}")

            if i == len(self.layers) - 1:
                layer_gradient.append("Y" + " " + self.layers[i].activation)
            else:
                layer_gradient.append(f"A{i + 1}" + " " + self.layers[i].activation)

            gradient_tape.append(layer_gradient)

            a_cache.append(a)
            z_cache.append(z)

            z = a

        return a_cache, z_cache, gradient_tape

    def cost(self, y_labels, output, derivative = False):
        if derivative == True:
            return self.softmax_categorical_crossentropy(y_labels, output, derivative = True)
        return self.softmax_categorical_crossentropy(y_labels, output)

    def back_propagation(self, y_labels, a_cache, z_cache, gradient_tape):
        output = a_cache[len(a_cache) - 1] # Value of activation in the last layer is the network output
        output_error = self.cost(y_labels, output, derivative = True)

        delta = []

        for i in range(len(self.layers), 0, -1):
            i -= 1

            layer_gradient = gradient_tape[i]

            


            

    def train(self, X, y_labels = None, epochs = 10, learning_rate = 0.01):
        for i in range(epochs):
            if i == 0:
                self.weights_init()

            a_cache, z_cache, gradient_tape = self.forward_propagation(X)

            cost = self.cost(y_labels, a_cache)

            gradient = self.back_propagation(y_labels, a_cache, z_cache, gradient_tape)


if __name__ == '__main__':
    net = NeuralNet()
    net.add_layer(DenseLayer(4, 2, activation = "relu"))
    net.add_layer(DenseLayer(2, 4, activation = "relu"))
    net.add_layer(DenseLayer(4, 1, activation = "relu"))
    X = np.random.randn(4, 1)

    net.weights_init()

    a_cache, z_cache, gradient_tape = net.forward_propagation(X)

    Y = [1, 0, 0]
    Y_pred = [0.99, 0.2, 0.1]

    print(gradient_tape)