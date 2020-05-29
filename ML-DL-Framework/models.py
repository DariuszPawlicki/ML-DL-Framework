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
            X[X > 0] = 1
            X[X <= 0] = 0

            return X

        return np.maximum(0, X)

    def sigmoid(self, X, derivative = False):
        if derivative == True:
            return self.sigmoid(X) * (1 - self.sigmoid(X))

        return 1 / (1 + np.exp(-X))

    def softmax(self, X):
        return np.exp(X) / np.sum(np.exp(X))

    def categorical_crossentropy(self, y_labels, output, derivative = False):
        if derivative == True: # Derivative of softmax crossentropy function with respect to softmax input "X"
            return output - y_labels

        return -np.sum(y_labels * np.log(output))

    def weights_init(self):
        weights = []

        for i in range(len(self.layers) - 1):
            weights.append(np.random.randn(self.layers[i].input_shape, self.layers[i].output))

        self.weights = np.array(weights)
        self.biases = np.ones((len(self.layers), 1))

    def forward_propagation(self, X):
        if X.shape[0] != 1:  # Reshaping X to row vector [1, n]
            X = X.T

        gradient_tape = []

        z = X

        z_cache = []
        a_cache = []

        a_cache.append(X)

        for i in range(len(self.layers)):
            layer_gradient = []

            if i == len(self.layers) - 1:
                a = self.softmax(z)
            else:
                a = self.relu(np.dot(z, self.weights[i]) + self.biases[i])
                layer_gradient.append(f"dot_product{i + 1}")

            layer_gradient.append(self.layers[i].activation)

            gradient_tape.append(list(reversed(layer_gradient)))

            a_cache.append(a)
            z_cache.append(z)

            z = a

        return a_cache, z_cache, gradient_tape

    def cost(self, y_labels, output, derivative = False):
        if derivative == True:
            return self.categorical_crossentropy(y_labels, output, derivative = True)
        return self.categorical_crossentropy(y_labels, output)

    def back_propagation(self, y_labels, a_cache, z_cache, gradient_tape):
        output = a_cache[-1] # Value of activation in the last layer
        output_error = self.cost(y_labels, output, derivative = True)

        delta = []

        for current_layer in range(len(self.layers) - 2, -1, -1): # Indexing from 0 plus last layer is output

            layer_gradient = output_error

            derived_layer = len(self.layers) - 2

            while derived_layer >= current_layer:
                if derived_layer == current_layer:
                    layer_gradient = np.dot(layer_gradient, a_cache[current_layer - 1].T)
                    break
                else:
                    layer_gradient = np.dot(layer_gradient, self.weights[current_layer + 1].T)
                    layer_gradient *= self.relu(z_cache[derived_layer], derivative = True)

                    derived_layer -= 1

            delta.append(layer_gradient)

        return delta


    def train(self, X, y_labels = None, epochs = 10, learning_rate = 0.01):
        for i in range(epochs):
            if i == 0:
                self.weights_init()

            a_cache, z_cache, gradient_tape = self.forward_propagation(X)

            cost = self.cost(y_labels, a_cache[-1])

            gradient = self.back_propagation(y_labels, a_cache, z_cache, gradient_tape)


if __name__ == '__main__':
    net = NeuralNet()

    net.add_layer(DenseLayer(4, 2, activation = "relu"))
    net.add_layer(DenseLayer(2, 4, activation = "relu"))
    net.add_layer(DenseLayer(4, 4, activation = "softmax"))

    X = np.random.randn(4, 1)

    net.weights_init()

    a_cache, z_cache, gradient_tape = net.forward_propagation(X)

    Y = np.array([1, 0, 0, 0])

    #print(a_cache[1])
    #print(net.softmax_categorical_crossentropy(Y, a_cache[1]))

    #print(net.softmax_categorical_crossentropy(Y, Y_pred, True))

    print(net.back_propagation(Y, a_cache, z_cache, gradient_tape))
    #print(net.weights[2].shape)