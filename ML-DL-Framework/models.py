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
        return np.exp(X) / np.sum(np.exp(X), axis = 1, keepdims = True)

    def categorical_crossentropy(self, y_labels, output, derivative = False):
        if derivative == True: # Derivative of softmax crossentropy function with respect to softmax input "X"
            return output - y_labels

        return -np.sum(y_labels * np.log(output))

    def weights_init(self):
        weights = []

        for i in range(len(self.layers)):
            xavier = np.sqrt(1/self.layers[i].input_shape)
            weights.append(np.random.randn(self.layers[i].input_shape, self.layers[i].output) * xavier)

        self.weights = np.array(weights)
        self.biases = np.ones((len(self.layers), 1))

    def forward_propagation(self, X):
        if X.shape[1] != self.layers[0].input_shape:  # Reshaping X to row vector [1, n]
            X = X.T

        z = X

        z_cache = []
        a_cache = []

        a_cache.append(X)

        for i in range(len(self.layers)):

            if i == len(self.layers) - 1:
                a = self.softmax(np.dot(z, self.weights[i]) + self.biases[i])
            else:
                a = self.relu(np.dot(z, self.weights[i]) + self.biases[i])

            a_cache.append(a)
            z_cache.append(z)

            z = a

        return a_cache, z_cache

    def cost(self, y_labels, output, derivative = False):
        if derivative == True:
            return self.categorical_crossentropy(y_labels, output, derivative = True)
        return self.categorical_crossentropy(y_labels, output)

    def back_propagation(self, y_labels, a_cache, z_cache):
        output = a_cache[-1]
        output_error = self.cost(y_labels, output, derivative = True)

        delta = []

        for layer in range(len(self.layers)):
            delta.append(np.zeros((self.layers[layer].input_shape, self.layers[layer].output)))

        for current_layer in range(len(self.layers) - 1, -1, -1): # Loop for deriving cost function
                                                                  # with respect to weights[current_layer]

            layer_gradient = output_error

            derived_layer = len(self.layers) - 1

            while derived_layer >= current_layer:
                if derived_layer == current_layer:
                    layer_gradient = np.dot(layer_gradient.T, a_cache[current_layer])
                    break
                else:
                    layer_gradient = np.dot(layer_gradient, self.weights[derived_layer].T)
                    layer_gradient *= self.relu(z_cache[derived_layer], derivative = True)

                    derived_layer -= 1

            delta[current_layer] += layer_gradient.T

        return np.array(delta)


    def train(self, X, y_labels = None, epochs = 10000, learning_rate = 0.001):
        for i in range(epochs):
            if i == 0:
                self.weights_init()

            a_cache, z_cache= self.forward_propagation(X)

            print(self.cost(y_labels, a_cache[-1]))

            gradient = self.back_propagation(y_labels, a_cache, z_cache)

            gradient *= learning_rate / len(X)

            for i, grad in enumerate(gradient):
                self.weights[i] -= grad


if __name__ == '__main__':
    net = NeuralNet()

    net.add_layer(DenseLayer(4, 2, activation = "relu"))
    net.add_layer(DenseLayer(2, 2, activation = "relu"))
    net.add_layer(DenseLayer(2, 3, activation = "softmax"))

    from sklearn.datasets import load_iris

    data, target = load_iris(True)

    one_hot = np.zeros((target.shape[0], 3))

    for index, label in enumerate(one_hot):
        true_class= target[index]
        label[true_class] = 1

    net.train(data, one_hot)