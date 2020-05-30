import numpy as np
from layers import *
from activations import *


class NeuralNet:

    def __init__(self):
        self.layers = []
        self.weights = None
        self.biases = None

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def categorical_crossentropy(self, y_labels, output, derivative = False):
        if derivative == True: # Derivative of softmax crossentropy function with respect to weights
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
        if X.shape[1] != self.layers[0].input_shape:  # Reshaping X [x, n] where n are features of data
            X = X.T

        z = X

        z_cache = []
        a_cache = []

        a_cache.append(X)

        for i in range(len(self.layers)):

            if i == len(self.layers) - 1:
                a = softmax(np.dot(z, self.weights[i]) + self.biases[i])
            else:
                a = relu(np.dot(z, self.weights[i]) + self.biases[i])

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

        delta_W = []
        delta_b = np.zeros((self.biases.shape[0], 1))

        delta_b[-1] += np.sum(output_error)

        for layer in range(len(self.layers)):
            delta_W.append(np.zeros((self.layers[layer].input_shape, self.layers[layer].output)))

        for current_layer in range(len(self.layers) - 1, -1, -1): # Loop for deriving cost function
                                                                  # with respect to weights[current_layer]

            layer_gradient = output_error

            derived_layer = len(self.layers) - 1

            while derived_layer >= current_layer:
                if derived_layer == current_layer:
                    delta_b[current_layer] += np.sum(layer_gradient)

                    layer_gradient = np.dot(layer_gradient.T, a_cache[current_layer])

                    delta_W[current_layer] += layer_gradient.T
                    break
                else:
                    layer_gradient = np.dot(layer_gradient, self.weights[derived_layer].T)
                    layer_gradient *= relu(z_cache[derived_layer], derivative = True)

                    derived_layer -= 1

        return np.array(delta_W), delta_b


    def train(self, X, y_labels = None, epochs = 1000, learning_rate = 0.01):
        for i in range(epochs):
            if i == 0:
                self.weights_init()

            a_cache, z_cache= self.forward_propagation(X)

            dW, db = self.back_propagation(y_labels, a_cache, z_cache)

            dW *= learning_rate / len(X)
            db /= len(X)

            for i, grad in enumerate(dW):
                self.weights[i] -= grad

            self.biases -= db

if __name__ == '__main__':
    net = NeuralNet()

    net.add_layer(DenseLayer(2, 100, activation = "relu"))
    net.add_layer(DenseLayer(100, 100, activation = "relu"))
    net.add_layer(DenseLayer(100, 100, activation = "relu"))
    net.add_layer(DenseLayer(100, 2, activation = "softmax"))

    from sklearn.datasets import make_moons

    X_moons, y_moons = make_moons(n_samples = 1000, noise = 0.25)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X_moons, y_moons, test_size = 0.2)

    shuffler = np.random.permutation(len(X_train))

    X_train = X_train[shuffler]
    y_train = y_train[shuffler]

    one_hot = np.zeros((y_train.shape[0], 2))

    for index, label in enumerate(one_hot):
        true_class = y_train[index]
        label[true_class] = 1

    net.train(X_train, one_hot)

    y_pred = []

    for X in X_test:
        output, _ = net.forward_propagation(X.reshape(X.shape[0], 1))
        y_pred.append(np.argmax(output[-1]))

    y_pred = np.array(y_pred)

    print(((y_test == y_pred).sum() / len(y_pred)) * 100, "%")