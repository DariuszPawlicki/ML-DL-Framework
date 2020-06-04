from deep_learning.layers.layers import *
from utils.activations import *
from utils.cost_functions import *
from utils.data_processing import *
from utils.metrics import *


class NeuralNet:

    def __init__(self):
        self.layers = []
        self.weights = None
        self.biases = None
        self.cost = None
        self.metrics = None

    def add_layer(self, layer: Layer):
        if len(self.layers) == 0 and type(layer).__name__ != InputLayer.__name__:
            raise TypeError("First layer of the neural network"
                            " must be the InputLayer type.")

        self.layers.append(layer)

        if len(self.layers) > 1:
            self.layers[-1].input_shape = self.layers[-2].output


    def build_model(self, cost: str, metrics: str, param_init: str):
        self.weights_init(param_init = param_init)
        self.cost = cost
        self.metrics = pick_metrics_method(metrics)


    def weights_init(self, param_init: str):
        weights = []

        for i in range(len(self.layers)):

            if param_init == "xavier":
                method = np.sqrt(1 / self.layers[i].input_shape)
            elif param_init == "he_normal":
                method = np.sqrt(2 / (self.layers[i - 1].input_shape + self.layers[i].input_shape))
            else:
                method = 1

            weights.append(np.random.randn(self.layers[i].input_shape, self.layers[i].output) * method)

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


    def back_propagation(self, y_labels, a_cache, z_cache):
        output = a_cache[-1]
        output_error = compute_cost(y_labels = y_labels, output = output,
                                    cost_function = self.cost, derivative = True)

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

        delta_W = np.array(delta_W)

        return delta_W, delta_b


    def train(self, X, y_labels, epochs = 100,
              learning_rate = 0.01, batch_size = 100,
              evaluate = True):

        for i in range(epochs):

            a_cache, z_cache = self.forward_propagation(X)

            dW, db = self.back_propagation(y_labels, a_cache, z_cache)

            dW *= learning_rate / len(X)
            db /= len(X)

            for i, grad in enumerate(dW):
                self.weights[i] -= grad

            self.biases -= db

        if evaluate == True:
            score = self.metrics(y_labels, one_hot_encoder(np.argmax(a_cache[-1], axis = 1)))
            print("Accuracy on train data: ", score)


    def predict(self, data):
        a_cache, _ = self.forward_propagation(data)
        return add_dimension(np.argmax(a_cache[-1], axis = 1))


if __name__ == '__main__':
    net = NeuralNet()

    net.add_layer(InputLayer(2, 100, activation = "relu"))
    net.add_layer(DenseLayer(100, activation = "relu"))
    net.add_layer(DenseLayer(100, activation = "relu"))
    net.add_layer(DenseLayer(2, activation = "softmax"))

    from sklearn.datasets import make_moons

    X_moons, y_moons = make_moons(n_samples = 1000, noise = 0.1)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X_moons, y_moons, test_size = 0.2)

    shuffler = np.random.permutation(len(X_train))

    X_train = X_train[shuffler]
    y_train = y_train[shuffler]

    one_hot = one_hot_encoder(y_train)

    net.build_model("categorical_crossentropy", "accuracy", "xavier")

    net.train(X_train, one_hot)

    y_pred = net.predict(X_test)

    y_pred = add_dimension(np.squeeze(y_pred))
    y_test = add_dimension(y_test)

    print("Accuracy on test data: ", net.metrics(y_test, y_pred))