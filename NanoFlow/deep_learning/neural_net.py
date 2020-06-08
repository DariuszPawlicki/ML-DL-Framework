from deep_learning.layers.layers import *
from utils.activations import *
from utils.cost_functions import *
from utils.data_processing import *
from utils.metrics import *
from decorators import add_second_dim


class NeuralNet:

    def __init__(self):
        self.layers = []
        self.weights = None
        self.biases = None
        self.cost = None
        self.metrics = None


    def add_layer(self, layer: Layer):
        if len(self.layers) == 0 and layer.__name__ != InputLayer.__name__:
            raise TypeError("First layer of model"
                            " must be an 'InputLayer' type.")

        self.layers.append(layer)

        if len(self.layers) > 1:
            self.layers[-1].input_shape = self.layers[-2].output


    def build_model(self, metrics: str, param_init: str):

            if self.layers[-1].__name__ != OutputLayer.__name__:
                raise TypeError("Last layer of model must be an"
                                " 'OutputLayer' type.")

            self.weights_init(param_init = param_init)
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
            current_layer = self.layers[i]

            if i == len(self.layers) - 1:
                a = current_layer.activate(np.dot(z, self.weights[i]) + self.biases[i])
            else:
                a = current_layer.activate(np.dot(z, self.weights[i]) + self.biases[i])

            a_cache.append(a)
            z_cache.append(z)

            z = a

        return a_cache, z_cache


    def back_propagation(self, y_labels, a_cache, z_cache):
        output_layer = self.layers[-1]
        output_error = output_layer.cost(y_labels, a_cache[-1],
                                         derivative = True)

        delta_W = []
        delta_b = np.zeros((self.biases.shape[0], 1))

        delta_b[-1] += np.sum(output_error)

        for layer in range(len(self.layers)):
            delta_W.append(np.zeros((self.layers[layer].input_shape, self.layers[layer].output)))

        for i in range(len(self.layers) - 1, -1, -1): # Loop for deriving cost function
                                                      # with respect to weights[current_layer]

            current_layer = self.layers[i]

            if current_layer.__name__ == BatchNormalization.__name__ or \
                            current_layer.__name__ == OutputLayer.__name__:
                continue

            layer_gradient = output_error

            derived_layer = len(self.layers) - 1

            while derived_layer >= i:
                if derived_layer == i:
                    delta_b[i] += np.sum(layer_gradient)

                    layer_gradient = np.dot(layer_gradient.T, a_cache[i])

                    delta_W[i] += layer_gradient.T

                    break
                else:
                    layer_gradient = np.dot(layer_gradient, self.weights[derived_layer].T)
                    layer_gradient *= current_layer.activate(z_cache[derived_layer], derivative = True)

                    derived_layer -= 1

        delta_W = np.array(delta_W)

        return delta_W, delta_b


    @to_numpy_array
    @add_second_dim
    def train(self, X, y_labels, epochs = 100,
              learning_rate = 0.01, batch_size = 100,
              verbose = True):

        for i in range(epochs):

            a_cache, z_cache = self.forward_propagation(X)

            dW, db = self.back_propagation(y_labels, a_cache, z_cache)

            dW *= learning_rate / len(X)
            db /= len(X)

            for layer, grad in enumerate(dW):
                self.weights[layer] -= grad

            self.biases -= db

            if verbose == True:
                cost = self.layers[-1].cost(y_labels, a_cache[-1])
                print("Cost in epoch {} :".format(i + 1), cost)


    def predict(self, data):
        a_cache, _ = self.forward_propagation(data)
        return np.argmax(a_cache[-1], axis = 1)


    def evaluate(self, y_labels, predictions):
        print("\nAccuracy: ",
              self.metrics(y_labels, predictions))


if __name__ == '__main__':
    net = NeuralNet()

    net.add_layer(InputLayer(2, 100, activation = "relu"))
    net.add_layer(BatchNormalization(100))
    net.add_layer(DenseLayer(100, activation = "relu"))
    net.add_layer(DenseLayer(100, activation = "relu"))
    net.add_layer(OutputLayer(2, activation = "softmax",
                              cost_function = "categorical_crossentropy"))

    from sklearn.datasets import make_moons

    X_moons, y_moons = make_moons(n_samples = 1000, noise = 0.1)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X_moons, y_moons, test_size = 0.2)

    one_hot = one_hot_encoder(y_train)

    net.build_model("accuracy", "xavier")

    net.train(X_train, one_hot)

    y_pred = net.predict(X_test)

    net.evaluate(y_test, y_pred)