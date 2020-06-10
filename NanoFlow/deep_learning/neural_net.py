from deep_learning.layers.layers import *
from utils.data_processing import *
from utils.metrics import *
from decorators import add_second_dim
from numpy import sqrt, random, ones, dot, zeros, sum, array, argmax


class NeuralNet:

    def __init__(self):
        self.layers = []
        self.weights = None
        self.biases = None
        self.cost = None
        self.metrics = None
        self.trainable_layers = None


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

            self.trainable_layers = sum([1 for layer in self.layers if layer.trainable])
            self.weights_init(param_init = param_init)
            self.metrics = pick_metrics_method(metrics)


    def weights_init(self, param_init: str):
        weights = []

        for cur_layer_index, current_layer in enumerate(self.layers):

            if current_layer.trainable == True:

                if param_init == "xavier":
                    method = sqrt(1 / current_layer.input_shape)
                elif param_init == "he_normal":
                    method = sqrt(2 / (self.layers[cur_layer_index - 1].input_shape +
                                       current_layer.input_shape))
                else:
                    method = 1

                weights.append(random.randn(current_layer.input_shape, current_layer.output) * method)

        self.weights = array(weights)

        self.biases = ones((self.trainable_layers, 1))


    def forward_propagation(self, X):
        if X.shape[1] != self.layers[0].input_shape:  # Reshaping X [x, n] where n are features of data
            X = X.T

        z = X

        non_trainable_activated = 0

        z_cache = []
        a_cache = []

        a_cache.append(X)

        for cur_layer_index, current_layer in enumerate(self.layers):
            cur_layer_index -= non_trainable_activated

            if current_layer.trainable:
                a = current_layer.activate(dot(z, self.weights[cur_layer_index]) +
                                           self.biases[cur_layer_index])
            else:
                a = current_layer.activate(z)
                non_trainable_activated += 1

            a_cache.append(a)
            z_cache.append(z)

            z = a

        return a_cache, z_cache


    def back_propagation(self, y_labels, a_cache, z_cache):
        output_layer = self.layers[-1]
        output_error = output_layer.cost(y_labels, a_cache[-1],
                                         derivative = True)

        delta_W = []
        delta_b = zeros((self.biases.shape[0], 1))

        delta_b[-1] += sum(output_error)

        for current_layer in self.layers:
            if current_layer.trainable == True:
                delta_W.append(zeros((current_layer.input_shape,
                                      current_layer.output)))

        for cur_layer_index, current_layer in reversed(list(enumerate(self.layers))): # Loop for deriving cost function
                                                                                      # with respect to weights[current_layer]

            if current_layer.trainable == False or \
                    current_layer.__name__ == OutputLayer.__name__:

                continue

            layer_gradient = output_error


            for derived_index, derived_layer in reversed(list(enumerate(self.layers))):
                if derived_layer.trainable == False:
                    continue

                if derived_index == cur_layer_index:
                    delta_b[cur_layer_index] += sum(layer_gradient)

                    layer_gradient = dot(layer_gradient.T, a_cache[cur_layer_index])

                    delta_W[cur_layer_index] += layer_gradient.T

                    break

                else:
                    layer_gradient = dot(layer_gradient, self.weights[derived_index].T)
                    layer_gradient *= current_layer.activate(z_cache[derived_index], derivative = True)

        delta_W = array(delta_W)

        return delta_W, delta_b


    @to_numpy_array
    @add_second_dim
    def train(self, X, y_labels, epochs = 10000,
              learning_rate = 0.01, batch_size = 100,
              verbose = True, patience = 10):

        previous_cost = 0
        patience_counter = patience

        for epoch in range(epochs):

            a_cache, z_cache = self.forward_propagation(X)

            dW, db = self.back_propagation(y_labels, a_cache, z_cache)

            dW *= learning_rate / len(X)
            db /= len(X)

            for layer, grad in enumerate(dW):
                if self.layers[layer].trainable == True:
                    self.weights[layer] -= grad

            self.biases -= db

            if verbose == True:
                cost = self.layers[-1].cost(y_labels, a_cache[-1])

                if cost >= previous_cost:
                    patience_counter -= 1

                    if patience_counter == 0:
                        print("Gradient descent is on plateau/diverging,"
                              "\ntry different learning rate or "
                              "feature engineering.\n")

                        print("Cost: ", cost)

                        break

                else:
                    patience_counter = patience

                previous_cost = cost

                print("Cost in epoch {} :".format(epoch + 1), cost)


    def predict(self, data):
        a_cache, _ = self.forward_propagation(data)
        return argmax(a_cache[-1], axis = 1)


    def evaluate(self, y_labels, predictions):
        print("\nAccuracy: ",
              self.metrics(y_labels, predictions))


if __name__ == '__main__':
    net = NeuralNet()

    net.add_layer(InputLayer(2, 1000, activation = "relu"))
    net.add_layer(Dropout(1000, rate = 0.2))
    net.add_layer(DenseLayer(500, activation = "relu"))
    net.add_layer(DenseLayer(900, activation = "relu"))
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