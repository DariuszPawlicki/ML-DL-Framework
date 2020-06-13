from utils.data_processing import *
from utils.metrics import *
from decorators import add_second_dim, to_numpy_array
from numpy import sqrt, random, dot, zeros, sum, array as np_array, argmax
from deep_learning.layers import *
from utils.regularizers import Regularizers


class NeuralNet:

    def __init__(self):
        self.layers = []
        self.weights = None
        self.biases = None
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
        biases = []

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
                biases.append(float(1))

            else:
                weights.append(0)
                biases.append(float(0))

        self.weights = np_array(weights)
        self.biases = np_array(biases)

    def forward_propagation(self, X):
        if X.shape[1] != self.layers[0].input_shape:  # Reshaping X [x, n] where n are features of data
            X = X.T

        prev_activations = X # Previous layer activation

        dot_product = None  # Dot product of previous layer
                           #  activations and current layer weights

        activations_cache = []
        dot_prod_cache = []

        activations_cache.append(X)

        for cur_layer_index, current_layer in enumerate(self.layers):

            if current_layer.trainable:
                dot_product = dot(prev_activations, self.weights[cur_layer_index]) + \
                              self.biases[cur_layer_index]

            prev_activations = current_layer.activate(dot_product)

            activations_cache.append(prev_activations)
            dot_prod_cache.append(dot_product)

        return activations_cache, dot_prod_cache


    def back_propagation(self, y_labels, activations_cache, dot_prod_cache):
        output_layer = self.layers[-1]

        cost_func_derivative = output_layer.cost(y_labels, activations_cache[-1],
                                                 derivative = True)
        """
        Cost function derivative, w.r.t to outputs.
        """
        output_weights_gradient = dot(activations_cache[-2].T,
                                      cost_func_derivative)
        """
        Cost function derivative, w.r.t to output layer weights.
        """

        delta_W = []
        delta_b = zeros((len(self.layers), 1))

        for current_layer in self.layers:
            if current_layer.trainable == True:
                delta_W.append(zeros((current_layer.input_shape,
                                      current_layer.output)))
            else:
                delta_W.append(0)

        delta_W[-1] += output_weights_gradient
        delta_b[-1] += sum(cost_func_derivative)

        for cur_layer_index, current_layer in reversed(list(enumerate(self.layers))):
            """
             Loop for deriving cost function
             with respect to 'current_layer' weights.
            """

            if current_layer.trainable == False or \
                    current_layer.__name__ == OutputLayer.__name__:

                continue

            layer_gradient = cost_func_derivative

            for derived_index, derived_layer in reversed(list(enumerate(self.layers))):
                if derived_layer.trainable == False:
                    continue

                if derived_index == cur_layer_index:
                    delta_b[cur_layer_index] += sum(layer_gradient)

                    layer_gradient = dot(layer_gradient.T, activations_cache[cur_layer_index])

                    delta_W[cur_layer_index] += layer_gradient.T

                    if current_layer.regularization != "":
                        reg_func = Regularizers.reg_methods[current_layer.regularization]
                        reg = reg_func.__func__(self.weights[cur_layer_index],
                                                current_layer.reg_strength)

                        delta_W[cur_layer_index] += reg

                    break

                else:
                    layer_gradient = dot(self.weights[derived_index], layer_gradient.T)

                    if derived_layer.__name__ != OutputLayer.__name__:
                        layer_gradient = dot(derived_layer.activate(dot_prod_cache[derived_index - 1],
                                                                    derivative = True),
                                                                    layer_gradient)
        delta_W = np_array(delta_W)
        delta_b = np_array(delta_b)

        return delta_W, delta_b


    @to_numpy_array
    @add_second_dim
    def train(self, X, y_labels, epochs = 100,
              learning_rate = 0.01, batch_size = 100,
              verbose = True, patience = 10):

        previous_cost = 0
        patience_counter = patience

        for epoch in range(epochs):

            activations_cache, dot_prod_cache = self.forward_propagation(X)

            dW, db = self.back_propagation(y_labels, activations_cache, dot_prod_cache)

            dW *= learning_rate / len(X)
            db /= len(X)

            self.weights -= dW
            self.biases -= db.squeeze()

            cost = self.layers[-1].cost(y_labels, activations_cache[-1])

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

            if verbose == True:
                print("Cost in epoch {} :".format(epoch + 1), cost)


    def predict(self, data):
        a_cache, _ = self.forward_propagation(data)
        return argmax(a_cache[-1], axis = 1)


    def evaluate(self, y_labels, predictions):
        print("\nAccuracy: ",
              self.metrics(y_labels, predictions))


if __name__ == '__main__':
    net = NeuralNet()

    net.add_layer(InputLayer(2, 5, activation = "relu"))
    net.add_layer(DenseLayer(3, activation = "relu"))
    net.add_layer(OutputLayer(1, activation = "sigmoid",
                              cost_function = "binary_crossentropy"))

    from sklearn.datasets import make_moons

    X_moons, y_moons = make_moons(n_samples = 1000, noise = 0.1)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X_moons, y_moons, test_size = 0.2)

    one_hot = y_train

    net.build_model("accuracy", "xavier")

    net.train(X_train[:50], one_hot[:50])

    y_pred = net.predict(X_test)

    net.evaluate(y_test, y_pred)