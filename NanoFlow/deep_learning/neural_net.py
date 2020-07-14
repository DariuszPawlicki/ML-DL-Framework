from utils.data_processing import *
from utils.metrics import *
from decorators import expand_dimension, convert_to_numpy_array
from deep_learning.layers import *
from utils.regularizers import Regularizers
from utils.optimizers import *


class NeuralNet:

    def __init__(self):
        self.__layers = []
        self.__weights = None
        self.__biases = None
        self.__metrics = None
        self.__optimizer: Optimizer = None
        self.__learning_rate = None
        self.__decay_rate = None


    def add_layer(self, layer: Layer):
        if len(self.__layers) == 0 and layer.__name__ != InputLayer.__name__:
            raise TypeError("First layer of model"
                            " must be an 'InputLayer' type.")

        self.__layers.append(layer)

        if len(self.__layers) > 1:
            self.__layers[-1].input_shape = self.__layers[-2].output_shape


    def build_model(self, metrics: str, param_init: str,
                    optimizer: Optimizer, learning_rate = 0.001,
                    decay_rate = 0):

            if self.__layers[-1].__name__ != OutputLayer.__name__:
                raise TypeError("Last layer of model must be an"
                                " 'OutputLayer' type.")

            self.__weights_init(param_init = param_init)
            self.__metrics = pick_metrics_method(metrics)
            self.__optimizer = optimizer
            self.__learning_rate = learning_rate
            self.__decay_rate = decay_rate

    def summary(self):
        print("\n\n")
        print("="*100)
        print("Layer" + " "*20 + "Input Shape" + " "*10 + "Output Shape" + " "*14 + "Activation")
        print("-"*100)

        for layer in self.__layers:
            print(layer.__name__ + " "*18, end = "")
            print(str(layer.input_shape) + " " * 20, end="")
            print(str(layer.output_shape) + " " * 20, end="")
            print(layer.activation + " " * 20)
            print("_"*100)

        print("="*100)
        print("\n\n")


    @convert_to_numpy_array
    @expand_dimension
    def train(self, X, target_labels, epochs=100,
              verbose=True, patience=10, gradient_checking = False):

        previous_cost = 0
        patience_counter = patience

        for epoch in range(epochs):

            activations_cache, dot_prod_cache = self.__forward_propagation(X)

            dW, db = self.__back_propagation(target_labels, activations_cache, dot_prod_cache)

            if gradient_checking == True:
                self.grad_check(dW, db, X, target_labels)

            self.__weights, self.__biases = self.__optimizer.update_weights(self.__weights,
                                                                            self.__biases,
                                                                            dW, db,
                                                                            self.__learning_rate,
                                                                            self.__layers)

            cost = self.__layers[-1].cost(target_labels, activations_cache[-1]).squeeze()

            if cost >= previous_cost:
                patience_counter -= 1

                if patience_counter == 0:
                    print("Gradient descent is on plateau/diverging.")

                    print("Cost: ", cost)
                    break
            else:
                patience_counter = patience

            previous_cost = cost

            if self.__decay_rate > 0:
                self.__learning_rate *= 1/(1 + self.__decay_rate * epoch)

            if verbose == True:
                print("Cost in epoch {} :".format(epoch + 1), cost)


    def grad_check(self, dW, db, X, target_labels, epsilon=1e-7):

        dW_approx = []

        for layer_gradient in dW:
            dW_approx.append(np.zeros(layer_gradient.shape))

        for layer_index, layer_weights in enumerate(self.__weights):

            if self.__layers[layer_index].trainable == False:
                continue

            for row_id, row in enumerate(layer_weights):

                for param_id, param in enumerate(row):

                    param_cached = param
                    param_plus = param + epsilon
                    param_minus = param - epsilon

                    self.__weights[layer_index][row_id][param_id] = param_plus

                    activations_plus, _ = self.__forward_propagation(X)

                    self.__weights[layer_index][row_id][param_id] = param_minus

                    activations_minus, _ = self.__forward_propagation(X)

                    self.__weights[layer_index][row_id][param_id] = param_cached

                    J_plus = self.__layers[-1].cost(target_labels, activations_plus[-1])
                    J_minus = self.__layers[-1].cost(target_labels, activations_minus[-1])

                    two_sided_difference = (J_plus - J_minus) / (2 * epsilon)
                    dW_approx[layer_index][row_id][param_id] = two_sided_difference

        dW_approx_vector = np.array([])
        dW_vector = np.array([])

        for approx_weights_grad, weights_grad in zip (dW_approx, dW):
            dW_approx_vector = np.concatenate((dW_approx_vector, approx_weights_grad.ravel()))
            dW_vector = np.concatenate((dW_vector, weights_grad.ravel()))

        numerator = np.linalg.norm(dW_vector - dW_approx_vector)
        denominator = np.linalg.norm(dW_vector) + np.linalg.norm(dW_approx_vector)

        difference = numerator / denominator

        if difference > 1e-7:
            raise ValueError("Numerical and analytical gradients differ a lot,\n"
                             "            check backpropagation derivation. \n"
                             "            Difference is equal {}".format(difference))


    def predict(self, data):
        a_cache, _ = self.__forward_propagation(data, training= False)
        return np.argmax(a_cache[-1], axis = 1)


    def evaluate(self, target_labels, predictions):
        print("\nAccuracy: ",
              self.__metrics(target_labels, predictions))


    def __divide_on_batches(self, X, y, batch_size):

        X_batches = []
        y_batches = []

        batches = int(y.shape[0] / batch_size)
        data_remaining = y.shape[0] % batch_size

        for i in range(batches):
            X_batches.append(X[i * batch_size: (i * batch_size) + batch_size])
            y_batches.append(y[i * batch_size: (i * batch_size) + batch_size])

        X_batches.append(X[X.shape[0] - data_remaining:])
        y_batches.append(y[y.shape[0] - data_remaining:])

        return X_batches, y_batches


    def __weights_init(self, param_init: str):
        weights = []
        biases = []

        for cur_layer_index, current_layer in enumerate(self.__layers):

            if current_layer.trainable == True:

                if param_init == "xavier":
                    method = np.sqrt(1 / current_layer.input_shape)
                elif param_init == "he_normal":
                    method = np.sqrt(2 / (self.__layers[cur_layer_index - 1].input_shape +
                                          current_layer.input_shape))
                else:
                    method = 1

                weights.append(np.random.randn(current_layer.input_shape,
                                               current_layer.output_shape) * method)

                biases.append(float(1))

            else:
                weights.append(np.array([0]))
                biases.append(0)


        self.__weights = weights
        self.__biases = np.array(biases)
        self.__biases = self.__biases.reshape(self.__biases.shape[0], 1)


    @convert_to_numpy_array
    @expand_dimension
    def __forward_propagation(self, X, training = True):
        if X.shape[1] != self.__layers[0].input_shape:  # Reshaping X [x, n] where n are features of data
            X = X.T

        prev_activations = X  # Previous layer activation

        dot_product = None  # Dot product of previous layer
                            #  activations and current layer weights

        activations_cache = []
        dot_prod_cache = []

        activations_cache.append(X)

        for cur_layer_index, current_layer in enumerate(self.__layers):

            if training == False and \
                    current_layer.__name__ == Dropout.__name__:
                continue

            if current_layer.trainable:
                dot_product = np.dot(prev_activations, self.__weights[cur_layer_index]) + \
                              self.__biases[cur_layer_index]

            prev_activations = current_layer.activate(dot_product)

            activations_cache.append(prev_activations)
            dot_prod_cache.append(dot_product)


        return activations_cache, dot_prod_cache


    def __back_propagation(self, target_labels, activations_cache, dot_prod_cache):
        output_layer = self.__layers[-1]

        cost_func_derivative = output_layer.cost(target_labels, activations_cache[-1],
                                                 derivative = True)
        """
        Cost function derivative, w.r.t to outputs.
        """
        output_weights_gradient = np.dot(activations_cache[-2].T,
                                            cost_func_derivative)
        """
        Cost function derivative, w.r.t to output layer weights.
        """

        delta_W = []
        delta_b = np.zeros((len(self.__layers), 1))

        for current_layer in self.__layers:
            if current_layer.trainable == True:
                delta_W.append(np.zeros((current_layer.input_shape,
                                      current_layer.output_shape)))
            else:
                delta_W.append(np.array([0]))

        delta_W[-1] += output_weights_gradient
        delta_b[-1] += np.sum(cost_func_derivative)

        for cur_layer_index, current_layer in reversed(list(enumerate(self.__layers))):
            """
             Loop for deriving cost function
             with respect to 'current_layer' weights.
            """

            if current_layer.trainable == False or \
                    current_layer.__name__ == OutputLayer.__name__:
                """
                Layers like batch normalization or dropout are not differentiable.
                Output layer isn't also derived here, because it was derived earlier.
                """
                continue

            layer_gradient = cost_func_derivative

            for derived_layer_index, derived_layer in reversed(list(enumerate(self.__layers))):
                if derived_layer.trainable == False:
                    continue

                if derived_layer.__name__ != OutputLayer.__name__:
                    activ_func_derivative = derived_layer.activate(dot_prod_cache[derived_layer_index],
                                                                   derivative=True)
                    layer_gradient *= activ_func_derivative

                if derived_layer_index == cur_layer_index:
                    delta_b[cur_layer_index] += np.sum(layer_gradient)

                    layer_gradient = np.dot(layer_gradient.T, activations_cache[cur_layer_index])

                    delta_W[cur_layer_index] += layer_gradient.T

                    break

                else:
                    layer_gradient = np.dot(layer_gradient, self.__weights[derived_layer_index].T)

        for layer_id, layer_gradient in enumerate(delta_W):
            if self.__layers[layer_id].trainable == True:
                layer_gradient /= len(activations_cache[0])

        delta_b = np.array(delta_b) / len(activations_cache[0])

        return delta_W, delta_b


if __name__ == '__main__':
    net = NeuralNet()

    net.add_layer(InputLayer(784, 128,activation="relu"))
    net.add_layer(DenseLayer(64, activation="relu"))
    net.add_layer(DenseLayer(32, activation="relu"))
    net.add_layer(OutputLayer(10, activation="softmax",
                              cost_function="categorical_crossentropy"))

    from datasets.load_data import load_mnist

    data = load_mnist()

    X_train = data["data"][:5000]
    y_train = data["labels"][:5000]

    X_test = data["data"][9000:]
    y_test = data["labels"][9000:]

    y_train = one_hot_encoder(y_train)
    y_test = one_hot_encoder(y_test)


    net.build_model("accuracy", "xavier", SGD(beta=0), learning_rate=0.001,
                    decay_rate=0)

    net.train(X_train, y_train, epochs=100)

    predictions = net.predict(X_test)

    net.evaluate(y_test, predictions)

    #TODO: dodać do funkcji grad_check sprawdzenie pochodnej biasu