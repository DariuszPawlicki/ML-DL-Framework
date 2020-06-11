from numpy import dot, random, sum
from utils.cost_functions import mean_squared_error, binary_crossentropy
from decorators import to_numpy_array, add_second_dim
from utils.metrics import pick_metrics_method
from abc import ABC, abstractmethod
from utils.activations import sigmoid


class LinearModel(ABC):
    def __init__(self):
        self.params = None
        self.features = None
        super().__init__()

    @abstractmethod
    def weights_init(self):
        pass

    @abstractmethod
    def train(self, X, Y, learning_rate = 0.0001,
              iterations = 1000, patience = 10,
              verbose = True):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def evaluate(self, Y, predictions, metrics: str):
        pass



class LinearRegressor(LinearModel):
    def __init__(self):
        super().__init__()

    def weights_init(self):
        self.params = random.randn(1, self.features + 1)

    @to_numpy_array
    @add_second_dim
    def train(self, X, Y, learning_rate=0.001,
              iterations=10000, patience=10,
              verbose=True):

        patience_counter = patience

        if X.shape[1] != len(Y):
            X = X.T

        self.features = X.shape[0]

        previous_cost = 0

        self.weights_init()

        for i in range(iterations):
            results = self.predict(X)
            dW, db = mean_squared_error(Y, results, X, derivative=True)

            cost = mean_squared_error(Y, results)

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

            self.params[1:] -= learning_rate * dW
            self.params[0] -= learning_rate * db

            if verbose == True:
                print("Cost in {} iteration: ".format(i), cost)

    @to_numpy_array
    @add_second_dim
    def predict(self, X):
        if X.shape[0] != self.features:
            X = X.T

        return dot(self.params[1:], X) + self.params[0]

    @to_numpy_array
    @add_second_dim
    def evaluate(self, Y, predictions, metrics: str):
        metrics_method = pick_metrics_method(metrics)

        print(f"\n{metrics[0].upper() + metrics[1:]}: ",
              metrics_method(Y, predictions))



class LogisticRegressor(LinearModel):
    def __init__(self):
        super().__init__()


    def weights_init(self):
        self.params = random.randn(1, self.features + 1)


    @to_numpy_array
    @add_second_dim
    def train(self, X, Y, learning_rate=0.01, iterations=1000,
              patience=10, verbose=True):

        if X.shape[1] != len(Y):
            X = X.T

        patience_counter = patience
        previous_cost = 0

        self.features = X.shape[0]
        self.weights_init()

        for i in range(iterations):
            results = self.predict(X)

            cost = binary_crossentropy(Y, results)

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

            dW = dot(binary_crossentropy(Y,results, derivative = True).T, X.T) / len(results)

            db = sum(binary_crossentropy(Y, results, derivative=True).T) / len(results)

            self.params[0][1:] -= (learning_rate * dW).squeeze()
            self.params[0][0] -= learning_rate * db

            if verbose == True:
                print("Cost in {} iteration: ".format(i), cost.squeeze())


    @to_numpy_array
    @add_second_dim
    def predict(self, X):
        if X.shape[0] != self.features:
            X = X.T

        return sigmoid(dot(self.params[0][1:], X) + self.params[0][0])

    @to_numpy_array
    @add_second_dim
    def evaluate(self, Y, predictions, metrics: str):

        metrics_method = pick_metrics_method(metrics)

        for index, pred in enumerate(predictions):
            if pred <= 0.5:
                predictions[index] = 0
            else:
                predictions[index] = 1

        print(f"\n{metrics[0].upper() + metrics[1:]}: ",
              metrics_method(Y, predictions))