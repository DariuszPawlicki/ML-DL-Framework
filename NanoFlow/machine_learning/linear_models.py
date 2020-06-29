from numpy import dot, random, sum
from utils.cost_functions import mean_squared_error, binary_crossentropy
from decorators import convert_to_numpy_array, expand_dimension
from utils.metrics import pick_metrics_method
from abc import ABC, abstractmethod
from utils.activations import sigmoid


class LinearModel(ABC):
    def __init__(self):
        self.__params = None
        self.features = None
        super().__init__()

    @abstractmethod
    def __weights_init(self):
        pass

    @abstractmethod
    def train(self, X, Y, learning_rate = 0.01,
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

    def __weights_init(self):
        self.__params = random.randn(1, self.features + 1)

    @convert_to_numpy_array
    @expand_dimension
    def train(self, X, Y, learning_rate=0.001,
              iterations=10000, patience=10,
              verbose=True):

        patience_counter = patience

        if X.shape[1] != len(Y):
            X = X.T

        self.features = X.shape[0]

        previous_cost = 0

        self.__weights_init()

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

            self.__params[1:] -= learning_rate * dW
            self.__params[0] -= learning_rate * db

            if verbose == True:
                print("Cost in {} iteration: ".format(i), cost)

    @convert_to_numpy_array
    @expand_dimension
    def predict(self, X):
        if X.shape[0] != self.features:
            X = X.T

        return dot(self.__params[1:], X) + self.__params[0]

    @convert_to_numpy_array
    @expand_dimension
    def evaluate(self, Y, predictions, metrics: str):
        metrics_method = pick_metrics_method(metrics)

        print(f"\n{metrics[0].upper() + metrics[1:]}: ",
              metrics_method(Y, predictions))



class LogisticRegressor(LinearModel):
    def __init__(self):
        super().__init__()


    def __weights_init(self):
        self.__params = random.randn(1, self.features + 1)


    @convert_to_numpy_array
    @expand_dimension
    def train(self, X, Y, learning_rate=0.01, iterations=1000,
              patience=10, verbose=True):

        if X.shape[1] != len(Y):
            X = X.T

        patience_counter = patience
        previous_cost = 0

        self.features = X.shape[0]
        self.__weights_init()

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

            self.__params[0][1:] -= (learning_rate * dW).squeeze()
            self.__params[0][0] -= learning_rate * db

            if verbose == True:
                print("Cost in {} iteration: ".format(i), cost.squeeze())


    @convert_to_numpy_array
    @expand_dimension
    def predict(self, X):
        if X.shape[0] != self.features:
            X = X.T

        return sigmoid(dot(self.__params[0][1:], X) + self.__params[0][0])

    @convert_to_numpy_array
    @expand_dimension
    def evaluate(self, Y, predictions, metrics: str):

        metrics_method = pick_metrics_method(metrics)

        for index, pred in enumerate(predictions):
            if pred <= 0.5:
                predictions[index] = 0
            else:
                predictions[index] = 1

        print(f"\n{metrics[0].upper() + metrics[1:]}: ",
              metrics_method(Y, predictions))