import numpy as np
from utils.cost_functions import mean_squared_error
from decorators import to_numpy_array, add_second_dim


class LinearRegressor:
    def __init__(self):
        self.params = None
        self.features = None

    def weights_init(self):
        self.params = np.random.randn(1, self.features + 1)

    @to_numpy_array
    @add_second_dim
    def train(self, X, Y, learning_rate = 0.0001,
              iterations = 10000, patience = 10,
              verbose = True):

            patience_counter = patience

            if X.shape[1] != len(Y):
                X = X.T

            self.features = X.shape[0]

            previous_cost = 0

            self.weights_init()

            for i in range(iterations):
                results = self.predict(X)
                dW, db = mean_squared_error(Y, results, X, derivative = True)

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

        return np.dot(self.params[0][1:], X) + self.params[0][0]


model = LinearRegressor()
x = [[1,2],[3,4],[0.5,1],[2,4]]
y = [10, 20, 30, 40]

model.train(x, y)


print(model.predict(x))