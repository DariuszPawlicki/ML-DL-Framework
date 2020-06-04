import numpy as np
from utils.cost_functions import mean_squared_error
from decorators import to_numpy_array


class LinearRegressor:
    def __init__(self):
        self.params = None
        self.features = None

    def weights_init(self):
        self.params = np.random.randn(1, self.features + 1)

    @to_numpy_array
    def fit(self, X, Y, iterations = 1000):
        X = X.reshape(X.shape[0], 1)

        if X.shape[1] != len(Y):
            X = X.T

        self.features = X.shape[0]

        self.weights_init()

        for i in range(iterations):
            results = np.dot(self.params[1:], X) + self.params[0]
            gradient = mean_squared_error(Y, results, derivative = True)




model = LinearRegressor()
model.fit([1, 2, 3, 4], [10, 20, 30, 40])