from numpy import sum, log, array as np_array, dot
from decorators import add_second_dim


def compute_cost(y_labels, output, cost_function: str, derivative = False):
    function = None

    if cost_function == "categorical_crossentropy":
        function = categorical_crossentropy

    elif cost_function == "binary_crossentropy":
        function = binary_crossentropy

    elif cost_function == "mean_squared_error":
        function = mean_squared_error

    if derivative == True:
        return function(y_labels, output, derivative=True)

    return function(y_labels, output)

@add_second_dim
def categorical_crossentropy(y_labels, output, derivative = False):
    if derivative == True:
        return output - y_labels

    return -sum(y_labels * log(output)) / len(output)

@add_second_dim
def binary_crossentropy(y_labels, output, derivative = False):
    if derivative == True:
        return (output - y_labels)

    cost = 0

    for index, label in enumerate(y_labels):
        if label == 0:
            cost += -log(1 - output[index])
        else:
            cost += -log(output[index])

    return cost / len(output)

@add_second_dim
def mean_squared_error(y_labels, output, X = None, derivative = False):
    if derivative == True:
        dW = sum((output - y_labels) * X.T) / len(output)
        db = sum((output - y_labels)) / len(output)

        return dW, db

    return sum((output - y_labels) ** 2) / len(output)