from numpy import sum, log
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


def categorical_crossentropy(y_labels, output, derivative = False):
    if derivative == True:  # Derivative of softmax crossentropy function with respect to weights
        return output - y_labels

    return -sum(y_labels * log(output)) / len(output)


def binary_crossentropy(y_labels, output, derivative = False):
    if derivative == True:
        return output - y_labels

    if y_labels[0] == 1:
        return -sum(log(1 - output)) / len(output)
    elif y_labels[1] == 1:
        return -sum(log(output)) / len(output)


@add_second_dim
def mean_squared_error(target, predictions, X = None, derivative = False):
    if derivative == True:
        dW = sum((predictions - target) * X.T) / len(predictions)
        db = sum((predictions - target)) / len(predictions)

        return dW, db

    return sum((predictions - target) ** 2) / len(predictions)