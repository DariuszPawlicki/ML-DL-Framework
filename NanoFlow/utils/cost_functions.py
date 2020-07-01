import numpy as np
from decorators import expand_dimension


def compute_cost(target_labels, predictions, cost_function: str, derivative = False):
    function = None

    if cost_function == "categorical_crossentropy":
        function = categorical_crossentropy

    elif cost_function == "binary_crossentropy":
        function = binary_crossentropy

    elif cost_function == "mean_squared_error":
        function = mean_squared_error

    if derivative == True:
        return function(target_labels, predictions, derivative=True)

    return function(target_labels, predictions)

@expand_dimension
def categorical_crossentropy(target_labels, predictions, derivative = False):
    if derivative == True:
        return predictions - target_labels

    return -np.sum(target_labels * np.log(predictions)) / len(predictions)

@expand_dimension
def binary_crossentropy(target_labels, predictions, derivative = False):
    if derivative == True:
        return (predictions - target_labels)

    cost = 0

    for index, label in enumerate(target_labels):
        if label == 0:
            cost += -np.log(1 - predictions[index])
        else:
            cost += -np.log(predictions[index])

    return cost / len(predictions)

@expand_dimension
def mean_squared_error(target_labels, predictions, X = None, derivative = False):
    if derivative == True:
        dW = np.sum((predictions - target_labels) * X.T) / len(predictions)
        db = np.sum((predictions - target_labels)) / len(predictions)

        return dW, db

    return np.sum((predictions - target_labels) ** 2) / len(predictions)