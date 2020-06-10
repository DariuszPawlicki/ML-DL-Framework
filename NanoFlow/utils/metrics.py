from numpy import sum, ndarray, var
from utils.data_processing import decode_one_hot
from decorators import to_numpy_array, add_second_dim
from utils.cost_functions import mean_squared_error


def pick_metrics_method(method: str):
    if method == "accuracy":
        return accuracy
    elif method == "r_squared":
        return r_squared


@to_numpy_array
@add_second_dim
def accuracy(target_labels: ndarray, predictions: ndarray):
    assert len(target_labels) == len(predictions), ("Incompatibile shapes, target_labels has length {}, "
                        "predictions has length {}.".format(len(target_labels), len(predictions)))

    predictions = decode_one_hot(predictions)
    target_labels = decode_one_hot(target_labels)

    return str((sum((target_labels == predictions)) / len(target_labels)) * 100) + " %"


@to_numpy_array
@add_second_dim
def r_squared(Y, predictions):
    variance = var(Y)
    mse = mean_squared_error(Y, predictions)

    return 1 - (variance / mse)