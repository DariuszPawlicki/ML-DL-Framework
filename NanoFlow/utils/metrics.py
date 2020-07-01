from utils.data_processing import decode_one_hot
from decorators import convert_to_numpy_array, expand_dimension
from utils.cost_functions import mean_squared_error
import numpy as np


def pick_metrics_method(method: str):
    if method == "accuracy":
        return accuracy
    elif method == "r_squared":
        return r_squared


@convert_to_numpy_array
@expand_dimension
def accuracy(target_labels: np.ndarray, predictions: np.ndarray):
    assert len(target_labels) == len(predictions), ("Incompatibile shapes, target_labels has length {}, "
                        "predictions has length {}.".format(len(target_labels), len(predictions)))

    predictions = predictions.squeeze()
    target_labels = decode_one_hot(target_labels)

    return str((np.sum((target_labels == predictions)) / len(target_labels)) * 100) + " %"


@convert_to_numpy_array
@expand_dimension
def r_squared(target_labels, predictions):
    variance = np.var(target_labels)
    mse = mean_squared_error(target_labels, predictions)

    return 1 - (variance / mse)