import numpy as np
from utils.data_processing import decode_one_hot
from decorators import to_numpy_array


def pick_metrics_method(method: str):
    if method == "accuracy":
        return accuracy


@to_numpy_array
def accuracy(target_labels: np.ndarray, predictions: np.ndarray):
    assert len(target_labels) == len(predictions), ("Incompatibile shapes, target_labels has length {}, "
                        "predictions has length {}.".format(len(target_labels), len(predictions)))

    predictions = decode_one_hot(predictions)
    target_labels = decode_one_hot(target_labels)

    return str((np.sum((target_labels == predictions)) / len(target_labels)) * 100) + " %"