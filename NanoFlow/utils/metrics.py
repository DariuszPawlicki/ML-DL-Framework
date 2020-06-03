import numpy as np
from utils.data_processing import decode_one_hot


def pick_metrics_method(method: str):
    if method == "accuracy":
        return accuracy


def accuracy(target_labels: np.ndarray, predictions: np.ndarray):
    assert len(target_labels) == len(predictions), ("Incompatibile shapes, target_labels has length {}, "
                        "predictions has length {}.".format(len(target_labels), len(predictions)))

    if not isinstance(target_labels, np.ndarray):
        target_labels = np.array(target_labels)

    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)

    predictions = decode_one_hot(predictions)
    target_labels = decode_one_hot(target_labels)

    return str(((target_labels == predictions).sum() / len(target_labels)) * 100) + " %"