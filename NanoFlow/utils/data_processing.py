import numpy as np
from decorators import to_numpy_array


@to_numpy_array
def one_hot_encoder(target_labels: np.ndarray):

    classes = np.unique(target_labels, return_counts = True)

    one_hot_labels = np.zeros((target_labels.shape[0], len(classes[1])))

    new_ids = {}

    for i, cl in enumerate(np.sort(classes[0])):
        """
        Changing classes numeration, e.g. if classes in 'target_labels'
        aren't continuous - [0, 1, 3] - this loop will transform them to [0, 1, 2].
        If 'target_labels' classes are continuous -  [0, 1, 2] - they'll not change.       
        """
        new_ids[cl] = i

    new_encoding = [new_ids[cl] for cl in target_labels]

    for index, label in enumerate(new_encoding):
        one_hot_labels[index][label] = 1

    return np.array(one_hot_labels)


@to_numpy_array
def decode_one_hot(labels: np.ndarray):

    try:
        assert labels.shape[1] >= 2

        classes = []

        for row in labels:
            classes.append(np.argmax(row))

        return np.array(classes)

    except AssertionError:
        return labels



@to_numpy_array
def data_split(data: np.ndarray, labels: np.ndarray, validation_split = False,
               shuffle_data = True, split_size = 0.2):

    assert len(data) == len(labels), ("Incompatibile shapes, data has length {} "
                        "labels has length {}.".format(len(data), len(labels)))

    if shuffle_data == True:
        shuffler = np.random.permutation(len(labels))

        data = data[shuffler]
        labels = labels[shuffler]

    train = []
    test = []

    if validation_split == True:
        train_size = int((1 - split_size * 2) * len(labels))
    else:
        train_size = int((1 - split_size) * len(labels))

    test_size = int(split_size * len(labels))

    train.append(data[:train_size])
    train.append(labels[:train_size])

    test.append(data[train_size:train_size + test_size])
    test.append(labels[train_size:train_size + test_size])

    if validation_split == True:
        valid = []
        valid_size = int(split_size * len(labels))

        valid.append(data[train_size + test_size:train_size + test_size + valid_size])
        valid.append(labels[train_size + test_size:train_size + test_size + valid_size])

        return train, test, valid

    return train, test


def probabilities_to_labels(predictions: np.ndarray):
    return np.argmax(predictions, axis = 1)