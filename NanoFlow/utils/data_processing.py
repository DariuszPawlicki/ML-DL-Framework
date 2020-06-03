import numpy as np


def one_hot_encoder(target_labels: np.ndarray):
    target_labels = np.array(target_labels)

    if target_labels.shape[0] == None:
        target_labels = target_labels.T

    classes = np.unique(target_labels, return_counts = True)

    encoded_labels = np.zeros((target_labels.shape[0], len(classes[1])))

    for index, label in enumerate(target_labels):
        if len(classes[1]) == 2 and [0, 1] in classes[0]:
            encoded_labels[index][label] = 1
        else:
            encoded_labels[index][label - 1] = 1

    return np.array(encoded_labels)


def decode_one_hot(labels: np.ndarray):

    if not labels.shape[1] >= 2:
        return labels

    if labels.shape[1] != len(np.unique(labels, return_counts = True)[1]):
        labels = labels.T

    classes = []

    for row in labels:
        classes.append(np.argmax(row))

    return np.array(classes)


def data_split(data: np.ndarray, labels: np.ndarray, validation_split = False,
               shuffle_data = True, split_size = 0.2):

    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

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