from numpy import unique, zeros, sort, ndarray, argmax, array, random
from decorators import to_numpy_array, add_second_dim


@to_numpy_array
def one_hot_encoder(target_labels: ndarray):

    classes = unique(target_labels, return_counts = True)

    one_hot_labels = zeros((target_labels.shape[0], len(classes[1])))

    new_ids = {}

    for i, cl in enumerate(sort(classes[0])):
        """
        Changing classes numeration, e.g. if classes in 'target_labels'
        aren't continuous - [0, 1, 3] - this loop will transform them to [0, 1, 2].
        If 'target_labels' classes are continuous -  [0, 1, 2] - they'll not change.       
        """
        new_ids[cl] = i

    new_encoding = [new_ids[cl] for cl in target_labels]

    for index, label in enumerate(new_encoding):
        one_hot_labels[index][label] = 1

    return array(one_hot_labels)


@to_numpy_array
def decode_one_hot(labels: ndarray):

    try:
        assert labels.shape[1] >= 2

        classes = []

        for row in labels:
            classes.append(argmax(row))

        return array(classes)

    except(AssertionError):
        return labels


@to_numpy_array
def data_split(data: ndarray, labels: ndarray, validation_split = False,
               shuffle_data = True, split_size = 0.2):

    """
    Returns splitted labels and data in 2 dimensional lists,
    where first dimension of list is data and second are labels.
    """
    assert len(data) == len(labels), ("Incompatibile shapes, data has length {} "
                        "labels has length {}.".format(len(data), len(labels)))

    if shuffle_data == True:
        shuffler = random.permutation(len(labels))

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