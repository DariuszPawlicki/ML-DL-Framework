import numpy as np

def one_hot_encoder(target_labels):
    target_labels = np.array(target_labels)

    if target_labels.shape[0] == None:
        target_labels = target_labels.T

    classes = np.unique(target_labels, return_counts = True)

    encoded_labels = np.zeros((target_labels.shape[0], len(classes[1])))

    for index, label in enumerate(target_labels):
        if 0 in classes[0]:
            encoded_labels[index][label] = 1
        else:
            encoded_labels[index][label - 1] = 1

    return encoded_labels


def data_split(data, labels, validation_split = True,
               shuffle_data = True, split_size = 0.2):

    assert len(data) == len(labels), ("Incompatibile shapes, data has length {} "
                        "labels has length {}.".format(len(data), len(labels)))

    if shuffle_data == True:
        shuffler = np.random.permutation(len(labels))

        data = data[shuffler]
        labels = labels[shuffler]

    train = []
    test = []

    train_size = int((1 - split_size * 2) * len(labels))
    test_size = int(split_size * len(labels))

    train.append(data[:train_size])
    train.append(labels[:train_size])

    test.append(data[train_size:train_size + test_size])
    test.append(labels[train_size:train_size + test_size])

    if validation_split == True:
        valid = []
        valid_size = len(labels) - (train_size + test_size)

        valid.append(data[train_size + test_size:train_size + test_size + valid_size])
        valid.append(labels[train_size + test_size:train_size + test_size + valid_size])

        return train, test, valid

    return train, test



print(data_split([1,2,3], [2]))