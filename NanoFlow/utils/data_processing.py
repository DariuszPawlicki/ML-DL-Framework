import numpy as np
from decorators import convert_to_numpy_array, expand_dimension
from preprocessing_extension import encode_one_hot


@convert_to_numpy_array
@expand_dimension
def one_hot_encoder(labels_list):
     return encode_one_hot(labels_list)


@convert_to_numpy_array
def decode_one_hot(labels_list):

    try:
        assert labels_list.shape[1] >= 2

        classes = []

        for row in labels_list:
            classes.append(np.argmax(row))

        return np.array(classes)

    except AssertionError:
        raise AssertionError("Cannot decode labels because of "
              "incorrect dimension size. One-Hot matrix should have"
              "labels dimension of size at least two.")

@convert_to_numpy_array
def data_split(data: np.ndarray, labels_list: np.ndarray, validation_split = False,
               shuffle_data = True, split_size = 0.2):

    """
    Returns splitted labels and data in 2 two-dimensional lists - or 3 lists if validation_split flag
    is set to True, then third list is an validation dataset -
    where first dimension is data and second are labels.
    """
    try:
        assert len(data) == len(labels_list)

        if shuffle_data == True:
            shuffler = np.random.permutation(len(labels_list))

            data = data[shuffler]
            labels_list = labels_list[shuffler]

        train = []
        test = []

        if validation_split == True:
            train_size = int((1 - split_size * 2) * len(labels_list))
        else:
            train_size = int((1 - split_size) * len(labels_list))

        test_size = int(split_size * len(labels_list))

        train.append(data[:train_size])
        train.append(labels_list[:train_size])

        test.append(data[train_size:train_size + test_size])
        test.append(labels_list[train_size:train_size + test_size])

        if validation_split == True:
            valid = []
            valid_size = int(split_size * len(labels_list))

            valid.append(data[train_size + test_size:train_size + test_size + valid_size])
            valid.append(labels_list[train_size + test_size:train_size + test_size + valid_size])

            return train, test, valid

        return train, test

    except AssertionError:
        raise AssertionError("Incompatibile shapes, data has length {} "
              "labels has length {}.".format(len(data), len(labels_list)))