import csv
import numpy as np


def load_mnist(size = None, reshape = False):
    """
    Returns a mnist dataset in form of a dictionary,
    with "labels" and "data" keys.

    If "size" argument is set to default - None - returns all
    10000 rows of data set, else if "size = n" returns "n" rows
    from dataset.

    If reshape = True all data will be
    reshaped from [n, ] to [n, 1] shape.
    """

    data = []
    labels = []

    with open("../datasets/mnist_test.csv", newline = "") as file:
        reader = csv.reader(file, delimiter = ",")

        for index, row in enumerate(reader):
            if size != None and index == size:
                break

            data.append(np.array(row[1:], dtype = np.float64))
            labels.append(row[0])

            if reshape == True:
                data[index] = data[index].reshape(data[index].shape[0], 1)

    return {"data": data, "labels": labels}