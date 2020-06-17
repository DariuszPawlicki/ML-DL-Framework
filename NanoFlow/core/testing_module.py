from utils.data_processing import one_hot_encoder
import time
import numpy as np
from data_processing import encode_one_hot

labels_size = 10

labels = np.random.randint(0, 10, labels_size)

sorted = encode_one_hot(labels)
print(sorted)
print(labels)

