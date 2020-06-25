import time
import numpy as np
from data_processing import encode_one_hot
from utils.data_processing import decode_one_hot

labels_size = 30

labels = np.random.randint(0, 10, labels_size)

