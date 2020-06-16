from utils.data_processing import one_hot_encoder
import time
import numpy as np
from data_processing import encode_one_hot


classes_num = 3
labels_size = 2500000

labels = np.random.randint(0, classes_num, labels_size)


start = time.time()
encode_one_hot(labels, labels_size, classes_num)
print("C++ ", time.time() - start)


start = time.time()
one_hot_encoder(labels)
print("\nPython ", time.time() - start)






