import time
import numpy as np
from preprocessing_extension import encode_one_hot
from utils.data_processing import decode_one_hot
from deep_learning.neural_net import *

labels_size = 30

labels = np.random.randint(0, 10, labels_size)

print(encode_one_hot(labels))


print("="*40 + " Model Architecture " + "="*40)
print("Layer" + " "*20 + "Input Shape" + " "*20 + "Output Shape" + " "*20 + "Activation")
print("-"*100)