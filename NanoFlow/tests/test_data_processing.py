import unittest
from utils.data_processing import *
import numpy as np


class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        self.class_vector = np.random.randint(0, 10, 500)
        self.one_hot = one_hot_encoder(self.class_vector)

    def test_one_hot_encoder(self):
        self.assertEqual(self.one_hot.shape[0], self.class_vector.shape[0])
        self.assertEqual(self.one_hot.shape[1], 10)
        self.assertEqual(np.sum(self.one_hot), 500)
        self.assertEqual(np.allclose(np.argmax(self.one_hot, axis = 1), self.class_vector), True)

    def test_decode_one_hot(self):
        one_hot_1_class = np.random.randint(0, 1, 500).reshape(500, 1)

        self.assertEqual(np.allclose(decode_one_hot(self.one_hot), self.class_vector), True)
        self.assertRaises(ValueError, decode_one_hot(one_hot_1_class))

    def test_data_split(self):
        pass



if __name__ == '__main__':
    unittest.main()