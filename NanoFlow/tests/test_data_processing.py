import unittest
from utils.data_processing import *
import numpy as np


class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        self.class_vector = np.sort(np.random.randint(0, 10, 500), axis=0)
        self.one_hot = one_hot_encoder(self.class_vector)


    def test_one_hot_encoder(self):
        self.assertEqual(self.one_hot.shape[0], self.class_vector.shape[0])
        self.assertEqual(self.one_hot.shape[1], 10)
        self.assertEqual(np.sum(self.one_hot), 500)
        self.assertEqual(np.allclose(np.argmax(self.one_hot, axis = 1), self.class_vector), True)


    def test_decode_one_hot(self):
        one_hot_1_class = np.random.randint(0, 1, 500).reshape(500, 1)

        self.assertEqual(np.allclose(decode_one_hot(self.one_hot), self.class_vector), True)
        self.assertRaises(AssertionError, decode_one_hot, one_hot_1_class)


    def test_data_split(self):
        test_data = np.random.randn(500, 5)

        split_test_data, split_test_labels = data_split(test_data, self.class_vector,
                                                        validation_split=False, shuffle_data=True,
                                                        split_size=0.2)

        split_test_dataV, split_test_labelsV, split_test_valid = data_split(test_data, self.class_vector,
                                                                            validation_split=True, shuffle_data=True,
                                                                            split_size=0.2)
        self.assertEqual(len(split_test_data[0]), 0.8 * len(test_data))
        self.assertEqual(len(split_test_dataV[0]), 0.6 * len(test_data))
        self.assertFalse(np.allclose(test_data[:int(0.8 * len(test_data))], split_test_data[0]))
        self.assertRaises(AssertionError, data_split, test_data, self.class_vector[:10])


if __name__ == '__main__':
    unittest.main()