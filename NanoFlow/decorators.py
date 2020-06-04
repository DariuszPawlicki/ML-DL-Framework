import numpy as np


def to_numpy_array(func):

    def wrapper(*args, **kwargs):
        args = list(args)

        for i, arg in enumerate(args):
            if not (isinstance(arg, np.ndarray)) and \
                    (isinstance(arg, list) or isinstance(arg, tuple)):

                args[i] = np.array(arg)

        for key, kwarg in kwargs.items():
            if not isinstance(kwarg, np.ndarray) and \
                    (isinstance(kwarg, list) or isinstance(kwarg, tuple)):

                kwargs[key] = np.array(kwarg)

        return func(*args, **kwargs)

    return wrapper