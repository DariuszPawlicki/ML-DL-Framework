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


def add_second_dim(func):
    def wrapper(*args, **kwargs):
        args = list(args)

        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):

                try:
                    assert arg.shape[1]
                except(IndexError):
                    arg = arg.reshape(arg.shape[0], 1)

                args[i] = np.array(arg)

        for key, kwarg in kwargs.items():
            if isinstance(kwarg, np.ndarray):
                try:
                    assert kwarg.shape[1]
                except(IndexError):
                    kwarg = kwarg.reshape(kwarg.shape[0], 1)

            kwargs[key] = kwarg

        return func(*args, **kwargs)

    return wrapper