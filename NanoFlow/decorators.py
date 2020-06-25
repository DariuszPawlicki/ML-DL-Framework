from numpy import ndarray, array


def convert_to_numpy_array(func):

    def wrapper(*args, **kwargs):
        args = list(args)

        for i, arg in enumerate(args):
            if not (isinstance(arg, ndarray)) and \
                    (isinstance(arg, list) or isinstance(arg, tuple)):

                args[i] = array(arg)

        for key, kwarg in kwargs.items():
            if not isinstance(kwarg, ndarray) and \
                    (isinstance(kwarg, list) or isinstance(kwarg, tuple)):

                kwargs[key] = array(kwarg)

        return func(*args, **kwargs)

    return wrapper


def expand_dimension(func):
    def wrapper(*args, **kwargs):
        args = list(args)

        for i, arg in enumerate(args):
            if isinstance(arg, ndarray):

                try:
                    assert arg.shape[1]
                except(IndexError):
                    arg = arg.reshape(arg.shape[0], 1)

                args[i] = array(arg)

        for key, kwarg in kwargs.items():
            if isinstance(kwarg, ndarray):
                try:
                    assert kwarg.shape[1]
                except(IndexError):
                    kwarg = kwarg.reshape(kwarg.shape[0], 1)

            kwargs[key] = kwarg

        return func(*args, **kwargs)

    return wrapper