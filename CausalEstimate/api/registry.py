ESTIMATOR_REGISTRY = {}


def register_estimator(cls):
    ESTIMATOR_REGISTRY[cls.__name__] = cls
    return cls
