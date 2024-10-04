ESTIMATOR_REGISTRY = {}


def register_estimator(cls: object) -> object:
    ESTIMATOR_REGISTRY[cls.__name__] = cls
    return cls
