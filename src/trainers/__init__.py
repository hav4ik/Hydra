import sys
from .naive import naive
from .averaging import averaging
from .mgda import mgda


def load_trainer(trainer_name):
    """Dynamically loads the specified trainer
    """
    if not hasattr(sys.modules[__name__], trainer_name):
        raise ValueError
    return getattr(sys.modules[__name__], trainer_name)
