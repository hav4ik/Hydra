import sys
from .naive import Naive
from .averaging import Averaging
from .mgda import MGDA


def load_trainer(trainer_name):
    """Dynamically loads the specified trainer
    """
    if not hasattr(sys.modules[__name__], trainer_name):
        raise ValueError('Trainer {} not found.'.format(trainer_name))
    return getattr(sys.modules[__name__], trainer_name)
