import sys
from .lenet import Lenet


def load_model(model_name, model_kwargs):
    """Dynamically loads the specified `nn.Module` object
    """
    if not hasattr(sys.modules[__name__], model_name):
        raise ValueError
    model_def = getattr(sys.modules[__name__], model_name)
    return model_def(**model_kwargs)
