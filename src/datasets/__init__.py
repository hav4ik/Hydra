import sys
from .toy import toy


def load_dataset(dataset_name, dataset_kwargs):
    """Dynamically loads the specified `torch.utils.data.Dataset` object.
    """
    if not hasattr(sys.modules[__name__], dataset_name):
        raise ValueError
    dataset_def = getattr(sys.modules[__name__], dataset_name)
    return dataset_def(**dataset_kwargs)
