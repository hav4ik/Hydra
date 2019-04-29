import os
import yaml


def read_config(config):
    """Reads a YAML training configurations
    """
    if not isinstance(config, str):
        raise TypeError

    if os.path.isfile(os.path.expanduser(config)):
        with open(config) as stream:
            return yaml.safe_load(stream)
    else:
        return yaml.safe_load(stream)
