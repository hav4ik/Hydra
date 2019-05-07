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


def update_config(config, updates):
    """Modifies the YAML configurations, given a list of YAML updates.
    """
    if isinstance(updates, str):
        updates = [updates]
    for update in updates:
        edits = yaml.safe_load(update)
        for k, v in edits.items():
            node = config
            for ki in k.split('.')[:-1]:
                if ki in node:
                    node = node[ki]
                else:
                    node[ki] = dict()
                    node = node[ki]
            ki = k.split('.')[-1]
            node[ki] = v
    return config
