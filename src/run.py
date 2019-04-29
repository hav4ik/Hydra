from argparse import ArgumentParser
import torch

from utils import config_utils
import datasets


def run(config):
    """Main runner that dynamically imports and executes other modules
    """
    cfg = config_utils.read_config(config)

    # import loaders
    train_loader, test_loader = datasets.load_dataset(
            cfg['datasets']['name'], cfg['datasets']['kwargs'])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', help='YAML configuration')

    args = parser.parse_args()
    run(args.config)
