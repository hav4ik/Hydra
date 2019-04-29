from argparse import ArgumentParser
from termcolor import colored

import torch

from utils import config_utils
import datasets
import models
import torchsummary


def run(config,
        n_workers=1):
    """Main runner that dynamically imports and executes other modules
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = config_utils.read_config(config)
    print('{} {}'.format(
        colored('Experiment:', 'green'),
        colored(cfg['experiment'], 'magenta', attrs=['bold'])))
    if torch.cuda.is_available():
        gpuid = torch.cuda.current_device()
        print('  - device: {}'.format(torch.cuda.get_device_name(gpuid)))
    else:
        print('  - device: CPU')
    print('  - output: {}'.format(cfg['out_dir']))

    # import loaders
    train_data, test_data = datasets.load_dataset(
            cfg['datasets']['name'], cfg['datasets']['kwargs'])
    train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=cfg['batch_size'],
            shuffle=True,
            num_workers=n_workers)
    test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=cfg['batch_size'],
            shuffle=False,
            num_workers=n_workers)
    print(colored('\nLoaded datasets:', 'green'))
    print('  - {}: {} train, {} test'.format(
        cfg['datasets']['id'], len(train_data), len(test_data)))

    # import models
    print(colored('\nLoaded model:', 'green'))
    model = models.load_model(cfg['models']['name'], cfg['models']['kwargs'])
    model = model.to(device)
    torchsummary.summary(model, input_size=(1, 28, 28))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('-t', '--workers', type=int, default=1)

    args = parser.parse_args()
    run(config=args.config,
        n_workers=args.workers)
