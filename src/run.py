from argparse import ArgumentParser
from termcolor import colored

import torch

from utils import config_utils, log_utils
import datasets
import models
import torchsummary
import trainers


def run(config,
        n_workers=1,
        resume=False):
    """Main runner that dynamically imports and executes other modules
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = config_utils.read_config(config)

    # Print experiment info
    print('{} {}'.format(
        colored('EXPERIMENT:', 'green'),
        colored(cfg['experiment'], 'magenta', attrs=['bold'])))
    if torch.cuda.is_available():
        gpuid = torch.cuda.current_device()
        print('  - device: {}'.format(torch.cuda.get_device_name(gpuid)))
    else:
        print('  - device: CPU')
    print('  - output: {}'.format(cfg['out_dir']))

    # Import Data Loaders
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
    print(colored('\nDATASETS:', 'green'))
    print('  - {}: {} train, {} test'.format(
        cfg['datasets']['task_id'], len(train_data), len(test_data)))

    # Prepare checkpoints and logging dirs
    tensorboard_logdir, checkpoints_logdir = \
            log_utils.prepare_dirs(cfg['experiment'], cfg['out_dir'], resume)

    # Import Models
    print(colored('\nMODEL:', 'green'))
    model_manager = models.ModelManager(checkpoints_logdir, cfg['task_ids'])
    model = model_manager.load_model(
            model_name=cfg['models']['name'],
            model_weights=cfg['models']['weights'],
            model_kwargs=cfg['models']['kwargs'])
    model = model.to(device)
    torchsummary.summary(model, input_size=(1, 28, 28))

    # Invoke Training
    trainer_def = trainers.load_trainer(cfg['trainer']['name'])
    trainer_def(device=device,
                train_loader=train_loader,
                test_loader=test_loader,
                model=model,
                batch_size=cfg['batch_size'],
                tensorboard_dir=tensorboard_logdir,
                model_manager=model_manager,
                **cfg['trainer']['kwargs'])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('-t', '--workers', type=int, default=1)
    parser.add_argument('-c', '--resume', action='store_true')

    args = parser.parse_args()
    run(config=args.config,
        n_workers=args.workers,
        resume=args.resume)
