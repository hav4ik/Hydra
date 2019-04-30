from argparse import ArgumentParser
import torch

from utils import config_utils, log_utils
import datasets
import models
import trainers


def run(config,
        n_workers=1,
        resume=False):
    """Main runner that dynamically imports and executes other modules
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = config_utils.read_config(config)
    log_utils.print_experiment_info(cfg['experiment'], cfg['out_dir'])

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
    log_utils.print_datasets_info(train_data, test_data)

    # Prepare checkpoints and logging dirs
    tensorboard_logdir, checkpoints_logdir = \
        log_utils.prepare_dirs(cfg['experiment'], cfg['out_dir'], resume)

    # Import Models
    model_manager = models.ModelManager(checkpoints_logdir, cfg['task_ids'])
    model = model_manager.load_model(
            model_name=cfg['models']['name'],
            model_weights=cfg['models']['weights'],
            model_kwargs=cfg['models']['kwargs'])
    model = model.to(device)
    log_utils.print_model_info(model)

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
