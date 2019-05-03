from argparse import ArgumentParser
import torch
from tensorboardX import SummaryWriter

from utils import config_utils, log_utils
from utils import losses as custom_losses
from utils import metrics as custom_metrics
import datasets
import models
import trainers


def import_data_loaders(config, n_workers):
    """Import datasets and wrap them into DataLoaders from configuration
    """
    train_loaders, test_loaders = dict(), dict()
    for dataset_config in config['datasets']:
        train_data, test_data = datasets.load_dataset(
                dataset_config['name'], dataset_config['kwargs'])
        train_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=n_workers)
        test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=n_workers)
        train_loaders[dataset_config['task_id']] = train_loader
        test_loaders[dataset_config['task_id']] = test_loader
    log_utils.print_datasets_info(train_loaders, test_loaders)
    return train_loaders, test_loaders


def import_models(config, checkpoints_logdir, device):
    """Import model from configuration file
    """
    model_manager = models.ModelManager(checkpoints_logdir, config['task_ids'])
    model, last_ckpt = model_manager.load_model(
            model_name=config['models']['name'],
            model_weights=config['models']['weights'],
            model_kwargs=config['models']['kwargs'])
    model = model.to(device)
    log_utils.print_model_info(model, last_ckpt)
    return model, model_manager


def import_losses_and_metrics(config):
    """Import losses and metrics from configuration
    """
    loss_dict = dict([(loss['task_id'], loss['name'])
                      for loss in config['losses']])
    losses = custom_losses.get_losses(loss_dict)
    metric_dict = dict([(metric['task_id'], metric['name'])
                        for metric in config['metrics']])
    metrics = custom_metrics.get_metrics(metric_dict)
    return losses, metrics


def run(config,
        epochs,
        n_workers=1,
        resume=False):
    """Main runner that dynamically imports and executes other modules
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = config_utils.read_config(config)
    log_utils.print_experiment_info(cfg['experiment'], cfg['out_dir'])

    tensorboard_logdir, checkpoints_logdir = \
        log_utils.prepare_dirs(cfg['experiment'], cfg['out_dir'], resume)

    train_loaders, test_loaders = import_data_loaders(cfg, n_workers)
    model, model_manager = import_models(cfg, checkpoints_logdir, device)
    losses, metrics = import_losses_and_metrics(cfg)
    tensorboard_writer = SummaryWriter(tensorboard_logdir)

    trainer_def = trainers.load_trainer(cfg['trainer']['name'])
    trainer_def(device=device,
                task_ids=cfg['task_ids'],
                train_loaders=train_loaders,
                test_loaders=test_loaders,
                model=model,
                losses=losses,
                metrics=metrics,
                batch_size=cfg['batch_size'],
                tensorboard_writer=tensorboard_writer,
                model_manager=model_manager,
                epochs=epochs,
                **cfg['trainer']['kwargs'])

    tensorboard_writer.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('-n', '--epochs', type=int, default=1)
    parser.add_argument('-t', '--workers', type=int, default=1)
    parser.add_argument('-c', '--resume', action='store_true')

    args = parser.parse_args()
    run(config=args.config,
        epochs=args.epochs,
        n_workers=args.workers,
        resume=args.resume)
