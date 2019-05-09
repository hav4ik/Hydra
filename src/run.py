from argparse import ArgumentParser
import torch
from tensorboardX import SummaryWriter

from utils import config_utils, log_utils
from utils import losses as custom_losses
from utils import metrics as custom_metrics
import datasets
import models
import trainers


def import_data_loaders(config, n_workers, verbose=1):
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
    log_utils.print_datasets_info(train_loaders, test_loaders, verbose)
    return train_loaders, test_loaders


def import_models(config, checkpoints_logdir, device, verbose=1):
    """Import model from configuration file
    """
    model_manager = models.ModelManager(checkpoints_logdir, config['task_ids'])
    model, last_ckpt = model_manager.load_model(
            model_name=config['models']['name'],
            model_weights=config['models']['weights'],
            model_kwargs=config['models']['kwargs'])
    model = model.to(device)
    log_utils.print_model_info(model, last_ckpt, verbose)
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
        resume=False,
        updates=None,
        verbose=1):
    """Main runner that dynamically imports and executes other modules
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = config_utils.read_config(config)
    if updates is not None:
        cfg = config_utils.update_config(cfg, updates)
    log_utils.print_experiment_info(cfg['experiment'], cfg['out_dir'])

    tensorboard_logdir, checkpoints_logdir = \
        log_utils.prepare_dirs(cfg['experiment'], cfg['out_dir'], resume)

    train_loaders, test_loaders = import_data_loaders(
            cfg, n_workers, verbose)
    model, model_manager = import_models(
            cfg, checkpoints_logdir, device, verbose)
    losses, metrics = import_losses_and_metrics(cfg)
    tensorboard_writer = SummaryWriter(tensorboard_logdir)

    trainer_def = trainers.load_trainer(cfg['trainer']['name'])
    trainer = trainer_def(device=device,
                          model=model,
                          model_manager=model_manager,
                          task_ids=cfg['task_ids'],
                          losses=losses,
                          metrics=metrics,
                          train_loaders=train_loaders,
                          test_loaders=test_loaders,
                          tensorboard_writer=tensorboard_writer,
                          **cfg['trainer']['kwargs'])

    starting_epoch = model_manager.last_epoch + 1
    for epoch in range(starting_epoch, starting_epoch + epochs):
        eval_losses, eval_metrics = trainer.run_epoch(epoch)

        if 'saving_freq' in cfg:
            if (epoch + 1) % cfg['saving_freq'] == 0:
                model_manager.save_model(model, eval_losses, epoch)

        if trainer.early_stop():
            log_utils.print_early_stopping()
            break
    tensorboard_writer.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('-n', '--epochs', type=int, default=1)
    parser.add_argument('-u', '--update', action='append')
    parser.add_argument('-t', '--workers', type=int, default=1)
    parser.add_argument('-c', '--resume', action='store_true')
    parser.add_argument('-v', '--verbose', type=int, default=1)

    args = parser.parse_args()
    run(config=args.config,
        epochs=args.epochs,
        n_workers=args.workers,
        resume=args.resume,
        updates=args.update,
        verbose=args.verbose)
