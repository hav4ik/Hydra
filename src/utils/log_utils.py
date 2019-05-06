import os
import pandas as pd
import torch
from tabulate import tabulate
from termcolor import colored


def prepare_dirs(experiment_name, out_dir, resume):
    """Prepares the output directory structure; the directory is
       uniquely numbered.
    """
    tensorboard_dir = os.path.join(
            os.path.expanduser(out_dir), 'tensorboard', experiment_name)
    checkpoints_dir = os.path.join(
            os.path.expanduser(out_dir), 'checkpoints', experiment_name)

    if not resume:
        for i in range(1000):
            num_tbrd_dir = tensorboard_dir + '-{:03d}'.format(i)
            num_ckpt_dir = checkpoints_dir + '-{:03d}'.format(i)
            if not os.path.isdir(num_tbrd_dir) and \
               not os.path.isdir(num_ckpt_dir):
                break
        if i == 1000:
            raise NameError(
                    'There are 999 experiments with the same name already.'
                    ' Please use another name for your experiments.')
    else:
        num_tbrd_dir = tensorboard_dir
        num_ckpt_dir = checkpoints_dir

    if not os.path.isdir(num_tbrd_dir):
        os.makedirs(num_tbrd_dir)
    if not os.path.isdir(num_ckpt_dir):
        os.makedirs(num_ckpt_dir)
    return num_tbrd_dir, num_ckpt_dir


def print_experiment_info(experiment, out_dir):
    """Pretty prints the basic information about current experiment
    """
    print('{} {}'.format(
        colored('EXPERIMENT:', 'green'),
        colored(experiment, 'magenta', attrs=['bold'])))
    if torch.cuda.is_available():
        gpuid = torch.cuda.current_device()
        print('  - device: {}'.format(torch.cuda.get_device_name(gpuid)))
    else:
        print('  - device: CPU')
    print('  - output: {}'.format(out_dir))


def print_datasets_info(train_loaders, test_loaders):
    """Pretty prints the basic information about loaded datasets
    """
    print(colored('\nDATASETS:', 'green'))
    for task_id in train_loaders.keys():
        ntrain = len(train_loaders[task_id].dataset)
        ntest = len(test_loaders[task_id].dataset)
        print('  - {}: {} train, {} test'.format(
            task_id, ntrain, ntest))


def print_model_info(model, last_checkpoint):
    """Pretty prints the basic information about the model
    """
    print(colored('\nMODEL:', 'green'))
    if last_checkpoint is not None:
        print(colored('  [checkpoint]:', 'cyan'), last_checkpoint)
    print(colored('  [module]:', 'cyan'))
    print('  ' + str(model).replace('\n', '\n  '))


def print_eval_info(train_losses, train_metrics, eval_losses, eval_metrics):
    """Pretty prints model evaluation results
    """
    if not isinstance(train_losses, dict) \
            and isinstance(train_metrics, dict) \
            and isinstance(eval_losses, dict) \
            and isinstance(eval_metrics, dict):
        raise TypeError('Parameters `losses` and `metrics` should be '
                        'a dict {"task_id": value}.')
    df = pd.DataFrame({
        'train losses': pd.Series(train_losses),
        'train metrics': pd.Series(train_metrics),
        'eval losses': pd.Series(eval_losses),
        'eval metrics': pd.Series(eval_metrics)
    })
    df.index.name = 'task_ids'
    print(colored('\n  [evaluations]:', 'cyan'))
    table_str = tabulate(df, headers='keys', tablefmt='simple')
    table_str = '  ' + table_str.replace('\n', '\n  ')
    print(table_str)


def print_on_epoch_begin(epoch, counter):
    if counter > 0:
        print(colored(
                '\nEPOCH {} (has not been improved '
                'in {} epochs)'.format(epoch, counter), 'green'))
    else:
        print(colored('\nEPOCH {}'.format(epoch), 'green'))


def print_arbitrary_info(name, s):
    print(colored('\n  [{}]:'.format(name), 'cyan'), str(s))


def print_early_stopping():
    print(colored('\nSTOPPING EARLY', 'green'))
