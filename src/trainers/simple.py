import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim

from utils import log_utils


def generate_idx(task_ids, loaders):
    """Generates a random queue of tasks
    """
    reverse_ids = dict(
            [(task_ids[i], i) for i in range(len(task_ids))])
    task_queue = None
    for task_id, loader in loaders.items():
        idx = reverse_ids[task_id]
        if task_queue is None:
            task_queue = np.ones((len(loader),), dtype=int) * idx
        else:
            subqueue = np.ones((len(loader),), dtype=int) * idx
            task_queue = np.concatenate([task_queue, subqueue], axis=0)
    np.random.shuffle(task_queue)
    return task_queue


def train_epoch(model,
                task_ids,
                device,
                train_loaders,
                losses,
                metrics,
                optimizers):
    """Trains the model on all data loaders for an epoch.
    """
    model.train()
    task_queue = generate_idx(task_ids, train_loaders)
    loader_iterators = dict([(k, iter(v)) for k, v in train_loaders.items()])
    train_losses = dict([(k, 0.) for k in task_ids])
    train_metrics = dict([(k, 0.) for k in task_ids])

    pbar = tqdm(desc='  train', total=len(task_queue), ascii=True)
    for idx in task_queue:
        task_id = task_ids[idx]
        data, target = loader_iterators[task_id].next()
        data, target = data.to(device), target.to(device)

        optimizers[task_id].zero_grad()
        output = model(data, task_id=task_id)
        loss = losses[task_id](output, target)
        loss.backward()
        optimizers[task_id].step()

        train_losses[task_id] += loss.sum().item()
        train_metrics[task_id] += metrics[task_id](output, target).item()
        pbar.update()

    for task_id in task_ids:
        train_losses[task_id] /= len(train_loaders[task_id].dataset)
        train_metrics[task_id] /= len(train_loaders[task_id].dataset)
    pbar.close()
    return train_losses, train_metrics


def eval_epoch(model,
               task_ids,
               device,
               test_loaders,
               losses,
               metrics):
    """Evaluate the model on all datasets
    """
    model.eval()
    total_batches = sum([len(loader) for _, loader in test_loaders.items()])
    eval_losses, eval_metrics = dict(), dict()

    pbar = tqdm(desc='  eval ', total=total_batches, ascii=True)
    with torch.no_grad():
        for task_id in task_ids:
            loader = test_loaders[task_id]
            loss, metric = 0, 0
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(device), target.to(device)
                output = model(data, task_id=task_id)
                loss += losses[task_id](output, target).sum().item()
                metric += metrics[task_id](output, target).item()
                pbar.update()

            eval_losses[task_id] = loss / len(loader.dataset)
            eval_metrics[task_id] = metric / len(loader.dataset)

    pbar.close()
    return eval_losses, eval_metrics


def simple(device,
           task_ids,
           train_loaders,
           test_loaders,
           model,
           losses,
           metrics,
           batch_size,
           tensorboard_writer,
           model_manager,
           epochs=1,
           optimizers=None):
    """Simple training function, assigns an optimizer for each task.
    """
    # Load Optimizers
    optimizer_def = getattr(optim, optimizers['method'])
    optimizers_dict = dict()
    for task_id in task_ids:
        task_params = list(model.body.parameters()) + \
                      list(model.heads[task_id].parameters())
        optimizers_dict[task_id] = optimizer_def(
                task_params, **optimizers['kwargs'])

    starting_epoch = model_manager.last_epoch + 1
    for epoch in range(starting_epoch, starting_epoch + epochs):
        log_utils.print_on_epoch_begin(epoch)

        train_losses, train_metrics = train_epoch(
                model=model,
                task_ids=task_ids,
                device=device,
                train_loaders=train_loaders,
                losses=losses,
                metrics=metrics,
                optimizers=optimizers_dict)

        eval_losses, eval_metrics = eval_epoch(
                model=model,
                task_ids=task_ids,
                device=device,
                test_loaders=test_loaders,
                losses=losses,
                metrics=metrics)

        log_utils.print_eval_info(
                train_losses, train_metrics,
                eval_losses, eval_metrics)
        model_manager.save_model(model, eval_losses, epoch)
