from tqdm import tqdm
import torch
import torch.optim as optim

from utils import log_utils


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
    loader_iterators = dict([(k, iter(v)) for k, v in train_loaders.items()])
    train_losses_ts = dict(
            [(k, torch.tensor(0.).to(device)) for k in task_ids])
    train_metrics_ts = dict(
            [(k, torch.tensor(0.).to(device)) for k in task_ids])
    total_batches = min([len(loader) for _, loader in train_loaders.items()])
    num_tasks = torch.tensor(len(task_ids)).to(device)

    pbar = tqdm(desc='  train', total=total_batches, ascii=True)
    for batch_idx in range(total_batches):
        # for each task, calculate head grads and accumulate body grads
        for task_id in task_ids:
            data, target = loader_iterators[task_id].next()
            data, target = data.to(device), target.to(device)

            # prepare grads
            model.body.zero_grad()
            model.heads[task_id].zero_grad()

            # do inference with backward
            output = model(data, task_id=task_id)
            loss = losses[task_id](output, target)
            loss.backward()

            # calculate training metrics
            with torch.no_grad():
                train_losses_ts[task_id] += loss.sum()
                train_metrics_ts[task_id] += metrics[task_id](output, target)

        # averaging out body gradients and optimize the body
        for p in model.body.parameters():
            p.grad /= num_tasks
        optimizers['body'].step()

        # optimize heads
        for task_id in task_ids:
            optimizers['head'][task_id].step()

        pbar.update()

    for task_id in task_ids:
        train_losses_ts[task_id] /= len(train_loaders[task_id].dataset)
        train_metrics_ts[task_id] /= len(train_loaders[task_id].dataset)

    train_losses = dict([(k, v.item()) for k, v in train_losses_ts.items()])
    train_metrics = dict([(k, v.item()) for k, v in train_metrics_ts.items()])
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
            loss = torch.tensor(0.).to(device)
            metric = torch.tensor(0.).to(device)
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(device), target.to(device)
                output = model(data, task_id=task_id)
                loss += losses[task_id](output, target).sum()
                metric += metrics[task_id](output, target)
                pbar.update()

            eval_losses[task_id] = loss.item() / len(loader.dataset)
            eval_metrics[task_id] = metric.item() / len(loader.dataset)

    pbar.close()
    return eval_losses, eval_metrics


def averaging(device,
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
    head_optimizers = dict()
    for task_id in task_ids:
        head_optimizers[task_id] = optimizer_def(
                model.heads[task_id].parameters(),
                **optimizers['kwargs'])
    body_optimizers = optimizer_def(
            model.body.parameters(), **optimizers['kwargs'])
    optimizers_dict = {
            'body': body_optimizers, 'head': head_optimizers}

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

        for task_id in task_ids:
            tensorboard_writer.add_scalar(
                    '{}/train/loss'.format(task_id),
                    train_losses[task_id], epoch)
            tensorboard_writer.add_scalar(
                    '{}/train/metric'.format(task_id),
                    train_metrics[task_id], epoch)
            tensorboard_writer.add_scalar(
                    '{}/val/loss'.format(task_id),
                    eval_losses[task_id], epoch)
            tensorboard_writer.add_scalar(
                    '{}/val/metric'.format(task_id),
                    eval_metrics[task_id], epoch)

        model_manager.save_model(model, eval_losses, epoch)
