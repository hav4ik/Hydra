from tqdm import tqdm
import torch
import torch.optim as optim

from .base_trainer import BaseTrainer


class Averaging(BaseTrainer):
    def __init__(self,
                 device,
                 model,
                 model_manager,
                 task_ids,
                 losses,
                 metrics,
                 train_loaders,
                 test_loaders,
                 tensorboard_writer,
                 optimizers):

        super().__init__(
                device, model, model_manager, task_ids, losses, metrics,
                train_loaders, test_loaders, tensorboard_writer)

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
        self.optimizers = optimizers_dict

    def train_epoch(self):
        """Trains the model on all data loaders for an epoch.
        """
        self.model.train()
        loader_iterators = dict([(k, iter(v))
                                 for k, v in self.train_loaders.items()])
        train_losses_ts = dict(
                [(k, torch.tensor(0.).to(self.device)) for k in self.task_ids])
        train_metrics_ts = dict(
                [(k, torch.tensor(0.).to(self.device)) for k in self.task_ids])
        total_batches = min([len(loader)
                             for _, loader in self.train_loaders.items()])
        num_tasks = torch.tensor(len(self.task_ids)).to(self.device)

        pbar = tqdm(desc='  train', total=total_batches, ascii=True)
        for batch_idx in range(total_batches):
            # for each task, calculate head grads and accumulate body grads
            for task_id in self.task_ids:
                data, target = loader_iterators[task_id].next()
                data, target = data.to(self.device), target.to(self.device)

                # prepare grads
                self.model.body.zero_grad()
                self.model.heads[task_id].zero_grad()

                # do inference with backward
                output = self.model(data, task_id=task_id)
                loss = self.losses[task_id](output, target)
                loss.backward()

                # calculate training metrics
                with torch.no_grad():
                    train_losses_ts[task_id] += loss.sum()
                    train_metrics_ts[task_id] += \
                        self.metrics[task_id](output, target)

            # averaging out body gradients and optimize the body
            for p in self.model.body.parameters():
                p.grad /= num_tasks
            self.optimizers['body'].step()

            # optimize heads
            for task_id in self.task_ids:
                self.optimizers['head'][task_id].step()
            pbar.update()

        for task_id in self.task_ids:
            train_losses_ts[task_id] /= \
                len(self.train_loaders[task_id].dataset)
            train_metrics_ts[task_id] /= \
                len(self.train_loaders[task_id].dataset)

        train_losses = dict([(k, v.item())
                             for k, v in train_losses_ts.items()])
        train_metrics = dict([(k, v.item())
                             for k, v in train_metrics_ts.items()])
        pbar.close()
        return train_losses, train_metrics
