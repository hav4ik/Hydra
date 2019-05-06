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
                 optimizers,
                 loss_weights):

        super().__init__(
                device, model, model_manager, task_ids, losses, metrics,
                train_loaders, test_loaders, tensorboard_writer)

        optimizer_def = getattr(optim, optimizers['method'])
        self.optimizers = optimizer_def(
                model.parameters(), **optimizers['kwargs'])
        self.loss_weights = dict(
                (k, torch.tensor(v, device=device))
                for k, v in loss_weights.items())

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
        num_branches = dict()
        for idx, (ctrl, block) in enumerate(self.model.control_blocks()):
            n_branches = max(len(ctrl.serving_tasks), 1.)
            num_branches[idx] = torch.tensor(n_branches, device=self.device)

        pbar = tqdm(desc='  train', total=total_batches, ascii=True)
        for batch_idx in range(total_batches):
            self.model.zero_grad()

            # for each task, calculate head grads and accumulate body grads
            for task_idx, task_id in enumerate(self.task_ids):
                data, target = loader_iterators[task_id].next()
                data, target = data.to(self.device), target.to(self.device)

                # do inference with backward
                output = self.model(data, task_id)
                loss = self.losses[task_id](output, target)
                wloss = self.loss_weights[task_id] * loss
                wloss.backward()

                # calculate training metrics
                with torch.no_grad():
                    train_losses_ts[task_id] += loss.sum()
                    train_metrics_ts[task_id] += \
                        self.metrics[task_id](output, target)

            # averaging out body gradients and optimize the body
            for idx, (_, block) in enumerate(self.model.control_blocks()):
                for p in block.parameters():
                    p.grad /= num_branches[idx]
            self.optimizers.step()
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
