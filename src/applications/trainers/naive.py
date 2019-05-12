import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim

from .base_trainer import BaseTrainer
from utils.regularizers import slimming_loss


class Naive(BaseTrainer):
    def __init__(self,
                 device,
                 model,
                 losses,
                 metrics,
                 train_loaders,
                 optimizers,
                 model_manager=None,
                 test_loaders=None,
                 tensorboard_writer=None,
                 slimming=None,
                 patience=None):

        super().__init__(device=device,
                         model=model,
                         losses=losses,
                         metrics=metrics,
                         train_loaders=train_loaders,
                         test_loaders=test_loaders,
                         model_manager=model_manager,
                         tensorboard_writer=tensorboard_writer,
                         patience=patience)

        self.slimming = slimming
        optimizer_def = getattr(optim, optimizers['method'])
        optimizers_dict = dict()
        for task_id in self.task_ids:
            task_params = list(model.parameters(task_id))
            optimizers_dict[task_id] = optimizer_def(
                    task_params, **optimizers['kwargs'])
        self.optimizers = optimizers_dict

    def _generate_idx(self):
        """Generates a random queue of tasks
        """
        reverse_ids = dict(
                [(self.task_ids[i], i) for i in range(len(self.task_ids))])
        task_queue = None
        for task_id, loader in self.train_loaders.items():
            if task_id not in self.task_ids:
                continue
            idx = reverse_ids[task_id]
            if task_queue is None:
                task_queue = np.ones((len(loader),), dtype=int) * idx
            else:
                subqueue = np.ones((len(loader),), dtype=int) * idx
                task_queue = np.concatenate([task_queue, subqueue], axis=0)
        np.random.shuffle(task_queue)
        return task_queue

    def train_epoch(self):
        """Trains the model on all data loaders for an epoch.
        """
        self.model.train()
        task_queue = self._generate_idx()
        loader_iterators = dict(
                [(k, iter(v)) for k, v in self.train_loaders.items()])
        train_losses_ts = dict(
                [(k, torch.tensor(0.).to(self.device)) for k in self.task_ids])
        train_metrics_ts = dict(
                [(k, torch.tensor(0.).to(self.device)) for k in self.task_ids])

        pbar = tqdm(desc='  train', total=len(task_queue), ascii=True)
        for idx in task_queue:
            task_id = self.task_ids[idx]
            data, target = loader_iterators[task_id].next()
            data, target = data.to(self.device), target.to(self.device)

            self.optimizers[task_id].zero_grad()
            output = self.model(data, task_id)
            loss = self.losses[task_id](output, target)
            if self.slimming is not None:
                loss = loss + self.slimming * slimming_loss(self.model)
            loss.backward()
            self.optimizers[task_id].step()

            with torch.no_grad():
                train_losses_ts[task_id] += loss.sum()
                train_metrics_ts[task_id] += \
                    self.metrics[task_id](output, target)
            pbar.update()

        with torch.no_grad():
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
